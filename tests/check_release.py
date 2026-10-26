"""Dependency-free checks for the source tree and all documented launch presets."""

import argparse
import ast
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def module_name(path):
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def exports(module):
    names = set()
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.Assign):
            names.update(n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name))
    return names


def main():
    paths = [p for p in ROOT.rglob("*.py") if ".git" not in p.parts]
    modules = {}
    for path in paths:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        modules[module_name(path)] = ast.parse(source)
        for pattern in (r"gh[pousr]_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{30,}",
                        r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"):
            assert not re.search(pattern, source), f"Possible secret in {path}"

    for name, module in modules.items():
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("pcode"):
                        assert alias.name in modules, f"{name}: missing {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    package = name.split(".") if (ROOT / Path(*name.split(".")) / "__init__.py").is_file() else name.split(".")[:-1]
                    base = ".".join(package[:len(package) - node.level + 1] + ([base] if base else []))
                if not base.startswith("pcode"):
                    continue
                assert base in modules, f"{name}: missing module {base}"
                for alias in node.names:
                    assert alias.name == "*" or base + "." + alias.name in modules or alias.name in exports(modules[base]), (
                        f"{name}: missing symbol {base}.{alias.name}"
                    )

    # Parse the actual argparse function without importing the training libraries.
    params = modules["parameters"]
    get_args = next(n for n in params.body if isinstance(n, ast.FunctionDef) and n.name == "get_args")
    parser_globals = {
        "argparse": argparse, "os": os, "sys": sys, "join": os.path.join,
        "models": SimpleNamespace(cnn_mlp=lambda: None, resnet18_mlp=lambda: None),
        "str2bool": lambda x: str(x).lower() in ("true", "1", "yes", "t", "y"),
    }
    exec(compile(ast.Module([get_args], type_ignores=[]), "parameters.py", "exec"), parser_globals)
    spec = spec_from_file_location("release_launcher", ROOT / "scripts/run_fedscad.py")
    launcher = module_from_spec(spec)
    spec.loader.exec_module(launcher)
    cases = [
        ["--dataset", "cifar10"], ["--dataset", "cifar100"],
        ["--dataset", "tiny_imagenet"],
        ["--dataset", "cifar10", "--partition", "pathological"],
        ["--dataset", "cifar100", "--partition", "pathological"],
        ["--anchor-init", "random"], ["--anchor-init", "paired_class"],
        ["--hc-weight", "0"], ["--kd-weight", "0"],
    ]
    old_argv = sys.argv
    try:
        for case in cases:
            command, _ = launcher.build_command(case + ["--dry-run"])
            sys.argv = command[1:]
            conf = parser_globals["get_args"]()
            assert conf.use_logit_adjust and conf.kd_warmup_rounds == 0
            assert conf.feat_dim >= {"cifar10": 9, "cifar100": 99, "tiny_imagenet": 199}[conf.data]
            if "--partition" in case:
                assert conf.partition_data == "pathological"
    finally:
        sys.argv = old_argv
    print(f"PASS: {len(paths)} Python files, local import closure, secret patterns and {len(cases)} launch presets.")


if __name__ == "__main__":
    main()
