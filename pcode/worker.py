# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

from pcode.workers.worker_fedscad import WorkerFedSCAD

def Worker(conf):
    return WorkerFedSCAD(conf)
