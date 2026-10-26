# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

import functools
import math
import numpy as np
import torch
import torch.distributed as dist
random_state_partition = np.random.RandomState(7)

class Partition(object):

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices
        self.replaced_targets = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        if self.replaced_targets is None:
            return self.data[data_idx]
        else:
            return (self.data[data_idx][0], self.replaced_targets[index])

    def update_replaced_targets(self, replaced_targets):
        self.replaced_targets = replaced_targets
        count = 0
        for index in range(len(replaced_targets)):
            data_idx = self.indices[index]
            if self.replaced_targets[index] == self.data[data_idx][1]:
                count += 1
        return count / len(replaced_targets)

    def clean_replaced_targets(self):
        self.replaced_targets = None

class DataSampler(object):

    def __init__(self, conf, data, data_scheme, data_percentage=None, selected_classes=None):
        self.conf = conf
        self.data = data
        self.data_size = len(self.data)
        self.data_scheme = data_scheme
        self.data_percentage = data_percentage
        self.selected_classes = selected_classes
        self.indices = np.array([x for x in range(0, self.data_size)])
        self.sampled_indices = None

    def sample_indices(self):
        if self.data_scheme == 'random_sampling':
            self.sampled_indices = random_state_partition.choice(self.indices, size=int(self.data_size * self.data_percentage), replace=False)
        elif self.data_scheme == 'class_selection':
            self.sampled_indices = [index for index, target in enumerate(self.data.targets) if target in self.selected_classes]
            if self.data_percentage is not None:
                self.sampled_indices = random_state_partition.choice(self.sampled_indices, size=int(len(self.sampled_indices) * self.data_percentage), replace=False)
        else:
            raise NotImplementedError('此采样策略尚未支持。')

    def use_indices(self, sampled_indices=None):
        assert sampled_indices is not None or self.sampled_indices is not None
        return Partition(self.data, indices=sampled_indices if sampled_indices is not None else self.sampled_indices)

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, conf, data, partition_sizes, partition_type, consistent_indices=True, known_distribution=None):
        self.conf = conf
        self.partition_sizes = partition_sizes
        self.partition_type = partition_type
        self.consistent_indices = consistent_indices
        self.partitions = []
        self.known_distribution = known_distribution
        self.data_size = len(data)
        if type(data) is not Partition:
            self.data = data
            indices = np.array([x for x in range(0, self.data_size)])
        else:
            self.data = data.data
            indices = data.indices
        self.partition_indices(indices)

    def partition_indices(self, indices):
        if self.conf.graph.rank == 0:
            indices = self._create_indices(indices)
        if self.consistent_indices:
            indices = self._get_consistent_indices(indices)
        from_index = 0
        for partition_size in self.partition_sizes:
            to_index = from_index + int(partition_size * self.data_size)
            self.partitions.append(indices[from_index:to_index])
            from_index = to_index
        if self.conf.graph.rank == 0 and len(self.partitions) > 1:
            summarize_partition_distribution(self.partitions, self.data.targets, print_fn=self.conf.logger.log, title=f'Final {self.partition_type} partitions after fixed-size slicing')
            self.class_distribution = record_class_distribution(self.partitions, self.data.targets, print_fn=self.conf.logger.log, plot_heatmap=getattr(self.conf, 'plot_data_distribution', False))

    def _create_indices(self, indices):
        if self.partition_type == 'origin':
            pass
        elif self.partition_type == 'random':
            random_state_partition.shuffle(indices)
        elif self.partition_type == 'sorted':
            indices = [i[0] for i in sorted([(idx, target) for idx, target in enumerate(self.data.targets) if idx in indices], key=lambda x: x[1])]
        elif self.partition_type == 'pathological':
            num_classes = len(np.unique(self.data.targets))
            num_indices = len(indices)
            num_users = len(self.partition_sizes)
            idxs_dict = {}
            for idx, target in enumerate(self.data.targets):
                if idx in indices:
                    target = torch.tensor(target).item()
                    if target not in idxs_dict.keys():
                        idxs_dict[target] = []
                    idxs_dict[target].append(idx)
            indices = build_non_iid_by_pathological(random_state=random_state_partition, idxs_dict=idxs_dict, num_classes=num_classes, num_users=num_users, shard_per_user=self.conf.shard_per_user)
        elif self.partition_type == 'practical':
            num_groups = 2
            num_classes = len(np.unique(self.data.targets))
            num_indices = len(indices)
            num_users = len(self.partition_sizes)
            label2indices = {}
            for idx, target in enumerate(self.data.targets):
                if idx in indices:
                    if target not in label2indices.keys():
                        label2indices[target] = []
                    label2indices[target].append(idx)
            indices = build_non_iid_by_practical(random_state=random_state_partition, label2indices=label2indices, num_classes=num_classes, num_groups=num_groups)
        elif self.partition_type == 'non_iid_dirichlet':
            num_classes = len(np.unique(self.data.targets))
            num_indices = len(indices)
            n_workers = len(self.partition_sizes)
            list_of_indices = build_non_iid_by_dirichlet(random_state=random_state_partition, indices2targets=np.array([(idx, target) for idx, target in enumerate(self.data.targets) if idx in indices]), non_iid_alpha=self.conf.non_iid_alpha, num_classes=num_classes, num_indices=num_indices, n_workers=n_workers)
            summarize_partition_distribution(list_of_indices, self.data.targets, print_fn=self.conf.logger.log, title=f'Raw Dirichlet partitions before flattening (alpha={self.conf.non_iid_alpha})')
            indices = functools.reduce(lambda a, b: a + b, list_of_indices)
        elif self.partition_type == 'from_known_distribution':
            indices = build_non_iid_by_known_distribution(known_distribution=self.known_distribution, targets=self.data.targets, partition_sizes=self.partition_sizes, data_size=self.data_size)
        else:
            raise NotImplementedError(f'分区方案 {self.partition_type} 尚未实现')
        return indices

    def _get_consistent_indices(self, indices):
        if dist.is_initialized():
            indices = torch.IntTensor(indices)
            dist.broadcast(indices, src=0)
            return list(indices)
        else:
            return indices

    def use(self, partition_ind):
        return Partition(self.data, self.partitions[partition_ind])

def build_non_iid_by_practical(random_state, label2indices, num_classes, num_groups):
    rand_set_all = list(range(num_classes))
    random_state.shuffle(rand_set_all)
    rand_set_all = np.array(rand_set_all).reshape((num_groups, -1))
    indices = []
    for group in range(num_groups):
        curr_indices = []
        for label in rand_set_all[group]:
            curr_indices += label2indices[label]
        random_state.shuffle(curr_indices)
        indices += curr_indices
    return indices

def build_non_iid_by_pathological(random_state, idxs_dict, num_classes, num_users, shard_per_user):
    indices = []
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)
        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x
    rand_set_all = list(range(num_classes)) * shard_per_class
    random_state.shuffle(rand_set_all)
    rand_set_all = np.array(rand_set_all).reshape((num_users, -1))
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = random_state.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        indices += list(np.concatenate(rand_set))
    return indices

def build_non_iid_by_dirichlet(random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers):
    if n_workers <= 0:
        raise ValueError('n_workers 必须大于 0')
    n_auxi_workers = min(10, n_workers)
    random_state.shuffle(indices2targets)
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [n_auxi_workers if idx < num_splits - 1 else n_workers - n_auxi_workers * (num_splits - 1) for idx in range(num_splits)]
    for idx, split_worker_count in enumerate(split_n_workers):
        to_index = from_index + int(split_worker_count / n_workers * num_indices)
        splitted_targets.append(indices2targets[from_index:num_indices if idx == num_splits - 1 else to_index])
        from_index = to_index
    idx_batch = []
    for _targets, _n_workers in zip(splitted_targets, split_n_workers):
        _targets = np.array(_targets)
        _targets_size = len(_targets)
        min_required_size = max(1, int(0.5 * _targets_size / _n_workers))
        min_size = 0
        n_retry = 0
        while min_size < min_required_size and n_retry < 100:
            n_retry += 1
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]
                if len(idx_class) == 0:
                    continue
                proportions = random_state.dirichlet(np.repeat(non_iid_alpha, _n_workers))
                balance_mask = np.array([float(len(idx_j) < _targets_size / _n_workers) for idx_j in _idx_batch])
                if balance_mask.sum() == 0:
                    balance_mask = np.ones(_n_workers, dtype=float)
                proportions = proportions * balance_mask
                if proportions.sum() <= 0:
                    proportions = balance_mask / balance_mask.sum()
                else:
                    proportions = proportions / proportions.sum()
                split_points = (np.cumsum(proportions) * len(idx_class)).astype(int)[:-1]
                _idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(_idx_batch, np.split(idx_class, split_points))]
            sizes = [len(idx_j) for idx_j in _idx_batch]
            min_size = min(sizes) if sizes else 0
        idx_batch += _idx_batch
    return idx_batch

def build_non_iid_by_known_distribution(known_distribution, targets, partition_sizes, data_size):
    weights = torch.zeros(data_size)
    local_idx = []
    _local_idx = []
    for client_id, local_known_distribution in known_distribution.items():
        local_known_distribution_dict = {}
        local_total = int(partition_sizes[client_id] * data_size)
        for class2num in local_known_distribution:
            local_known_distribution_dict[class2num[0]] = class2num[1]
        for idx, _ in enumerate(weights):
            target = targets[idx]
            if weights[idx] != 0:
                weights[idx] = 0.1
            if idx in local_idx:
                weights[idx] = 0
            elif target in local_known_distribution_dict:
                weights[idx] = local_known_distribution_dict.get(target)
        _local_idx = torch.multinomial(weights, local_total).tolist()
        local_idx += _local_idx
    return local_idx

def summarize_partition_distribution(partitions, targets, print_fn, title):
    targets_np = np.array(targets)
    sizes = [len(partition) for partition in partitions]
    if len(sizes) == 0:
        print_fn(f'[PartitionDiag] {title}: no partitions.')
        return
    num_classes = int(len(np.unique(targets_np)))
    class_coverage = np.zeros(num_classes, dtype=np.int64)
    class_totals = np.zeros(num_classes, dtype=np.int64)
    client_class_counts = []
    client_entropies = []
    client_num_classes = []
    for partition in partitions:
        if len(partition) == 0:
            counts = np.zeros(num_classes, dtype=np.int64)
        else:
            labels = targets_np[partition]
            labels, counts_raw = np.unique(labels, return_counts=True)
            counts = np.zeros(num_classes, dtype=np.int64)
            for label, count in zip(labels, counts_raw):
                label = int(label)
                if 0 <= label < num_classes:
                    counts[label] = int(count)
        nonzero = counts > 0
        class_coverage += nonzero.astype(np.int64)
        class_totals += counts
        client_class_counts.append(counts)
        client_num_classes.append(int(nonzero.sum()))
        total = int(counts.sum())
        if total > 0:
            probs = counts[nonzero].astype(np.float64) / float(total)
            entropy = float(-(probs * np.log(probs)).sum())
        else:
            entropy = 0.0
        client_entropies.append(entropy)
    print_fn(f'[PartitionDiag] {title}: clients={len(partitions)}, size min/mean/max={min(sizes)}/{np.mean(sizes):.2f}/{max(sizes)}, classes per client min/mean/max={min(client_num_classes)}/{np.mean(client_num_classes):.2f}/{max(client_num_classes)}, entropy min/mean/max={min(client_entropies):.4f}/{np.mean(client_entropies):.4f}/{max(client_entropies):.4f}')
    print_fn('[PartitionDiag] class coverage (#clients per class): ' + ', '.join((f'{cls}:{int(count)}' for cls, count in enumerate(class_coverage))))
    print_fn('[PartitionDiag] class totals: ' + ', '.join((f'{cls}:{int(count)}' for cls, count in enumerate(class_totals))))
    for client_id, counts in enumerate(client_class_counts):
        dist = ', '.join((f'{cls}:{int(count)}' for cls, count in enumerate(counts) if int(count) > 0))
        if dist == '':
            dist = 'empty'
        print_fn(f'[PartitionDiag] client-{client_id:03d} size={int(counts.sum())} classes={client_num_classes[client_id]} entropy={client_entropies[client_id]:.4f} dist=[{dist}]')

def record_class_distribution(partitions, targets, print_fn, plot_heatmap=False):
    targets_of_partitions = {}
    targets_np = np.array(targets)
    print_fn(f'Start calculating class distribution for {len(partitions)} partitions...')
    for idx, partition in enumerate(partitions):
        if idx % 10 == 0:
            print_fn(f'Processing partition {idx}/{len(partitions)}...')
        unique_elements, counts_elements = np.unique(targets_np[partition], return_counts=True)
        targets_of_partitions[idx] = list(zip(unique_elements, counts_elements))
    print_fn('Finished calculating class distribution.')
    if plot_heatmap:
        try:
            plot_class_distribution(targets_of_partitions, num_classes=len(np.unique(targets_np)), n_clients=len(partitions))
            print_fn("Class distribution heatmap saved to 'class_distribution_heatmap.pdf'")
        except Exception as e:
            print_fn(f'Failed to plot class distribution: {e}')
    return targets_of_partitions

def plot_class_distribution(targets_of_partitions, num_classes=10, n_clients=20):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    heatmap_data = np.zeros((n_clients, num_classes))
    for client_id, distribution in targets_of_partitions.items():
        for cls, count in distribution:
            if cls < num_classes:
                heatmap_data[client_id, cls] = count
    plt.figure(figsize=(20, 10))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(heatmap_data.T, cmap='Blues', cbar_kws={'label': 'Number of Samples'})
    ax.invert_yaxis()
    plt.xlabel('Client ID', fontsize=14)
    plt.ylabel('Class Label', fontsize=14)
    plt.title('Data Heterogeneity Distribution (Heatmap)', fontsize=16)
    plt.tight_layout()
    plt.savefig('class_distribution_heatmap.pdf', format='pdf', bbox_inches='tight')
    plt.close()
if __name__ == '__main__':
    num_classes = 10
    num_indices = 50000
    targets = np.array(functools.reduce(lambda a, b: a + b, [[idx] * int(num_indices / num_classes) for idx in range(num_classes)]))
    partitions = build_non_iid_by_dirichlet(random_state=np.random.RandomState(7), indices2targets=np.array([(idx, target) for idx, target in enumerate(targets)]), non_iid_alpha=1, num_classes=num_classes, num_indices=num_indices, n_workers=20)
    record_class_distribution(partitions, targets, print_fn=print)

def get_imagenet1k_classes(num_overlap_classes, random_state, num_total_classes=100):
    _selected_imagenet_classes = []
    _selected_cifar100_classes = []
    _cifar100_classes = list(cifar100_class_id_2_imagenet1k_class_id.keys())
    while len(_selected_imagenet_classes) < num_overlap_classes:
        random_state.shuffle(_cifar100_classes)
        _cifar100_class = _cifar100_classes[0]
        _imagenet_classes = cifar100_class_id_2_imagenet1k_class_id[_cifar100_class]
        if len(_imagenet_classes) > 0:
            _imagenet_class = random_state.choice(_imagenet_classes, size=1, replace=False)[0]
            if _imagenet_class not in _selected_imagenet_classes:
                _selected_cifar100_classes.append(_cifar100_class)
                _selected_imagenet_classes.append(_imagenet_class)
            if len(_selected_imagenet_classes) == num_overlap_classes:
                break
    if num_overlap_classes < num_total_classes:
        matched_classes = functools.reduce(lambda a, b: a + b, list(cifar100_class_id_2_imagenet1k_class_id.values()))
        unmatched_classes = list(set(range(1000)) - set(matched_classes))
        _selected_imagenet_classes.extend(random_state.choice(unmatched_classes, size=int(num_total_classes - num_overlap_classes), replace=False).tolist())
    return _selected_imagenet_classes
cifar100_class_id_2_imagenet1k_class_id = {0: [], 1: [], 2: [], 3: [295, 294, 296, 297], 4: [337], 5: [564], 6: [309], 7: [302, 301, 304, 303, 300, 307, 587], 8: [444, 671], 9: [440, 720, 737, 898, 907], 10: [659, 809], 11: [], 12: [821, 839, 888], 13: [654, 779, 874], 14: [326, 322, 325], 15: [354], 16: [653], 17: [483], 18: [], 19: [345], 20: [423, 559, 765], 21: [367], 22: [409, 530, 892], 23: [], 24: [314], 25: [], 26: [118, 120, 121, 119], 27: [49], 28: [968], 29: [], 30: [148], 31: [386, 385], 32: [], 33: [], 34: [279, 280, 278, 277], 35: [], 36: [333], 37: [498, 598], 38: [104], 39: [508, 878], 40: [470, 818, 846], 41: [621], 42: [288], 43: [291], 44: [], 45: [123], 46: [], 47: [], 48: [], 49: [970, 980], 50: [673], 51: [947, 992, 992], 52: [], 53: [950], 54: [], 55: [360], 56: [], 57: [], 58: [717], 59: [], 60: [], 61: [923], 62: [], 63: [334], 64: [105], 65: [332, 330], 66: [], 67: [5, 6], 68: [], 69: [657], 70: [], 71: [], 72: [], 73: [4], 74: [], 75: [361], 76: [], 77: [113], 78: [65], 79: [73, 72, 75, 74, 76, 77, 567], 80: [], 81: [829], 82: [], 83: [945], 84: [526, 736, 532], 85: [847, 565], 86: [528, 707], 87: [851], 88: [292], 89: [866], 90: [], 91: [], 92: [], 93: [610, 37, 35, 36], 94: [894], 95: [], 96: [], 97: [272, 271, 269, 270], 98: [], 99: [110, 111, 783]}
cifar100_name_2_imagenet1k_name = {'apple': [], 'aquarium_fish': [], 'baby': [], 'bear': ['American black bear, black bear, Ursus americanus, Euarctos americanus', 'brown bear, bruin, Ursus arctos', 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'sloth bear, Melursus ursinus, Ursus ursinus'], 'beaver': ['beaver'], 'bed': ['four-poster'], 'bee': ['bee'], 'beetle': ['ground beetle, carabid beetle', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'leaf beetle, chrysomelid', 'long-horned beetle, longicorn, longicorn beetle', 'tiger beetle', 'weevil', 'hammer'], 'bicycle': ['bicycle-built-for-two, tandem bicycle, tandem', 'mountain bike, all-terrain bike, off-roader'], 'bottle': ['beer bottle', 'pill bottle', 'pop bottle, soda bottle', 'water bottle', 'wine bottle'], 'bowl': ['mixing bowl', 'soup bowl'], 'boy': [], 'bridge': ['steel arch bridge', 'suspension bridge', 'viaduct'], 'bus': ['minibus', 'school bus', 'trolleybus, trolley coach, trackless trolley'], 'butterfly': ['lycaenid, lycaenid butterfly', 'ringlet, ringlet butterfly', 'sulphur butterfly, sulfur butterfly'], 'camel': ['Arabian camel, dromedary, Camelus dromedarius'], 'can': ['milk can'], 'castle': ['castle'], 'caterpillar': [], 'cattle': ['ox'], 'chair': ['barber chair', 'folding chair', 'rocking chair, rocker'], 'chimpanzee': ['chimpanzee, chimp, Pan troglodytes'], 'clock': ['analog clock', 'digital clock', 'wall clock'], 'cloud': [], 'cockroach': ['cockroach, roach'], 'couch': [], 'crab': ['Dungeness crab, Cancer magister', 'fiddler crab', 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica', 'rock crab, Cancer irroratus'], 'crocodile': ['African crocodile, Nile crocodile, Crocodylus niloticus'], 'cup': ['cup'], 'dinosaur': [], 'dolphin': ['killer whale, killer, orca, grampus, sea wolf, Orcinus orca'], 'elephant': ['African elephant, Loxodonta africana', 'Indian elephant, Elephas maximus'], 'flatfish': [], 'forest': [], 'fox': ['Arctic fox, white fox, Alopex lagopus', 'grey fox, gray fox, Urocyon cinereoargenteus', 'kit fox, Vulpes macrotis', 'red fox, Vulpes vulpes'], 'girl': [], 'hamster': ['hamster'], 'house': ['cinema, movie theater, movie theatre, movie house, picture palace', 'home theater, home theatre'], 'kangaroo': ['wallaby, brush kangaroo'], 'keyboard': ['computer keyboard, keypad', 'typewriter keyboard'], 'lamp': ['candle, taper, wax light', 'spotlight, spot', 'table lamp'], 'lawn_mower': ['lawn mower, mower'], 'leopard': ['leopard, Panthera pardus'], 'lion': ['lion, king of beasts, Panthera leo'], 'lizard': [], 'lobster': ['spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish'], 'man': [], 'maple_tree': [], 'motorcycle': [], 'mountain': ['alp', 'volcano'], 'mouse': ['mouse, computer mouse'], 'mushroom': ['mushroom', 'agaric', 'agaric'], 'oak_tree': [], 'orange': ['orange'], 'orchid': [], 'otter': ['otter'], 'palm_tree': [], 'pear': [], 'pickup_truck': ['pickup, pickup truck'], 'pine_tree': [], 'plain': [], 'plate': ['plate'], 'poppy': [], 'porcupine': ['porcupine, hedgehog'], 'possum': ['koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus'], 'rabbit': ['Angora, Angora rabbit', 'wood rabbit, cottontail, cottontail rabbit'], 'raccoon': [], 'ray': ['electric ray, crampfish, numbfish, torpedo', 'stingray'], 'road': [], 'rocket': ['missile'], 'rose': [], 'sea': [], 'seal': [], 'shark': ['hammerhead, hammerhead shark'], 'shrew': [], 'skunk': ['skunk, polecat, wood pussy'], 'skyscraper': [], 'snail': ['snail'], 'snake': ['sea snake'], 'spider': ['barn spider, Araneus cavaticus', 'black and gold garden spider, Argiope aurantia', 'black widow, Latrodectus mactans', 'garden spider, Aranea diademata', 'tarantula', 'wolf spider, hunting spider', 'frying pan, frypan, skillet'], 'squirrel': [], 'streetcar': ['streetcar, tram, tramcar, trolley, trolley car'], 'sunflower': [], 'sweet_pepper': ['bell pepper'], 'table': ['desk', 'pool table, billiard table, snooker table', 'dining table, board'], 'tank': ['tank, army tank, armored combat vehicle, armoured combat vehicle', 'freight car'], 'telephone': ['dial telephone, dial phone', 'pay-phone, pay-station'], 'television': ['television, television system'], 'tiger': ['tiger, Panthera tigris'], 'tractor': ['tractor'], 'train': [], 'trout': [], 'tulip': [], 'turtle': ['jersey, T-shirt, tee shirt', 'box turtle, box tortoise', 'mud turtle', 'terrapin'], 'wardrobe': ['wardrobe, closet, press'], 'whale': [], 'willow_tree': [], 'wolf': ['coyote, prairie wolf, brush wolf, Canis latrans', 'red wolf, maned wolf, Canis rufus, Canis niger', 'timber wolf, grey wolf, gray wolf, Canis lupus', 'white wolf, Arctic wolf, Canis lupus tundrarum'], 'woman': [], 'worm': ['flatworm, platyhelminth', 'nematode, nematode worm, roundworm', 'screw']}
