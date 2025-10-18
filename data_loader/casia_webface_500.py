import numpy as np
import torch
import os
import random

np.bool = np.bool_
import mxnet as mx
import numbers

from copy import deepcopy

from mxnet import ndarray as nd
from mxnet import recordio
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

from .data_utils import subsample_instances

class FaceDataset500(Dataset):
    def __init__(
        self,
        path_imgrec,
        transform=None,
        num_labels=500,
        min_samples_per_label=50
    ):
        self.transform = transform
        self.target_transform = None
        self.samples = []
        self.targets = []
        self.uq_idxs = np.array([])

        assert path_imgrec
        path_imgidx = path_imgrec[:-4] + ".idx"
        self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        s = self.imgrec.read_idx(0)
        header, _ = recordio.unpack(s)

        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = []
            self.id2range = {}
            self.seq_identity = range(int(header.label[0]), int(header.label[1]))
            for identity in self.seq_identity:
                s = self.imgrec.read_idx(identity)
                header, _ = recordio.unpack(s)
                a, b = int(header.label[0]), int(header.label[1])
                self.id2range[identity] = (a, b)
                self.imgidx.extend(range(a, b))
        else:
            self.imgidx = list(self.imgrec.keys)

        label2indices = defaultdict(list)
        for idx in self.imgidx:
            s = self.imgrec.read_idx(idx)
            header, _ = recordio.unpack(s)
            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            label2indices[label].append(idx)

        valid_labels = [lbl for lbl, idxs in label2indices.items()
                        if len(idxs) >= min_samples_per_label]

        if len(valid_labels) < num_labels:
            raise ValueError(
                f"Condition (at least {min_samples_per_label} samples per label) is not met. Only {len(valid_labels)} labels found."
            )
        chosen_labels = random.sample(valid_labels, num_labels)

        self.label_mapping = {old_label: i for i, old_label in enumerate(chosen_labels)}

        filtered_seq = []
        for old_label in chosen_labels:
            indices = label2indices[old_label]
            filtered_seq.extend(indices)

        self.samples = []
        self.targets = []
        for idx in filtered_seq:
            s = self.imgrec.read_idx(idx)
            header, _ = recordio.unpack(s)
            old_label = header.label
            if not isinstance(old_label, numbers.Number):
                old_label = old_label[0]

            new_label = self.label_mapping[old_label]
            self.samples.append([idx, new_label])
            self.targets.append(new_label)

        self.uq_idxs = np.arange(len(self.samples))

    def __getitem__(self, index):
        imgrec_idx, label = self.samples[index]

        s = self.imgrec.read_idx(imgrec_idx)
        _, s = recordio.unpack(s)

        if self.target_transform is not None:
            label = self.target_transform(label)

        _data = mx.image.imdecode(s)
        _data = nd.transpose(_data, axes=(2, 0, 1))
        _data = _data.asnumpy()

        if self.transform is not None:
            img = torch.from_numpy(_data)
            img = to_pil_image(img)
            img = self.transform(img)

        else:
            img = _data
            img = torch.from_numpy(img)

        uq_idx = self.uq_idxs[index]

        return img, label, uq_idx

    def __len__(self):
        return len(self.samples)

    def get_label_counts(self):
        from collections import Counter
        return dict(Counter(self.targets))

def split_dataset_by_ratio(dataset, train_ratio=0.8):
    from copy import deepcopy
    import random
    from collections import defaultdict

    label2indices = defaultdict(list)
    for i, label in enumerate(dataset.targets):
        label2indices[label].append(i)

    train_idxs, test_idxs = [], []
    for label, idxs in label2indices.items():
        random.shuffle(idxs)
        split_point = int(len(idxs) * train_ratio)
        train_idxs.extend(idxs[:split_point])
        test_idxs.extend(idxs[split_point:])

    train_dataset = deepcopy(dataset)
    test_dataset = deepcopy(dataset)

    train_dataset.samples = np.array(train_dataset.samples)[train_idxs].tolist()
    train_dataset.targets = np.array(train_dataset.targets)[train_idxs].tolist()
    train_dataset.uq_idxs = np.arange(len(train_idxs))

    test_dataset.samples = np.array(test_dataset.samples)[test_idxs].tolist()
    test_dataset.targets = np.array(test_dataset.targets)[test_idxs].tolist()
    test_dataset.uq_idxs = np.arange(len(test_idxs))

    return train_dataset, test_dataset

def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()

    dataset.uq_idxs = dataset.uq_idxs[mask]

    dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    dataset.targets = [int(x) for x in dataset.targets]

    return dataset

def subsample_classes(dataset, include_classes=range(250)):

    cls_idxs = [x for x, l in enumerate(dataset.targets) if l in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_indices(train_dataset, val_instances_per_class=5):

    train_classes = list(set(train_dataset.targets))

    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=(val_instances_per_class,))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

def get_casiaface500_datasets(train_transform, test_transform, train_classes=range(250), prop_train_labels=0.5,
                               seed=0, split_train_val=False, args=None, root='/home/compu/Datasets/casia_face/train.rec'):
    np.random.seed(seed)

    dataset = FaceDataset500(transform=None,
                                path_imgrec=os.path.join(args.data_path, 'train.rec'), num_labels=500, min_samples_per_label=50)

    train_dataset, test_dataset = split_dataset_by_ratio(dataset, train_ratio=0.8)

    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    if split_train_val:
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled,
                                                     val_instances_per_class=5)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform

    else:
        train_dataset_labelled_split, val_dataset_labelled_split = None, None

    unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)

    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), np.array(list(unlabelled_indices)))

    unlabelled_classes = list(set(train_dataset.targets) - set(train_classes))
    target_xform_dict = {}
    for i, k in enumerate(list(train_classes) + unlabelled_classes):
        target_xform_dict[k] = i

    test_dataset.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]

    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets

if __name__ == '__main__':
    np.random.seed(0)

    x = get_casiaface500_datasets(None, None, prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))
    print('Printing number of labelled classes...')
    print(len(set(x['train_labelled'].targets)))
    print('Printing total number of classes...')
    print(len(set(x['train_unlabelled'].targets)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')