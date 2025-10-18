from .data_utils import MergedDataset

from .cifar import get_cifar_10_datasets, get_cifar_100_datasets
from .herbarium_19 import get_herbarium_datasets
from .stanford_cars import get_scars_datasets
from .imagenet import get_imagenet_100_datasets
from .cub import get_cub_datasets
from .fgvc_aircraft import get_aircraft_datasets
from .youtube_faces_1000 import get_youtubefaces1000_datasets
from .youtube_faces_500 import get_youtubefaces500_datasets
from .youtube_faces_2000 import get_youtubefaces2000_datasets
from .casia_webface_1000 import get_casiaface1000_datasets
from .casia_webface_500 import get_casiaface500_datasets
from .casia_webface_2000 import get_casiaface2000_datasets

from .cifar import subsample_classes as subsample_dataset_cifar
from .herbarium_19 import subsample_classes as subsample_dataset_herb
from .stanford_cars import subsample_classes as subsample_dataset_scars
from .imagenet import subsample_classes as subsample_dataset_imagenet
from .cub import subsample_classes as subsample_dataset_cub
from .fgvc_aircraft import subsample_classes as subsample_dataset_air
from .youtube_faces_1000 import subsample_classes as subsample_dataset_youtube_1000
from .youtube_faces_500 import subsample_classes as subsample_dataset_youtube_500
from .youtube_faces_2000 import subsample_classes as subsample_dataset_youtube_2000
from .casia_webface_1000 import subsample_classes as subsample_dataset_casia_1000
from .casia_webface_500 import subsample_classes as subsample_dataset_casia_500
from .casia_webface_2000 import subsample_classes as subsample_dataset_casia_2000
from .augmentations import get_transform

import pickle
import os
import torch

from copy import deepcopy
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader as DataLoader_pytorch
from timm import utils

sub_sample_class_funcs = {
    'cifar10': subsample_dataset_cifar,
    'cifar100': subsample_dataset_cifar,
    'imagenet_100': subsample_dataset_imagenet,
    'herbarium_19': subsample_dataset_herb,
    'cub': subsample_dataset_cub,
    'aircraft': subsample_dataset_air,
    'scars': subsample_dataset_scars,
    'youtubefaces_1000': subsample_dataset_youtube_1000,
    'youtubefaces_2000': subsample_dataset_youtube_2000,
    'youtubefaces_500': subsample_dataset_youtube_500,
    'casiafaces_1000': subsample_dataset_casia_1000,
    'casiafaces_500': subsample_dataset_casia_500,
    'casiafaces_2000': subsample_dataset_casia_2000,
}

get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'imagenet_100': get_imagenet_100_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets,
    'youtubefaces_1000': get_youtubefaces1000_datasets,
    'youtubefaces_2000': get_youtubefaces2000_datasets,
    'youtubefaces_500': get_youtubefaces500_datasets,
    'casiafaces_1000': get_casiaface1000_datasets,
    'casiafaces_500': get_casiaface500_datasets,
    'casiafaces_2000': get_casiaface2000_datasets,
}


class DataLoader(object):
    def __init__(self, args, _logger, SSK=False, target_transform=None):
        self.args = args
        self.dataset_name = self.args.dataset
        self._logger = _logger
        self.target_transform = target_transform

        if SSK:
            self.train_transform = None
            self.val_transform = None

        else:
            self.train_transform, self.val_transform = get_transform(self.args.transform,
                                                                     image_size=self.args.image_size,
                                                                     args=self.args)

            if not args.not_contrastive:
                self.train_transform = ContrastiveLearningViewGenerator(base_transform=self.train_transform,
                                                                        n_views=self.args.n_views)

        self.train_dataset, self.test_dataset, self.unlabelled_train_examples_test, self.datasets = self.get_datasets(
            self.dataset_name, self.args)

        if args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'cifar100' or args.dataset.lower() == 'imagenet_100' or args.dataset.lower() == 'herbarium_19' or \
                args.dataset.lower() == 'youtubefaces_1000' or args.dataset.lower() == 'youtubefaces_500' or args.dataset.lower() == 'youtubefaces_2000' or \
                args.dataset.lower() == 'casiafaces_1000' or args.dataset.lower() == 'casiafaces_500' or args.dataset.lower() == 'casiafaces_2000':
            self.args.nb_classes = len(set(self.unlabelled_train_examples_test.targets))

        elif args.dataset.lower() == 'scars':
            self.args.nb_classes = len(set(self.unlabelled_train_examples_test.target))

        elif args.dataset.lower() == 'aircraft':
            self.args.nb_classes = len(set(i[1] for i in self.unlabelled_train_examples_test.samples))

        else:
            self.args.nb_classes = len(set(self.unlabelled_train_examples_test.data["target"].values))

        if self.args.only_label:
            self.train_dataset = self.train_dataset.labelled_dataset
            label_len = len(self.train_dataset)
            sampler = torch.utils.data.DistributedSampler(self.train_dataset, shuffle=True)

            if self.args.dataset.lower() == 'cifar10' or self.args.dataset.lower() == 'cifar100' or self.args.dataset.lower() == 'imagenet_100' or self.args.dataset.lower() == 'herbarium_19' or \
                    self.args.dataset.lower() == 'youtubefaces_1000' or self.args.dataset.lower() == 'youtubefaces_500' or args.dataset.lower() == 'youtubefaces_2000' or \
                    args.dataset.lower() == 'casiafaces_1000' or args.dataset.lower() == 'casiafaces_500' or args.dataset.lower() == 'casiafaces_2000':
                if utils.is_primary(args):
                    self._logger.info('-----------------Only Label Version-----------------')
                    self._logger.info(f'Num Labelled Classes: {len(set(self.train_dataset.targets))}')
                    self._logger.info(f'Len labelled set: {len(self.train_dataset)}')

                    self._logger.info(f'Train labelled Classes ID: {set(self.train_dataset.targets)}')
                    self._logger.info(f'Test labelled Classes ID: {set(self.test_dataset.targets)}')

            elif self.args.dataset.lower() == 'cub':
                if utils.is_primary(args):
                    self._logger.info('-----------------Only Label Version-----------------')
                    self._logger.info(f'Num Labelled Classes: {len(set(self.train_dataset.data["target"].values))}')
                    self._logger.info(f'Len labelled set: {len(self.train_dataset)}')

                    self._logger.info(f'Train labelled Classes ID: {set(self.train_dataset.data["target"].values)}')
                    self._logger.info(f'Test labelled Classes ID: {set(self.test_dataset.data["target"].values)}')

            elif self.args.dataset.lower() == 'scars':
                if utils.is_primary(args):
                    self._logger.info('-----------------Only Label Version-----------------')
                    self._logger.info(f'Num Labelled Classes: {len(set(self.train_dataset.target))}')
                    self._logger.info(f'Len labelled set: {len(self.train_dataset)}')

                    self._logger.info(f'Train labelled Classes ID: {set(self.train_dataset.target)}')
                    self._logger.info(f'Test labelled Classes ID: {set(self.test_dataset.target)}')

            elif self.args.dataset.lower() == 'aircraft':
                if utils.is_primary(args):
                    self._logger.info('-----------------Only Label Version-----------------')
                    self._logger.info(f'Num Labelled Classes: {len(set(i[1] for i in self.train_dataset.samples))}')
                    self._logger.info(f'Len labelled set: {len(self.train_dataset)}')

                    self._logger.info(f'Train labelled Classes ID: {set(i[1] for i in self.train_dataset.samples)}')
                    self._logger.info(f'Test labelled Classes ID: {set(i[1] for i in self.test_dataset.samples)}')

        elif self.args.only_unlabel:
            self.train_dataset = self.train_dataset.unlabelled_dataset

            sampler = torch.utils.data.DistributedSampler(self.train_dataset, shuffle=True)

            if self.args.dataset.lower() == 'cifar10' or self.args.dataset.lower() == 'cifar100' or self.args.dataset.lower() == 'imagenet_100' or self.args.dataset.lower() == 'herbarium_19' or \
                    self.args.dataset.lower() == 'youtubefaces_1000' or self.args.dataset.lower() == 'youtubefaces_500' or args.dataset.lower() == 'youtubefaces_2000' or \
                    args.dataset.lower() == 'casiafaces_1000' or args.dataset.lower() == 'casiafaces_500' or args.dataset.lower() == 'casiafaces_2000':
                if utils.is_primary(args):
                    self._logger.info('-----------------Only Unlabel Version-----------------')
                    self._logger.info(f'Num Unlabelled Classes: {len(set(self.train_dataset.targets))}')
                    self._logger.info(f'Len Unlabelled set: {len(self.train_dataset)}')

                    self._logger.info(f'Train Unlabelled Classes ID: {set(self.train_dataset.targets)}')
                    self._logger.info(f'Test Unlabelled Classes ID: {set(self.test_dataset.targets)}')

            elif self.args.dataset.lower() == 'cub':
                if utils.is_primary(args):
                    self._logger.info('-----------------Only Unlabel Version-----------------')
                    self._logger.info(f'Num Unlabelled Classes: {len(set(self.train_dataset.data["target"].values))}')
                    self._logger.info(f'Len Unlabelled set: {len(self.train_dataset)}')

                    self._logger.info(f'Train Unlabelled Classes ID: {set(self.train_dataset.data["target"].values)}')
                    self._logger.info(f'Test Unlabelled Classes ID: {set(self.test_dataset.data["target"].values)}')

            elif self.args.dataset.lower() == 'scars':
                if utils.is_primary(args):
                    self._logger.info('-----------------Only Unlabel Version-----------------')
                    self._logger.info(f'Num Unlabelled Classes: {len(set(self.train_dataset.target))}')
                    self._logger.info(f'Len Unlabelled set: {len(self.train_dataset)}')

                    self._logger.info(f'Train Unlabelled Classes ID: {set(self.train_dataset.target)}')
                    self._logger.info(f'Test Unlabelled Classes ID: {set(self.test_dataset.target)}')

            elif self.args.dataset.lower() == 'aircraft':
                if utils.is_primary(args):
                    self._logger.info('-----------------Only Unlabel Version-----------------')
                    self._logger.info(f'Num Unlabelled Classes: {len(set(i[1] for i in self.train_dataset.samples))}')
                    self._logger.info(f'Len Unlabelled set: {len(self.train_dataset)}')

                    self._logger.info(f'Train Unlabelled Classes ID: {set(i[1] for i in self.train_dataset.samples)}')
                    self._logger.info(f'Test Unlabelled Classes ID: {set(i[1] for i in self.test_dataset.samples)}')


        else:
            label_len = len(self.train_dataset.labelled_dataset)
            unlabelled_len = len(self.train_dataset.unlabelled_dataset)

            sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in
                              range(len(self.train_dataset))]
            sample_weights = torch.DoubleTensor(sample_weights)
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(self.train_dataset))

        self.train_loader = DataLoader_pytorch(self.train_dataset, num_workers=args.workers,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               sampler=sampler, drop_last=True)
        self.test_loader_unlabelled = DataLoader_pytorch(self.unlabelled_train_examples_test,
                                                         num_workers=args.workers,
                                                         batch_size=args.batch_size, shuffle=False, drop_last=True)
        self.test_loader_labelled = DataLoader_pytorch(self.test_dataset, num_workers=args.workers,
                                                       batch_size=args.batch_size, shuffle=False, drop_last=True)

        self.dataloader = {'train': self.train_loader, 'val_unlabelled': self.test_loader_unlabelled,
                           'val_labelled': self.test_loader_labelled}

        args.num_labeled_classes = len(args.train_classes)
        args.num_unlabeled_classes = len(args.unlabeled_classes)

    def get_dataloader(self, retun_labeled_num=False):
        if retun_labeled_num:
            if self.args.only_label:
                if self.args.dataset.lower() == 'cifar10' or self.args.dataset.lower() == 'cifar100' or self.args.dataset.lower() == 'imagenet_100' or self.args.dataset.lower() == 'herbarium_19' or \
                        self.args.dataset.lower() == 'youtubefaces_1000' or self.args.dataset.lower() == 'youtubefaces_500' or self.args.dataset.lower() == 'youtubefaces_2000' or \
                        self.args.dataset.lower() == 'casiafaces_1000' or self.args.dataset.lower() == 'casiafaces_500' or self.args.dataset.lower() == 'casiafaces_2000':
                    return self.dataloader, len(set(self.train_dataset.targets))

                elif self.args.dataset.lower() == 'cub':
                    return self.dataloader, len(set(self.train_dataset.data["target"].values))

                elif self.args.dataset.lower() == 'scars':
                    return self.dataloader, len(set(self.train_dataset.target))

                elif self.args.dataset.lower() == 'aircraft':
                    return self.dataloader, len(set(i[1] for i in self.train_dataset.samples))

            else:
                if self.args.dataset.lower() == 'cifar10' or self.args.dataset.lower() == 'cifar100' or self.args.dataset.lower() == 'imagenet_100' or self.args.dataset.lower() == 'herbarium_19' or \
                        self.args.dataset.lower() == 'youtubefaces_1000' or self.args.dataset.lower() == 'youtubefaces_500' or self.args.dataset.lower() == 'youtubefaces_2000' or \
                        self.args.dataset.lower() == 'casiafaces_1000' or self.args.dataset.lower() == 'casiafaces_500' or self.args.dataset.lower() == 'casiafaces_2000':
                    return self.dataloader, len(set(self.train_dataset.labelled_dataset.targets))

                elif self.args.dataset.lower() == 'cub':
                    return self.dataloader, len(set(self.train_dataset.labelled_dataset.data["target"].values))

                elif self.args.dataset.lower() == 'scars':
                    return self.dataloader, len(set(self.train_dataset.labelled_dataset.target))

                elif self.args.dataset.lower() == 'aircraft':
                    return self.dataloader, len(set(i[1] for i in self.train_dataset.labelled_dataset.samples))

        return self.dataloader

    def return_datasets(self):
        return self.train_dataset, self.test_dataset, self.unlabelled_train_examples_test, self.datasets

    def get_datasets(self, dataset_name, args):
        if dataset_name.lower() not in get_dataset_funcs.keys():
            raise ValueError

        get_dataset_f = get_dataset_funcs[dataset_name.lower()]
        datasets = get_dataset_f(train_transform=self.train_transform, test_transform=self.val_transform,
                                 train_classes=self.args.train_classes,
                                 prop_train_labels=self.args.prop_train_labels,
                                 split_train_val=False,
                                 args=args)

        if args.local_rank == 0:
            if dataset_name.lower() == 'cifar10' or dataset_name.lower() == 'cifar100' or dataset_name.lower() == 'imagenet_100' or dataset_name.lower() == 'herbarium_19' or \
                    self.args.dataset.lower() == 'youtubefaces_1000' or self.args.dataset.lower() == 'youtubefaces_500' or args.dataset.lower() == 'youtubefaces_2000' or \
                    args.dataset.lower() == 'casiafaces_1000' or args.dataset.lower() == 'casiafaces_500' or args.dataset.lower() == 'casiafaces_2000':
                self._logger.info('Printing lens...')
                for k, v in datasets.items():
                    if v is not None:
                        self._logger.info(f'{k}: {len(v)}')

                self._logger.info('Printing labelled and unlabelled overlap...')
                self._logger.info(set.intersection(set(datasets['train_labelled'].uq_idxs),
                                                   set(datasets['train_unlabelled'].uq_idxs)))
                self._logger.info('Printing total instances in train...')
                self._logger.info(
                    len(set(datasets['train_labelled'].uq_idxs)) + len(set(datasets['train_unlabelled'].uq_idxs)))

                self._logger.info(f'Num Labelled Classes: {len(set(datasets["train_labelled"].targets))}')
                self._logger.info(f'Num Unabelled Classes: {len(set(datasets["train_unlabelled"].targets))}')
                self._logger.info(f'Len labelled set: {len(datasets["train_labelled"])}')
                self._logger.info(f'Len unlabelled set: {len(datasets["train_unlabelled"])}')

                self._logger.info(f'Labelled Classes ID: {set(datasets["train_labelled"].targets)}')
                self._logger.info(f'Unlabelled Classes ID: {set(datasets["train_unlabelled"].targets)}')

            elif dataset_name.lower() == 'scars':
                self._logger.info('Printing lens...')
                for k, v in datasets.items():
                    if v is not None:
                        self._logger.info(f'{k}: {len(v)}')

                self._logger.info('Printing labelled and unlabelled overlap...')
                self._logger.info(set.intersection(set(datasets['train_labelled'].uq_idxs),
                                                   set(datasets['train_unlabelled'].uq_idxs)))
                self._logger.info('Printing total instances in train...')
                self._logger.info(
                    len(set(datasets['train_labelled'].uq_idxs)) + len(set(datasets['train_unlabelled'].uq_idxs)))

                self._logger.info(f'Num Labelled Classes: {len(set(datasets["train_labelled"].target))}')
                self._logger.info(f'Num Unabelled Classes: {len(set(datasets["train_unlabelled"].target))}')
                self._logger.info(f'Len labelled set: {len(datasets["train_labelled"])}')
                self._logger.info(f'Len unlabelled set: {len(datasets["train_unlabelled"])}')

                self._logger.info(f'Labelled Classes ID: {set(datasets["train_labelled"].target)}')
                self._logger.info(f'Unlabelled Classes ID: {set(datasets["train_unlabelled"].target)}')

            elif dataset_name.lower() == 'aircraft':
                for k, v in datasets.items():
                    if v is not None:
                        self._logger.info(f'{k}: {len(v)}')

                self._logger.info('Printing labelled and unlabelled overlap...')
                self._logger.info(set.intersection(set(datasets['train_labelled'].uq_idxs),
                                                   set(datasets['train_unlabelled'].uq_idxs)))
                self._logger.info('Printing total instances in train...')
                self._logger.info(
                    len(set(datasets['train_labelled'].uq_idxs)) + len(set(datasets['train_unlabelled'].uq_idxs)))

                self._logger.info(f'Num Labelled Classes: {len(set(i[1] for i in datasets["train_labelled"].samples))}')
                self._logger.info(
                    f'Num Labelled Classes: {len(set(i[1] for i in datasets["train_unlabelled"].samples))}')
                self._logger.info(f'Len labelled set: {len(datasets["train_labelled"])}')
                self._logger.info(f'Len unlabelled set: {len(datasets["train_unlabelled"])}')

                self._logger.info(f'Labelled Classes ID: {set(i[1] for i in datasets["train_labelled"].samples)}')
                self._logger.info(f'Unlabelled Classes ID: {set(i[1] for i in datasets["train_unlabelled"].samples)}')

            else:
                self._logger.info('Printing lens...')
                for k, v in datasets.items():
                    if v is not None:
                        self._logger.info(f'{k}: {len(v)}')

                self._logger.info('Printing labelled and unlabelled overlap...')
                self._logger.info(set.intersection(set(datasets['train_labelled'].uq_idxs),
                                                   set(datasets['train_unlabelled'].uq_idxs)))
                self._logger.info('Printing total instances in train...')
                self._logger.info(
                    len(set(datasets['train_labelled'].uq_idxs)) + len(set(datasets['train_unlabelled'].uq_idxs)))

                self._logger.info(f'Num Labelled Classes: {len(set(datasets["train_labelled"].data["target"].values))}')
                self._logger.info(
                    f'Num Unabelled Classes: {len(set(datasets["train_unlabelled"].data["target"].values))}')
                self._logger.info(f'Len labelled set: {len(datasets["train_labelled"])}')
                self._logger.info(f'Len unlabelled set: {len(datasets["train_unlabelled"])}')

                self._logger.info(f'Labelled Classes ID: {set(datasets["train_labelled"].data["target"].values)}')
                self._logger.info(f'Unlabelled Classes ID: {set(datasets["train_unlabelled"].data["target"].values)}')

        target_transform_dict = {}
        for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
            target_transform_dict[cls] = i
        target_transform = lambda x: target_transform_dict[x]

        for dataset_name, dataset in datasets.items():
            if dataset is not None:
                dataset.target_transform = target_transform

        train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                      unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

        test_dataset = datasets['test']
        unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
        unlabelled_train_examples_test.transform = self.val_transform

        return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


def get_class_splits(args):
    if args.dataset.lower() in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    if args.dataset.lower() == 'cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset.lower() == 'cifar100':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset.lower() == 'tinyimagenet':

        args.image_size = 64
        args.train_classes = range(100)
        args.unlabeled_classes = range(100, 200)

    elif args.dataset.lower() == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(args.osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']

    elif args.dataset.lower() == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset.lower() == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(args.osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset.lower() == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(args.osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset.lower() == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(args.osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)

    elif args.dataset.lower() == 'chinese_traffic_signs':

        args.image_size = 224
        args.train_classes = range(28)
        args.unlabeled_classes = range(28, 56)

    elif args.dataset.lower() == 'youtubefaces_1000':

        args.image_size = 112
        args.train_classes = range(500)
        args.unlabeled_classes = range(500, 1000)

    elif args.dataset.lower() == 'youtubefaces_500':

        args.image_size = 112
        args.train_classes = range(250)
        args.unlabeled_classes = range(250, 500)

    elif args.dataset.lower() == 'youtubefaces_2000':

        args.image_size = 112
        args.train_classes = range(1000)
        args.unlabeled_classes = range(1000, 2000)

    elif args.dataset.lower() == 'casiafaces_1000':
        args.image_size = 112
        args.train_classes = range(500)
        args.unlabeled_classes = range(500, 1000)

    elif args.dataset.lower() == 'casiafaces_500':
        args.image_size = 112
        args.train_classes = range(250)
        args.unlabeled_classes = range(250, 500)

    elif args.dataset.lower() == 'casiafaces_2000':
        args.image_size = 112
        args.train_classes = range(1000)
        args.unlabeled_classes = range(1000, 2000)

    else:

        raise NotImplementedError

    return args


class ContrastiveLearningViewGenerator(object):
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
