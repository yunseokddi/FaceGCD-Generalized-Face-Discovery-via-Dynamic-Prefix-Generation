import argparse
import torch
import warnings
import logging
import os
import sys
import numpy as np
import model.dino_vision_transformer as original_dino_vit

from timm import utils
from timm.models import load_checkpoint
from utils.dino_utils import load_pretrained_weights
from tqdm import tqdm

from torch.utils.data import DataLoader
from parse_config import get_args_parser

from data_loader.augmentations import get_transform
from data_loader.data_loaders import get_class_splits
from data_loader.youtube_faces_1000 import YoutubeFaces1000
from data_loader.youtube_faces_2000 import YoutubeFaces2000
from data_loader.youtube_faces_500 import YoutubeFaces500
from data_loader.casia_webface_1000 import FaceDataset, split_dataset_by_ratio
from data_loader.casia_webface_500 import FaceDataset500
from data_loader.casia_webface_2000 import FaceDataset2000
from model.ViT_face import face_landmark_4simmin_glo_loc
from model.dino_vision_transformer import DINOHead, DINO
from trainer.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from utils.cluster_and_log_utils import log_accs_from_preds

warnings.simplefilter("ignore", UserWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

_NON_IN1K_FILTERS = ['*_in21k', '*_in22k', '*in12k', '*_dino', '*fcmae', '*seer']

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

_logger = logging.getLogger('train')


def load_part_checkpoint_landmark(path, model, pretrain_name=['stn', 'output']):
    # pdb.set_trace()
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    # pretrained_dict=list(pretrained_dict.keys())
    back_remove = list(pretrained_dict.keys())
    for keys in back_remove:
        if 'dummy_orthogonal_classifier' in keys:
            # pdb.set_trace()
            continue
        pretrained_dict[keys.replace('module.', '')] = pretrained_dict.pop(keys)

    # pdb.set_trace()
    # for name_space in pretrain_name:
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrain_name[0] in k or pretrain_name[1] in k}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrain_name[0] in k or pretrain_name[1] in k}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict, strict=True)
    # model.encoder.output_layer.load_state_dict(pretrained_dict,strict=True)
    model_dict = model.state_dict()
    # freeze stn and output layer
    for name, param in model.named_parameters():
        # if not param.requires_grad:
        if pretrain_name[0] in name or pretrain_name[1] in name:
            # pdb.set_trace()
            param.requires_grad = False


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise "check parser"


def custom_collate_fn(batch, batch_size):
    images, labels, uq_idxs = zip(*batch)

    # 만약 배치가 batch_size에 맞지 않는다면 패딩을 추가
    if len(images) < batch_size:
        padding_size = batch_size - len(images)

        # 첫 번째 이미지, 라벨, 인덱스를 사용하여 패딩
        img_padding = [images[0]] * padding_size
        label_padding = [labels[0]] * padding_size
        uq_idx_padding = [uq_idxs[0]] * padding_size

        images += tuple(img_padding)
        labels += tuple(label_padding)
        uq_idxs += tuple(uq_idx_padding)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    uq_idxs = torch.tensor(uq_idxs)

    return images, labels, uq_idxs


def ssk(model, landmark_cnn, feature_extractor, loader, args, device):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            inputs, labels, idxs = batch[:3]
            inputs = inputs.to(device)

            if landmark_cnn is not None:
                land_label, input_embedding = landmark_cnn(inputs)

            else:
                input_embedding = inputs

            layer_aware_embeds = feature_extractor.get_intermediate_layers(
                input_embedding)  # (layer, batch, 197, 768)

            layer_aware_embeds = torch.stack(layer_aware_embeds)

            features = model(input_embedding, layer_aware_embeds=layer_aware_embeds, mode='val')

            all_feats.append(features.cpu().numpy())
            targets = np.append(targets, labels.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                             else False for x in labels]))

    # -----------------------
    # K-MEANS
    # -----------------------
    all_feats = np.concatenate(all_feats)

    all_feats = torch.from_numpy(all_feats).to(device)

    print('Fitting Semi-Supervised K-Means...')
    kmeans = SemiSupKMeans(k=args.K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                           n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                           mode=args.ssk_mode, args=args, device=device)

    kmeans.fit(all_feats)

    preds = kmeans.labels_.cpu().numpy()

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    eval_funcs=args.eval_funcs,
                                                    save_name='SS-K-Means Test ACC Unlabelled', print_output=True)

    return all_acc, old_acc, new_acc, kmeans


def main(args):
    utils.setup_default_logging()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)

    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    utils.random_seed(args.seed, args.rank)

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 112
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = 65536

    args.interpolation = 3
    args.crop_pct = 0.875

    if args.landmark_cnn:
        landmarkcnn = face_landmark_4simmin_glo_loc(loss_type='CosFace',
                                                    num_patches=196,
                                                    image_size=args.image_size,
                                                    patch_size=8,  # 8
                                                    dim=512,  # 512
                                                    emb_dropout=0.1)
        landmarkcnn = landmarkcnn.to(device)
        load_part_checkpoint_landmark(path=args.landmark_path, model=landmarkcnn, pretrain_name=['stn', 'output'])
        landmarkcnn.eval()

    else:
        landmarkcnn = None

    feature_extractor = original_dino_vit.__dict__[args.arch](img_size=[args.image_size],
                                                              patch_size=args.patch_size,
                                                              num_classes=0, )
    load_pretrained_weights(feature_extractor, args.pretrained_weights, args.checkpoint_key, args.arch,
                            args.patch_size)

    model = original_dino_vit.__dict__[args.arch](img_size=[args.image_size],
                                                  patch_size=args.patch_size,
                                                  num_classes=0,
                                                  prefix_tuning=args.prefix_tuning, prefix_length=args.prefix_length,
                                                  prefix_init=args.initializer,
                                                  layer_embed=args.layer_embed,
                                                  batch_size=args.batch_size, )

    projection_head = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)

    model.adapt_prefix_generator()

    model = DINO(model, projection_head)

    load_checkpoint(model, args.checkpoint_weights, args.use_ema)

    feature_extractor.to(device)
    model.to(device)

    model.backbone.change_val_mode()

    # ============ building dataloader ============
    _, val_transform = get_transform('imagenet', image_size=args.image_size, args=args)

    if args.dataset == 'youtubefaces_1000':
        train_dataset = YoutubeFaces1000(transform=val_transform,
                                         root=os.path.join(args.data_path, 'train'))
        test_dataset = YoutubeFaces1000(transform=val_transform,
                                        root=os.path.join(args.data_path, 'test'))
        targets = list(set(train_dataset.targets))

    elif args.dataset == 'youtubefaces_500':
        train_dataset = YoutubeFaces500(transform=val_transform,
                                        root=os.path.join(args.data_path, 'train'))
        test_dataset = YoutubeFaces500(transform=val_transform,
                                       root=os.path.join(args.data_path, 'test'))
        targets = list(set(train_dataset.targets))

    elif args.dataset == 'youtubefaces_2000':
        train_dataset = YoutubeFaces2000(transform=val_transform,
                                         root=os.path.join(args.data_path, 'train'))
        test_dataset = YoutubeFaces2000(transform=val_transform,
                                        root=os.path.join(args.data_path, 'test'))
        targets = list(set(train_dataset.targets))

    elif args.dataset == 'casiafaces_1000':
        dataset = FaceDataset(path_imgrec=os.path.join(args.data_path, 'train.rec'), num_labels=1000,
                              min_samples_per_label=50)

        train_dataset, test_dataset = split_dataset_by_ratio(dataset, train_ratio=0.8)

        train_dataset.transform = val_transform
        test_dataset.transform = val_transform

        targets = list(set(train_dataset.targets))

    elif args.dataset == 'casiafaces_500':
        dataset = FaceDataset500(path_imgrec=os.path.join(args.data_path, 'train.rec'), num_labels=500,
                                 min_samples_per_label=50)

        train_dataset, test_dataset = split_dataset_by_ratio(dataset, train_ratio=0.8)

        train_dataset.transform = val_transform
        test_dataset.transform = val_transform

        targets = list(set(train_dataset.targets))

    elif args.dataset == 'casiafaces_2000':
        dataset = FaceDataset2000(path_imgrec=os.path.join(args.data_path, 'train.rec'), num_labels=2000,
                                 min_samples_per_label=50)

        train_dataset, test_dataset = split_dataset_by_ratio(dataset, train_ratio=0.8)

        train_dataset.transform = val_transform
        test_dataset.transform = val_transform

        targets = list(set(train_dataset.targets))

    else:
        raise NotImplementedError

    if args.local_rank == 0:
        _logger.info("Train dataset num : {}".format(len(train_dataset)))
        _logger.info("Test dataset num : {}".format(len(test_dataset)))
        _logger.info("Targets : {}".format(targets))

    # ----------------------
    # DATALOADER
    # ----------------------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.workers, collate_fn=lambda x: custom_collate_fn(x, args.batch_size))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.workers, collate_fn=lambda x: custom_collate_fn(x, args.batch_size))

    all_acc, old_acc, new_acc, kmeans = ssk(model, landmarkcnn, feature_extractor, train_loader, args, device)

    args.save_dir = os.path.join(args.save_dir, f'{args.experiment}')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cluster_save_path = os.path.join(args.save_dir, 'ss_kmeans_cluster_centres.pt')
    torch.save(kmeans.cluster_centers_, cluster_save_path)


if __name__ == "__main__":
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    get_args_parser(parser)
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    main(args)

    sys.exit(0)
