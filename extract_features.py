import argparse
import torch
import os
import warnings
import importlib
import logging
import model.dino_vision_transformer as original_dino_vit

from data_loader.data_loaders import DataLoader, get_class_splits
from model.ViT_face import face_landmark_4simmin_glo_loc
from utils.dino_utils import load_pretrained_weights
from model.dino_vision_transformer import DINOHead, DINO
from data_loader.augmentations import get_transform
from data_loader.youtube_faces_1000 import YoutubeFaces1000
from data_loader.youtube_faces_2000 import YoutubeFaces2000
from data_loader.youtube_faces_500 import YoutubeFaces500
from torch.utils.data import DataLoader

from timm import utils
from timm.layers import convert_sync_batchnorm, set_fast_norm
from timm.models import load_checkpoint
from tqdm import tqdm

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
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError as e:
    has_functorch = False

warnings.simplefilter("ignore", UserWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

_logger = logging.getLogger('train')

def load_part_checkpoint_landmark(path, model, pretrain_name=['stn', 'output']):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()

    back_remove = list(pretrained_dict.keys())
    for keys in back_remove:
        if 'dummy_orthogonal_classifier' in keys:
            continue
        pretrained_dict[keys.replace('module.', '')] = pretrained_dict.pop(keys)

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrain_name[0] in k or pretrain_name[1] in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)
    model_dict = model.state_dict()
    for name, param in model.named_parameters():
        if pretrain_name[0] in name or pretrain_name[1] in name:
            param.requires_grad = False

def custom_collate_fn(batch, batch_size):
    images, labels, uq_idxs = zip(*batch)

    if len(images) < batch_size:
        padding_size = batch_size - len(images)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    # Model parameters
    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model', default='vit_base_patch16_224_dino', type=str, metavar='MODEL',
                       help='Name of model to train (default: "vit_base_patch16_224")')
    group.add_argument('--feature-extractor', default='my_vit_base_patch16_224', type=str, metavar='MODEL',
                       help='Name of model to train (default: "vit_base_patch16_224")')
    group.add_argument('--pretrained', action='store_true', default=False,
                       help='Start with pretrained version of specified network (if avail)')
    group.add_argument('--pretrained-path', default=None, type=str,
                       help='Load this checkpoint as if they were the pretrained weights (with adaptation).')
    group.add_argument('--pretrained_weights', default=None, type=str,
                       help='Load this checkpoint as if they were the pretrained weights (with adaptation).')
    group.add_argument('--checkpoint_weights', default=None, type=str,
                       help='Load this checkpoint as if they were the checkpoint weights (with adaptation).')
    group.add_argument('--use-ema', dest='use_ema', action='store_true',
                       help='use ema version of weights if present')
    parser.add_argument('--arch', default='vit_base', type=str, help='Architecture')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--experiment', type=str, default='')
    parser.add_argument('--warmup_model_dir',
                        type=str,
                        default=None)
    parser.add_argument('--use_best_model', type=str, default='_best')
    parser.add_argument('--model_name', type=str, default='vpt-model', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset', type=str, default='aircraft', help='options: cifar10, cifar100, scars')
    parser.add_argument('--data-path', default='/home/compu/Datasets', type=str,
                        help='dataset path')
    parser.add_argument('--transform', type=str, default='imagenet')

    # landmark_path
    parser.add_argument('--landmark_cnn', action='store_true', default=False,
                        help='use landmark cnn')
    parser.add_argument('--landmark_path',
                        default='/pretrain_output/part_vit_B_34epoch.pth',
                        type=str,
                        help='Please specify path to the Pretrained landmark CNN.')

    parser.add_argument('--return-embed', action='store_true', default=False,
                        help='return immediate embeddings')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    parser.add_argument('--device', default='cuda', type=str,
                        help="Device (accelerator) to use.")
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--amp-dtype', default='float16', type=str,
                        help='lower precision AMP dtype (default: float16)')
    parser.add_argument('--amp-impl', default='native', type=str,
                        help='AMP impl to use, "native" or "apex" (default: native)')
    parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                        help='Force broadcast buffers for native DDP to off.')
    parser.add_argument('--synchronize-step', action='store_true', default=False,
                        help='torch.cuda.synchronize() end of each step')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--device-modules', default=None, type=str, nargs='+',
                        help="Python imports for device backend modules.")
    parser.add_argument('--save_dir',
                        default='results/',
                        type=str, help='Feature result DIR')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                        help='The number of steps to accumulate gradients (default: 1)')
    parser.add_argument('--grad-checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing through model blocks/stages')
    parser.add_argument('--grad-from-block', type=int, default=11)
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--fuser', default='', type=str,
                        help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
    parser.add_argument('--fast-norm', default=False, action='store_true',
                        help='enable experimental fast-norm')
    parser.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
    parser.add_argument('--head-init-scale', default=None, type=float,
                        help='Head initialization scale')
    parser.add_argument('--head-init-bias', default=None, type=float,
                        help='Head initialization bias value')

    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--sync-bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='reduce',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                       metavar='N N N',
                       help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')

    # Prefix parameters
    group = parser.add_argument_group('Prefix parameters')
    group.add_argument('--prefix_tuning', default=False, action='store_true')
    group.add_argument('--prefix_length', default=10, type=int, )
    group.add_argument('--initializer', default='uniform', type=str, )
    group.add_argument('--layer-embed', action='store_true', default=False,
                       help='add layer embedding to input embeddings')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, f'{args.experiment}')

    utils.setup_default_logging()

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)

    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    args = get_class_splits(args)

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

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

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

    if args.local_rank == 0:
        _logger.info('Creating base directories...')

    # ----------------------
    # INIT SAVE DIRS
    # Create a directory for each class
    # ----------------------
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for fold in ('train', 'test'):
        fold_dir = os.path.join(args.save_dir, fold)
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)

        for t in targets:
            target_dir = os.path.join(fold_dir, f'{t}')
            if not os.path.exists(target_dir):
                # os.mkdir(target_dir, exist_ok=True)
                os.mkdir(target_dir)