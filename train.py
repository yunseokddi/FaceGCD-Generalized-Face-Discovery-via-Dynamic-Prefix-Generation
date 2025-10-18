import argparse
import sys
import logging
import warnings
import os

import torch
import yaml
import importlib

import model.dino_vision_transformer as original_dino_vit

from utils.dino_utils import load_pretrained_weights
from parse_config import get_args_parser
from data_loader.data_loaders import DataLoader, get_class_splits
from model.dino_vision_transformer import DINOHead, DINO
from model.ViT_face import face_landmark_4simmin_glo_loc
from trainer.trainer import Trainer

from contextlib import suppress
from datetime import datetime
from functools import partial
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from utils.losses import SupConLoss

from timm import utils
from timm.data import resolve_data_config
from timm.layers import convert_sync_batchnorm, set_fast_norm
from timm.models import create_model, safe_model_name, load_checkpoint
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

warnings.simplefilter("ignore", UserWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


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


def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()

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

    args.image_size = 112
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = 65536

    args.interpolation = 3
    args.crop_pct = 0.875

    # ============ building dataloader ============
    data_loader, labeled_class_num = DataLoader(args, _logger).get_dataloader(retun_labeled_num=True)

    args.num_classes = labeled_class_num

    if args.landmark_cnn:
        landmarkcnn = face_landmark_4simmin_glo_loc(loss_type='CosFace',
                                                    num_patches=196,
                                                    image_size=args.image_size,
                                                    patch_size=8,
                                                    dim=512,
                                                    emb_dropout=0.1)
        landmarkcnn = landmarkcnn.cuda()
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
    load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch,
                            args.patch_size)

    projection_head = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)

    model.adapt_prefix_generator()

    if args.freeze:
        for _, p in feature_extractor.named_parameters():
            p.requires_grad = False

        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    non_freeze_feature_extractor_list = []
    non_freeze_model_list = []
    n_hypernetwork_parameters = 0

    for n, p in feature_extractor.named_parameters():
        if p.requires_grad:
            non_freeze_feature_extractor_list.append(n)

    for n, p in model.named_parameters():
        if n.startswith('prefix_generator'):
            n_hypernetwork_parameters += p.numel()

        if p.requires_grad:
            non_freeze_model_list.append(n)


    n_feature_trainable_extractor_parameters = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)
    n_model_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_head_trainable_parameters = sum(p.numel() for p in projection_head.parameters() if p.requires_grad)

    n_feature_extractor_parameters = sum(p.numel() for p in feature_extractor.parameters())
    n_head_parameters = sum(p.numel() for p in projection_head.parameters())
    n_model_parameters = sum(p.numel() for p in model.parameters())

    if args.grad_checkpointing:
        projection_head.set_grad_checkpointing(enable=True)
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        _logger.info(
            f'Total params:{n_feature_extractor_parameters + n_head_parameters + n_model_parameters}')
        _logger.info(
            f'Total trainable params:{n_model_trainable_parameters + n_head_trainable_parameters}')

        _logger.info(
            f'Total additional params:{n_hypernetwork_parameters}'
        )


    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    model = DINO(model, projection_head)

    feature_extractor.to(device=device)
    model.to(device=device)

    feature_extractor.eval()

    if args.distributed and args.sync_bn:
        args.dist_bn = ''
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                f'and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    amp_autocast = suppress
    loss_scaler = None

    if use_amp == 'apex':
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O0')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        try:
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        except (AttributeError, TypeError):
            assert device.type == 'cuda'
            amp_autocast = torch.cuda.amp.autocast
        if device.type == 'cuda' and amp_dtype == torch.float16:
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info('AMP not enabled. Training in float32.')

        if args.distributed:
            if has_apex and use_amp == 'apex':
                if utils.is_primary(args):
                    _logger.info("Using NVIDIA APEX DistributedDataParallel.")
                model = ApexDDP(model, delay_allreduce=True)

            else:
                if utils.is_primary(args):
                    _logger.info("Using native Torch DistributedDataParallel.")
                model = NativeDDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)

    if args.distributed:
        if has_apex and use_amp == 'apex':
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)

        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)

    train_sup_con_crit = SupConLoss().to(device=device)
    val_sup_con_crit = SupConLoss().to(device=device)

    eval_metric = args.eval_metric if data_loader is not None else 'loss'
    decreasing_metric = eval_metric == 'loss'
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing_metric,
            max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            wandb.init(project='GCD_base_prefix_gen', config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`")

    updates_per_epoch = (len(data_loader['train']) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args, decreasing_metric=decreasing_metric),
        updates_per_epoch=updates_per_epoch,
    )

    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch

    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')

    trainer = Trainer(args, model, feature_extractor, landmarkcnn,data_loader, optimizer, train_sup_con_crit, val_sup_con_crit,
                      lr_scheduler, saver,
                      output_dir, amp_autocast,
                      loss_scaler, start_epoch, num_epochs, device, _logger, has_wandb)

    trainer.train()

    if utils.is_primary(args):
        _logger.info('Finish')


if __name__ == '__main__':
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    get_args_parser(parser)

    main()

    sys.exit(0)
