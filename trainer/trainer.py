import time
import os
import json

import torch
import torchvision
import numpy as np

from collections import OrderedDict

from timm import utils
from utils.losses import info_nce_logits
from utils.cluster_and_log_utils import log_accs_from_preds
from sklearn.cluster import KMeans


class Trainer(object):
    def __init__(self, args, model, feature_extractor, landmark_cnn, data_loader, optimizer, train_loss_fn,
                 validate_loss_fn,
                 lr_scheduler, saver,
                 output_dir,
                 amp_autocast,
                 loss_scaler, start_epoch, num_epochs, device, _logger, has_wandb):
        self.args = args
        self.model = model
        self.feature_extractor = feature_extractor
        self.landmark_cnn = landmark_cnn
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.train_loss_fn = train_loss_fn
        self.validate_loss_fn = validate_loss_fn
        self.lr_scheduler = lr_scheduler
        self.saver = saver
        self.output_dir = output_dir
        self.amp_autocast = amp_autocast
        self.loss_scaler = loss_scaler
        self.device = device
        self._logger = _logger
        self.has_wandb = has_wandb
        self.info_nce_logits = info_nce_logits

        self.start_epoch = start_epoch
        self.num_epochs = num_epochs

        self.dino_fe = self.args.dino_fe

        self.best_metric = None
        self.best_epoch = None

    def train(self):
        results = []

        for epoch in range(self.start_epoch, self.num_epochs):
            train_metrics = self.train_one_epoch(epoch, self.data_loader['train'], self.optimizer, self.model,
                                 self.feature_extractor)

            if self.args.distributed and self.args.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(self.args):
                    self._logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(self.model, self.args.world_size, self.args.dist_bn == 'reduce')

            if (epoch+1) % self.args.val_interval == 0:
                if self.args.dataset.lower().startswith('casiafaces'):
                    self.model.backbone.change_val_mode()

                else:
                    self.model.module.backbone.change_val_mode()
                val_unlabelled_metrics = self.validate(epoch, self.model, self.feature_extractor, self.data_loader['val_unlabelled'], save_name='Val ACC Unlabelled')
                val_labelled_metrics = self.validate(epoch, self.model, self.feature_extractor,
                                           self.data_loader['val_labelled'], save_name='Val ACC Labelled')

                combined_metrics = OrderedDict()
                for key in val_unlabelled_metrics.keys():
                    combined_metrics[key] = (val_unlabelled_metrics[key] + val_labelled_metrics[key]) / 2

                if utils.is_primary(self.args):
                    self._logger.info(
                        f'Combined Result '
                        f'all_acc: {combined_metrics["all_acc"]:.3f} '
                        f'old_acc: {combined_metrics["old_acc"]:.3f} '
                        f'new_acc: {combined_metrics["new_acc"]:.3f} '
                    )

                if self.args.dataset.lower().startswith('casiafaces'):
                    self.model.backbone.change_val_mode()

                else:
                    self.model.module.backbone.change_val_mode()

                if self.output_dir is not None:
                    lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
                    utils.update_summary(
                        epoch,
                        train_metrics,
                        combined_metrics,
                        filename=os.path.join(self.output_dir, 'summary.csv'),
                        lr=sum(lrs) / len(lrs),
                        write_header=self.best_metric is None,
                        log_wandb=self.args.log_wandb and self.has_wandb,
                    )

                latest_metric = combined_metrics['all_acc']

                if self.saver is not None:
                    self.best_metric, self.best_epoch = self.saver.save_checkpoint(epoch, metric=latest_metric)

                results.append({
                    'epoch': epoch,
                    'train': train_metrics,
                    'validation': combined_metrics,
                })

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch + 1)


        results = {'all': results}
        if self.best_metric is not None:
            results['best'] = results['all'][self.best_epoch - self.start_epoch]
            self._logger.info('*** Best metric: {0} (epoch {1})'.format(self.best_metric, self.best_epoch))

        if utils.is_primary(self.args):
            self._logger.info(f'--result\n{json.dumps(results, indent=4)}')

    def train_one_epoch(self, epoch, loader, optimizer, model, feature_extractor):
        second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        has_no_sync = hasattr(model, "no_sync")
        update_time_m = utils.AverageMeter()
        data_time_m = utils.AverageMeter()
        losses_m = utils.AverageMeter()

        feature_extractor.eval()
        model.train()

        accum_steps = self.args.grad_accum_steps
        last_accum_steps = len(loader) % accum_steps
        updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
        num_updates = epoch * updates_per_epoch
        last_batch_idx = len(loader) - 1
        last_batch_idx_to_accum = len(loader) - last_accum_steps

        data_start_time = update_start_time = time.time()
        optimizer.zero_grad()
        update_sample_count = 0

        for batch_idx, batch in enumerate(loader):
            last_batch = batch_idx == last_batch_idx
            need_update = last_batch or (batch_idx + 1) % accum_steps == 0
            update_idx = batch_idx // accum_steps
            if batch_idx >= last_batch_idx_to_accum:
                accum_steps = last_accum_steps


            input, target, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            target, mask_lab = target.to(self.device), mask_lab.to(self.device).bool()

            input = torch.cat(input, dim=0).to(self.device)

            data_time_m.update(accum_steps * (time.time() - data_start_time))

            def _forward():
                with self.amp_autocast():
                    with torch.no_grad():
                        if self.landmark_cnn is not None:
                            land_label, input_embedding = self.landmark_cnn(input)

                        else:
                            input_embedding = input

                        layer_aware_embeds = self.feature_extractor.get_intermediate_layers(
                                input_embedding)

                        layer_aware_embeds = torch.stack(layer_aware_embeds)

                    features = model(input_embedding, layer_aware_embeds=layer_aware_embeds)

                    if self.args.contrast_unlabel_only:
                        f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                        con_feats = torch.cat([f1, f2], dim=0)
                    else:
                        con_feats = features

                    contrastive_logits, contrastive_labels = self.info_nce_logits(features=con_feats,
                                                                                  args=self.args,
                                                                                  device=self.device)
                    contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels).to(
                        self.device)

                    f1, f2 = [f[mask_lab] for f in features.chunk(2)]
                    sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    sup_con_labels = target[mask_lab]

                    sup_con_loss = self.train_loss_fn(sup_con_feats, labels=sup_con_labels)

                    loss = (1 - self.args.sup_con_weight) * contrastive_loss + self.args.sup_con_weight * sup_con_loss

                    return loss

            def _backward(_loss):
                if self.loss_scaler is not None:
                    self.loss_scaler(
                        _loss,
                        optimizer,
                        clip_grad=self.args.clip_grad,
                        clip_mode=self.args.clip_mode,
                        parameters=model.parameters(),
                        create_graph=second_order,
                        need_update=need_update,
                    )

                else:
                    _loss.backward(create_graph=second_order)
                    if need_update:
                        if self.args.clip_grad is not None:
                            utils.dispatch_clip_grad(
                                model.parameters(),
                                value=self.args.clip_grad,
                                mode=self.args.clip_mode,
                            )
                        optimizer.step()

            if has_no_sync and not need_update:
                with model.no_sync():
                    loss = _forward()
                    _backward(loss)

            else:
                loss = _forward()
                _backward(loss)

            if not self.args.distributed:
                losses_m.update(loss.item() * accum_steps, input.size(0))
            update_sample_count += input.size(0)

            if not need_update:
                data_start_time = time.time()
                continue

            num_updates += 1
            optimizer.zero_grad()

            if self.args.synchronize_step and self.device.type == 'cuda':
                torch.cuda.synchronize()
            time_now = time.time()
            update_time_m.update(time.time() - update_start_time)
            update_start_time = time_now

            if update_idx % self.args.log_interval == 0:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if self.args.distributed:
                    reduced_loss = utils.reduce_tensor(loss.data, self.args.world_size)
                    losses_m.update(reduced_loss.item() * accum_steps, input.size(0))
                    update_sample_count *= self.args.world_size

                if utils.is_primary(self.args):
                    self._logger.info(
                        f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                        f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                        f'Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  '
                        f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                        f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                        f'LR: {lr:.3e}  '
                        f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                    )

                    if self.args.save_images and self.output_dir:
                        torchvision.utils.save_image(
                            input,
                            os.path.join(self.output_dir, 'train-batch-%d.jpg' % batch_idx),
                            padding=0,
                            normalize=True
                        )

            if self.saver is not None and self.args.recovery_interval and (
                    (update_idx + 1) % self.args.recovery_interval == 0):
                self.saver.save_recovery(epoch, batch_idx=update_idx)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            update_sample_count = 0
            data_start_time = time.time()

        if hasattr(optimizer, 'sync_lookahead'):
            optimizer.sync_lookahead()

        return OrderedDict([('loss', losses_m.avg)])

    def validate(self, epoch, model, feature_extractor, loader, save_name):
        model.eval()
        feature_extractor.eval()

        all_feats = []
        targets = np.array([])
        mask = np.array([])

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                input, label, _ = batch
                input = input.to(self.device)

                with self.amp_autocast():
                    if self.landmark_cnn is not None:
                        land_label, input_embedding = self.landmark_cnn(input)

                    else:
                        input_embedding = input
                    layer_aware_embeds = self.feature_extractor.get_intermediate_layers(
                        input_embedding)

                    layer_aware_embeds = torch.stack(layer_aware_embeds)
                    feats = model(input_embedding, layer_aware_embeds=layer_aware_embeds, mode='val')

                    all_feats.append(feats.cpu().numpy())
                    targets = np.append(targets, label.cpu().numpy())
                    mask = np.append(mask, np.array([True if x.item() in range(len(self.args.train_classes))
                                                     else False for x in label]))

        all_feats = np.concatenate(all_feats)
        kmeans = KMeans(n_clusters=self.args.num_labeled_classes + self.args.num_unlabeled_classes, random_state=0).fit(all_feats)
        preds = kmeans.labels_

        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                        T=epoch, eval_funcs=self.args.eval_funcs, save_name=save_name)

        if utils.is_primary(self.args):
            self._logger.info(
                f'{save_name} '
                f'all_acc: {all_acc:.3f} '
                f'old_acc: {old_acc:.3f} '
                f'new_acc: {new_acc:.3f} '
            )

        metrics = OrderedDict([('all_acc', all_acc), ('old_acc', old_acc), ('new_acc', new_acc)])

        return metrics