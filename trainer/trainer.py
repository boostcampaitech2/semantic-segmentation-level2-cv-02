import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, label_accuracy_score, add_hist
from utils import Wandb
import wandb


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.use_amp = self.config["trainer"]["use_amp"]

        self.train_metrics = MetricTracker(
            *["train/loss", "train/acc", "train/mIoU", "charts/learning_rate"],
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer,
        )
        self.valid_metrics = MetricTracker(
            *["valid/loss", "valid/acc", "valid/mIoU"], *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        n_class = 11
        best_loss = 9999999
        self.model.train()
        self.train_metrics.reset()

        hist = np.zeros((n_class, n_class))
        scaler = torch.cuda.amp.GradScaler(enabled=True) if self.use_amp else None

        torch.cuda.empty_cache()
        for batch_idx, (data, target, _) in enumerate(self.data_loader):
            data = torch.stack(data)
            target = torch.stack(target).long()

            # gpu 연산을 위해 device 할당
            data, target = data.to(self.device), target.to(self.device)

            # Mixed-Precision
            if self.use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    # inference
                    output = self.model(data)
                    loss = self.criterion(output, target)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                # inference
                output = self.model(data)
                # loss
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

            self.optimizer.zero_grad()

            output = torch.argmax(output, dim=1).detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            hist = add_hist(hist, target, output, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            self.train_metrics.update("train/loss", loss.item())
            self.train_metrics.update("train/acc", acc)
            self.train_metrics.update("train/mIoU", mIoU)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            # step 주기에 따른 loss 출력
            if (batch_idx + 1) % 25 == 0:
                print(
                    f"Epoch {epoch}, Step [{batch_idx+1}/{len(self.data_loader)}], Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}"
                )

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{k: v for k, v in val_log.items()})
        if self.lr_scheduler is not None:
            log.update(**{"charts/learning_rate": self.lr_scheduler.get_last_lr()[0]})
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        n_class = 11
        total_loss = 0
        cnt = 0

        hist = np.zeros((n_class, n_class))

        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(self.valid_data_loader):
                data = torch.stack(data)
                target = torch.stack(target).long()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss
                cnt += 1

                output = torch.argmax(output, dim=1).detach().cpu().numpy()
                target = target.detach().cpu().numpy()

                hist = add_hist(hist, target, output, n_class=n_class)

            self.wandb.show_images_wandb(data[:30], target[:30], output[:30])
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            avrg_loss = total_loss / cnt

            self.valid_metrics.update("valid/acc", acc)
            self.valid_metrics.update("valid/mIoU", mIoU)
            self.valid_metrics.update("valid/loss", avrg_loss.item())

            print(
                f"Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}"
            )

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
