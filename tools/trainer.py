import time
import torch
import os
from torch import nn
import torch.nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
import torchvision
import matplotlib.pyplot as plt
import shutil
import logging
from torch.utils.tensorboard import SummaryWriter
import tqdm
import datetime
import glob
import lpips
from .model import *
from lion_pytorch import Lion
from .losses import PSNRLoss, PSNRLossPositive, L_color, GradientLoss, FnMSE
from .metrics import PSNR, fPSNR, MSE, fMSE
from .util import calculate_psnr, tensor2img, calculate_fpsnr
from torch import distributed as dist
from .dataset import *


class Trainer:
    def __init__(self, rank, world_size, args, distributed=True):
        print(rank, world_size, args)
        self.args = args
        self.rank = rank
        self.init_writer()
        self.log(f"rank {self.rank}")
        self.log(f"loading configs {self.args}", level="LOG")

        torch.manual_seed(42)
        import random

        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.benchmark = True
        self.init_distributed(rank, world_size)
        self.init_datasets()
        self.init_model()
        self.log(f"model {self.model}")

        self.train()
        self.destroy()

    def init_distributed(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.log("Initializing distributed")
        # os.environ['MASTER_ADDR'] = self.args.distributed_addr
        # os.environ['MASTER_PORT'] = self.args.distributed_port
        # dist.init_process_group("nccl", rank=rank, world_size=world_size)
        if world_size == 1:
            dist.init_process_group(
                "gloo",
                init_method=f"file:///tmp/gloo{time.time()}",
                rank=rank,
                world_size=world_size,
            )
        else:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def find_free_port(self):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            return str(s.getsockname()[1])

    def init_datasets(self):
        self.train_dataset = globals()[self.args.dataset_to_use_train](
            opt=self.args.datasets, stage="train", apply_augs=True
        )
        self.log(f"Initializing dataset {len(self.train_dataset)}", level="LOG")

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            sampler=DistributedSampler(self.train_dataset, rank=self.rank),
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=True,
        )

        self.val_dataloaders = dict()
        for subset in self.args.datasets_to_use_val:
            dataset = subset
            val_dataset = globals()[dataset](
                opt=self.args.datasets, stage="val", apply_augs=False
            )
            val_dataloader = DataLoader(
                dataset=val_dataset,
                sampler=DistributedSampler(val_dataset, rank=self.rank),
                batch_size=1,
                num_workers=self.args.num_workers,
                persistent_workers=True,
            )
            self.val_dataloaders[subset] = val_dataloader

    def init_model(self, model_decoder="UnetDecoder"):
        self.log("Initializing model")

        self.model = PHNet(
            enc_sizes=self.args.model.enc_sizes,
            skips=self.args.model.skips,
            grid_count=self.args.model.grid_counts,
            init_weights=self.args.model.init_weights,
            init_value=self.args.model.init_value,
        ).to(self.rank)

        if self.args.load_pretrained:
            self.log(f"Restoring from checkpoint: {self.args.checkpoint}", level="LOG")
            self.load_checkpoint(self.args.checkpoint)

        self.model_ddp = DDP(
            self.model,
            device_ids=[self.rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        self.lpips_loss = lpips.LPIPS(net="vgg").to(self.rank)
        # self.psnr_loss = PSNRLoss(max_val=1).to(self.rank)
        self.psnr_loss = PSNRLossPositive(max_val=1).to(self.rank)
        self.fnmse_loss = FnMSE(min_area=100.0).to(self.rank)
        self.color_loss = L_color().to(self.rank)
        self.gradient_loss = GradientLoss(loss_f=torch.nn.L1Loss()).to(self.rank)
        # self.optimizer = torch.optim.AdamW(
        #     self.model_ddp.parameters(), lr=self.args.lr)
        self.optimizer = Lion(self.model_ddp.parameters(), lr=self.args.lr)
        self.scaler = GradScaler()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        # self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode='min')

    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            weights = glob.glob("checkpoints/*/*")
            weights.sort(key=os.path.getmtime)
            checkpoint = weights[-1]

        self.model.load_state_dict(
            torch.load(checkpoint, map_location=f"cuda:{self.rank}"), strict=False
        )

    def init_writer(self):
        if self.rank == 0:
            self.timestamp = f"{datetime.datetime.now():%d_%B_%H_%M}"
            os.makedirs(f"{self.args.checkpoint_dir}/{self.timestamp}", exist_ok=True)
            self.log("CHECKPOINT CREATED")
            for src_file in (
                glob.glob("*.py") + glob.glob("config/*.yaml") + glob.glob("tools/*.py")
            ):
                shutil.copy(src_file, f"{self.args.checkpoint_dir}/{self.timestamp}/")
            # self.logger = logging.getLogger()
            # self.logger.setLevel(logging.INFO)
            # fh = logging.FileHandler(f"{self.args.checkpoint_dir}/{self.timestamp}/logs.log")
            # self.logger.addHandler(fh)
            print("LOGGER CREATED")
            self.log("Initializing writer")
            self.writer = SummaryWriter(
                f"{self.args.log_dir}/{self.args.experiment_name}_{self.timestamp}"
            )
            # logging.basicConfig(filename=f"{self.args.checkpoint_dir}/{self.timestamp}/log.txt", level=logging.INFO)

    def train(self):
        self.log(f"training for {self.args.epochs} epochs", level="LOG")
        self.log(f"Number of steps: {len(self.train_dataloader)}", level="LOG")
        psnr_thrsh = 34.0
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            self.train_dataloader.sampler.set_epoch(epoch)
            total_step = epoch * len(self.train_dataloader)

            self.log(f"Training epoch: {epoch}", level="LOG")
            self.model_ddp.train()

            for i, input_dict in tqdm.tqdm(enumerate(self.train_dataloader)):
                self.train_step(input_dict, total_step + i, epoch)
                # self.scheduler.step()

            if not self.args.disable_validation:
                for subset in self.val_dataloaders.keys():
                    val_metrics = self.validate(epoch, subset=subset)
                    if self.rank == 0 and val_metrics["psnr"] >= psnr_thrsh:
                        self.save()
                # if epoch % self.args.save_model_interval == 0:

            # self.scheduler.step()

    def train_step(self, input_dict, step, epoch):
        inputs = input_dict["inputs"].to(self.rank)
        composite = input_dict["comp"].to(self.rank)
        real = input_dict["real"].to(self.rank)
        mask = input_dict["mask"].to(self.rank)
        path = input_dict["img_path"]
        revert_mask = (1 - mask).to(self.rank)
        with autocast(enabled=not self.args.disable_mixed_precision):
            predicted = self.model_ddp(composite, mask)
            predicted_blend = predicted * mask + composite * (1 - mask)
            losses = {}
            losses["PSNR"] = self.psnr_loss(input=predicted, target=real)
            losses["FnMSE"] = self.fnmse_loss(
                pred=predicted, target=real, mask=mask
            ).mean()
            # losses["PSNR"] = self.psnr_loss(input=predicted, target=real)
            # losses["combined"] = torch.nn.MSELoss()(real * mask, predicted * mask) + torch.nn.L1Loss()(real * revert_mask, predicted * revert_mask)
            losses["Gradient"] = self.gradient_loss(real * mask, predicted * mask)
            if epoch > 100:
                losses["Color"] = self.color_loss(predicted).sum()
            losses["LPIPS"] = self.lpips_loss(
                self.normalize(real), self.normalize(predicted)
            ).mean()
            losses["L2"] = torch.nn.MSELoss()(real, predicted)
            losses["L1"] = torch.nn.L1Loss()(real * mask, predicted * mask)

            total_loss = 0.0
            lambda_default = self.args.lambda_losses["default"]
            for key in losses.keys():
                if key in self.args.lambda_losses:
                    lambda_loss = self.args.lambda_losses[key]
                else:
                    lambda_loss = lambda_default
                total_loss += lambda_loss * losses[key]
                self.log(f"loss[{key}]={losses[key]}")
                if self.rank == 0:
                    self.writer.add_scalar(f"loss_{key}", losses[key], step)
            # if self.rank == 0:
            # self.writer.add_scalar(f"learning_rate", self.optimizer.param_groups[-1]['lr'], step)
            # self.writer.add_scalar(f"learning_rate", self.scheduler.get_last_lr(), step)

            self.log(f"total loss:{total_loss}", level="LOG")

        # total_loss.backward()
        # self.optimizer.step()
        # self.optimizer.zero_grad()

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        # if self.rank == 0 and step > 0 and step % self.args.save_model_interval == 0:
        #     self.save()

        if self.rank == 0 and step % self.args.log_image_interval == 0:
            self.log_image_grid(
                step,
                alpha=mask,
                inputs=inputs,
                real=real,
                composite=composite,
                predicted=predicted,
                predicted_blend=predicted_blend,
                real_masked=real * mask,
                predicted_mask=predicted * mask,
            )

    def normalize(self, x):
        return 2 * (x - x.min() / x.max() - x.min()) - 1

    def log_image_grid(self, step, **kwargs):
        for name, tensor in kwargs.items():
            image_grid = torchvision.utils.make_grid(
                tensor[: self.args.log_image_number, ::].detach().cpu()
            )
            self.writer.add_image(f"{name}", image_grid, step)

    def validate(self, step, subset="HCOCO"):
        self.log("validating")

        if self.rank == 0:
            self.log(f"Validating at the start of epoch: {self.epoch}")
            self.model_ddp.eval()
            total_loss, total_count = 0, 0
            psnr_scores = 0
            fpsnr_scores = 0
            mse_scores = 0
            mse_scores_img = 0
            fmse_scores = 0

            fmse_scores_ratio = {"5": 0, "15": 0, "100": 0}
            mse_scores_ratio = {"5": 0, "15": 0, "100": 0}
            ratio_count = {"5": 0, "15": 0, "100": 0}

            val_dataloader = self.val_dataloaders[subset]
            with torch.no_grad():
                with autocast(enabled=not self.args.disable_mixed_precision):
                    for i, input_dict in enumerate(tqdm.tqdm(val_dataloader)):
                        inputs = input_dict["inputs"].to(self.rank)
                        composite = input_dict["comp"].to(self.rank)
                        real = input_dict["real"].to(self.rank)
                        mask = input_dict["mask"].to(self.rank)
                        harmonized = self.model_ddp(composite, mask)
                        blending = mask * harmonized + (1 - mask) * composite
                        blending_img = 255 * blending
                        harmonized_img = tensor2img(harmonized, bit=8)
                        real_img = 255 * real
                        psnr_score = PSNR()(blending_img, real_img, mask)
                        fore_area = torch.sum(mask)
                        fore_ratio = fore_area / (mask.shape[-1] * mask.shape[-2]) * 100
                        mse_score_img = MSE()(blending_img, real_img, mask)
                        fmse_score = fMSE()(blending_img, real_img, mask)
                        fpsnr_score = fPSNR()(blending_img, real_img, mask)
                        if fore_ratio < 5:
                            ratio_count["5"] += 1
                            fmse_scores_ratio["5"] += fmse_score
                            mse_scores_ratio["5"] += mse_score_img
                        elif fore_ratio < 15:
                            ratio_count["15"] += 1
                            fmse_scores_ratio["15"] += fmse_score
                            mse_scores_ratio["15"] += mse_score_img

                        else:
                            ratio_count["100"] += 1
                            fmse_scores_ratio["100"] += fmse_score
                            mse_scores_ratio["100"] += mse_score_img

                        print(
                            f"psnr: {psnr_score}, mse: {mse_score_img}, mse_img: {mse_score_img}, fmse: {fmse_score}"
                        )
                        print(f"ratio: {fore_ratio}, fmse: {fmse_score}")
                        print(ratio_count, fmse_scores_ratio)

                        psnr_scores += psnr_score
                        fpsnr_scores += fpsnr_score
                        mse_scores += mse_score_img
                        mse_scores_img += mse_score_img
                        fmse_scores += fmse_score

                        batch_size = inputs.shape[0]
                        total_count += batch_size
            psnr_scores_mu = psnr_scores / total_count
            fpsnr_scores_mu = fpsnr_scores / total_count
            mse_score_mu = mse_scores / total_count
            fmse_score_mu = fmse_scores / total_count
            mse_score_img_mu = mse_scores_img / total_count

            for k in ratio_count.keys():
                mse_scores_ratio[k] /= ratio_count[k] + 1e-8
                fmse_scores_ratio[k] /= ratio_count[k] + 1e-8
            # avg_loss = total_loss / total_count
            self.log(f"Dataset: {subset}, Validation setresults:", level="LOG")
            self.log(f"Validation set psnr score: {psnr_scores_mu}", level="LOG")
            self.writer.add_scalar(f"{subset} psnr", psnr_scores_mu, step)
            self.log(f"Validation set fpsnr score: {fpsnr_scores_mu}", level="LOG")
            self.writer.add_scalar(f"{subset} fpsnr", fpsnr_scores_mu, step)
            self.log(f"Validation set MSE score: {mse_score_img_mu}", level="LOG")
            self.writer.add_scalar(f"{subset} mse", mse_score_img_mu, step)
            self.log(f"Validation set fMSE score: {fmse_score_mu}", level="LOG")
            self.writer.add_scalar(f"{subset} fmse", fmse_score_mu, step)
            self.log(
                f"PSNR: {psnr_scores_mu}, FPSNR: {fpsnr_scores_mu}, mse img: {mse_score_img_mu}",
                level="LOG",
            )
            self.model_ddp.train()

            return {"psnr": psnr_scores_mu, "mse": mse_score_img_mu}

    def save(self):
        if self.rank == 0:
            self.model.eval()
            os.makedirs(f"{self.args.checkpoint_dir}/{self.timestamp}", exist_ok=True)
            torch.save(
                self.model.state_dict(),
                f"{self.args.checkpoint_dir}/{self.timestamp}/epoch-{self.epoch}.pth",
            )
            self.log("Model saved", level="LOG")

    def destroy(self):
        dist.destroy_process_group()

    def log(self, msg, level="INFO"):
        if self.rank == 0:
            print(f"[GPU{self.rank}] {msg}")
        # if self.rank == 0 and level == "LOG":
        #     self.logger.info(f'[GPU{self.rank}] {msg}')
