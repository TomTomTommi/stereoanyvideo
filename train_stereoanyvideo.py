import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import os
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from munch import DefaultMunch
import json
from pytorch_lightning.lite import LightningLite
from torch.cuda.amp import GradScaler

from stereoanyvideo.train_utils.utils import (
    run_test_eval,
    save_ims_to_tb,
    count_parameters,
)
from stereoanyvideo.train_utils.logger import Logger

from stereoanyvideo.evaluation.core.evaluator import Evaluator
from stereoanyvideo.train_utils.losses import sequence_loss, temporal_loss
import stereoanyvideo.datasets.video_datasets as datasets
from stereoanyvideo.models.core.stereoanyvideo import StereoAnyVideo


def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler"""
    for name, param in model.named_parameters():
        if any([key in name for key in ['depthanything']]):
            param.requires_grad_(False)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    return optimizer, scheduler


def forward_batch(batch, model, args):
    output = {}
    disparities = model(
        batch["img"][:, :, 0],
        batch["img"][:, :, 1],
        iters=args.train_iters,
        test_mode=False,
    )
    num_traj = len(batch["disp"][0])
    for i in range(num_traj):
        seq_loss, metrics = sequence_loss(
            disparities[:, i], -batch["disp"][:, i, 0], batch["valid_disp"][:, i, 0])
        output[f"disp_{i}"] = {"loss": seq_loss / num_traj, "metrics": metrics}

    output["disparity"] = {
        "predictions": torch.cat(
            [disparities[-1, i] for i in range(num_traj)], dim=1).detach(),
    }
    return output

class Lite(LightningLite):
    def run(self, args):
        self.seed_everything(0)

        evaluator = Evaluator()

        eval_vis_cfg = {
            "visualize_interval": 0,  # Use 0 for no visualization
            "exp_dir": args.ckpt_path,
        }
        eval_vis_cfg = DefaultMunch.fromDict(eval_vis_cfg, object())
        evaluator.setup_visualization(eval_vis_cfg)

        model = StereoAnyVideo()
        model.cuda()

        with open(args.ckpt_path + "/meta.json", "w") as file:
            json.dump(vars(args), file, sort_keys=True, indent=4)

        train_loader = datasets.fetch_dataloader(args)
        train_loader = self.setup_dataloaders(train_loader, move_to_device=False)

        logging.info(f"Train loader size:  {len(train_loader)}")

        optimizer, scheduler = fetch_optimizer(args, model)
        print("Parameter Count:", {count_parameters(model)})
        logging.info(f"Parameter Count:  {count_parameters(model)}")
        total_steps = 0
        logger = Logger(model, scheduler, args.ckpt_path)

        folder_ckpts = [
            f
            for f in os.listdir(args.ckpt_path)
            if not os.path.isdir(f) and f.endswith(".pth") and not "final" in f
        ]
        if len(folder_ckpts) > 0:
            ckpt_path = sorted(folder_ckpts)[-1]
            ckpt = self.load(os.path.join(args.ckpt_path, ckpt_path))
            logging.info(f"Loading checkpoint {ckpt_path}")
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            if "optimizer" in ckpt:
                logging.info("Load optimizer")
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                logging.info("Load scheduler")
                scheduler.load_state_dict(ckpt["scheduler"])
            if "total_steps" in ckpt:
                total_steps = ckpt["total_steps"]
                logging.info(f"Load total_steps {total_steps}")

        elif args.restore_ckpt is not None:
            assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(
                ".pt"
            )
            logging.info("Loading checkpoint...")
            strict = True

            state_dict = self.load(args.restore_ckpt)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            model.load_state_dict(state_dict, strict=strict)

            logging.info(f"Done loading checkpoint")
        model, optimizer = self.setup(model, optimizer, move_to_device=False)
        model.cuda()
        model.train()
        model.module.module.freeze_bn()  # We keep BatchNorm frozen

        scaler = GradScaler(enabled=args.mixed_precision)

        should_keep_training = True
        global_batch_num = 0
        epoch = -1
        while should_keep_training:
            epoch += 1

            for i_batch, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                if batch is None:
                    print("batch is None")
                    continue
                for k, v in batch.items():
                    batch[k] = v.cuda()

                assert model.training

                output = forward_batch(batch, model, args)

                loss = 0
                logger.update()
                for k, v in output.items():
                    if "loss" in v:
                        loss += v["loss"]
                        logger.writer.add_scalar(
                            f"live_{k}_loss", v["loss"].item(), total_steps
                        )
                    if "metrics" in v:
                        logger.push(v["metrics"], k)

                if self.global_rank == 0:
                    if len(output) > 1:
                        logger.writer.add_scalar(
                            f"live_total_loss", loss.item(), total_steps
                        )
                    logger.writer.add_scalar(
                        f"learning_rate", optimizer.param_groups[0]["lr"], total_steps
                    )
                    global_batch_num += 1
                self.barrier()
                self.backward(scaler.scale(loss))
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                if total_steps < args.num_steps:
                    scheduler.step()
                scaler.update()
                total_steps += 1

                if self.global_rank == 0:

                    if (total_steps % args.save_steps == 0) or (total_steps == 1 and args.validate_at_start):
                        ckpt_iter = "0" * (6 - len(str(total_steps))) + str(total_steps)
                        save_path = Path(
                            f"{args.ckpt_path}/model_{args.name}_{ckpt_iter}.pth"
                        )

                        save_dict = {
                            "model": model.module.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "total_steps": total_steps,
                        }

                        logging.info(f"Saving file {save_path}")
                        self.save(save_dict, save_path)

                self.barrier()

                if total_steps > args.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        PATH = f"{args.ckpt_path}/{args.name}_final.pth"
        torch.save(model.module.module.state_dict(), PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="StereoAnyVideo", help="name your experiment")
    parser.add_argument("--restore_ckpt", help="restore checkpoint")
    parser.add_argument("--ckpt_path", help="path to save checkpoints")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size used during training."
    )
    parser.add_argument(
        "--train_datasets",
        nargs="+",
        default=["things", "monkaa", "driving"],
        help="training datasets.",
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="max learning rate.")

    parser.add_argument(
        "--num_steps", type=int, default=80000, help="length of training schedule."
    )
    parser.add_argument(
        "--save_steps", type=int, default=3000, help="length of training schedule."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs="+",
        default=[320, 720],
        help="size of the random image crops used during training.",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=12,
        help="number of updates to the disparity field in each forward pass.",
    )
    parser.add_argument(
        "--wdecay", type=float, default=0.00001, help="Weight decay in optimizer."
    )

    parser.add_argument(
        "--sample_len", type=int, default=5, help="length of training video samples"
    )
    parser.add_argument(
        "--validate_at_start", action="store_true", help="validate the model at start"
    )
    parser.add_argument(
        "--evaluate_every_n_epoch",
        type=int,
        default=1,
        help="evaluate every n epoch",
    )

    parser.add_argument(
        "--num_workers", type=int, default=6, help="number of dataloader workers."
    )
    # Validation parameters
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=32,
        help="number of updates to the disparity field in each forward pass during validation.",
    )
    # Data augmentation
    parser.add_argument(
        "--img_gamma", type=float, nargs="+", default=None, help="gamma range"
    )
    parser.add_argument(
        "--saturation_range",
        type=float,
        nargs="+",
        default=None,
        help="color saturation",
    )
    parser.add_argument(
        "--do_flip",
        default=False,
        choices=["h", "v"],
        help="flip the images horizontally or vertically",
    )
    parser.add_argument(
        "--spatial_scale",
        type=float,
        nargs="+",
        default=[0, 0],
        help="re-scale the images randomly",
    )
    parser.add_argument(
        "--noyjitter",
        action="store_true",
        help="don't simulate imperfect rectification",
    )
    args = parser.parse_args()

    Path(args.ckpt_path).mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=logging.INFO,
        filename=args.ckpt_path + '/' + args.name + '.log',
        filemode='a',
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    from pytorch_lightning.strategies import DDPStrategy

    Lite(
        strategy=DDPStrategy(find_unused_parameters=True),
        devices="auto",
        accelerator="gpu",
        precision=32,
    ).run(args)
