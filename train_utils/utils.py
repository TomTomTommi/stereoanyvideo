import numpy as np
import os
import torch

import json
import flow_vis
import matplotlib.pyplot as plt

import stereoanyvideo.datasets.video_datasets as datasets
from stereoanyvideo.evaluation.utils.utils import aggregate_and_print_results


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_test_eval(ckpt_path, eval_type, evaluator, model, dataloaders, writer, step):
    for ds_name, dataloader in dataloaders:
        # evaluator.visualize_interval = 1 if not "sintel" in ds_name else 0

        evaluate_result = evaluator.evaluate_sequence(
            model=model.module.module,
            test_dataloader=dataloader,
            writer=writer if not "sintel" in ds_name else None,
            step=step,
            train_mode=True,
        )

        aggregate_result = aggregate_and_print_results(
            evaluate_result,
        )

        save_metrics = [
            "flow_mean_accuracy_5px",
            "flow_mean_accuracy_3px",
            "flow_mean_accuracy_1px",
            "flow_epe_traj_mean",
        ]
        for epe_name in ("epe", "temp_epe", "temp_epe_r"):
            for m in [
                f"disp_{epe_name}_bad_0.5px",
                f"disp_{epe_name}_bad_1px",
                f"disp_{epe_name}_bad_2px",
                f"disp_{epe_name}_bad_3px",
                f"disp_{epe_name}_mean",
            ]:
                save_metrics.append(m)

        for k, v in aggregate_result.items():
            if k in save_metrics:
                writer.add_scalars(
                    f"{ds_name}_{k.rsplit('_', 1)[0]}",
                    {f"{ds_name}_{k}": v},
                    step,
                )

        result_file = os.path.join(
            ckpt_path,
            f"result_{ds_name}_{eval_type}_{step}_mimo.json",
        )
        print(f"Dumping {eval_type} results to {result_file}.")
        with open(result_file, "w") as f:
            json.dump(aggregate_result, f)


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image

    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    image = Image.frombytes("RGB", (w, h), buf.tobytes())
    image = np.asarray(image)
    return image


def save_ims_to_tb(writer, batch, output, total_steps):
    writer.add_image(
        "train_im",
        torch.cat([torch.cat([im[0], im[1]], dim=-1) for im in batch["img"][0]], dim=-2)
        / 255.0,
        total_steps,
        dataformats="CHW",
    )
    if "disp" in batch and len(batch["disp"]) > 0:
        disp_im = [
            (torch.cat([im[0], im[1]], dim=-1) * torch.cat([val[0], val[1]], dim=-1))
            for im, val in zip(batch["disp"][0], batch["valid_disp"][0])
        ]

        disp_im = torch.cat(disp_im, dim=1)

        figure = plt.figure()
        plt.imshow(disp_im.cpu()[0])
        disp_im = fig2data(figure).copy()

        writer.add_image(
            "train_disp",
            disp_im,
            total_steps,
            dataformats="HWC",
        )

    for k, v in output.items():
        if "predictions" in v:
            pred = v["predictions"]
            if k == "disparity":
                figure = plt.figure()
                plt.imshow(pred.cpu()[0])
                pred = fig2data(figure).copy()
                dataformat = "HWC"
            else:
                pred = torch.tensor(
                    flow_vis.flow_to_color(
                        pred.permute(1, 2, 0).cpu().numpy(), convert_to_bgr=False
                    )
                    / 255.0
                )
                dataformat = "HWC"
            writer.add_image(
                f"pred_{k}",
                pred,
                total_steps,
                dataformats=dataformat,
            )
        if "gt" in v:
            gt = v["gt"]
            gt = torch.tensor(
                flow_vis.flow_to_color(
                    gt.permute(1, 2, 0).cpu().numpy(), convert_to_bgr=False
                )
                / 255.0
            )
            dataformat = "HWC"
            writer.add_image(
                f"gt_{k}",
                gt,
                total_steps,
                dataformats=dataformat,
            )
