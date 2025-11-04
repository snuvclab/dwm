import os
import argparse
import imageio.v3 as iio
from tqdm import tqdm
from PIL import Image

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from dreamsim import dreamsim

parser = argparse.ArgumentParser()
parser.add_argument("--txt_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

save_path = os.path.join(args.output_dir, "results.txt")
with open(args.txt_path, "r") as f:
    lines = f.readlines()
validation_video_list = [line.strip() for line in lines]

data_root = "/virtual_lab/jhb_vclab/world_model/data"

# ---- TorchMetrics setup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
torch.set_grad_enabled(False)

# ---- DreamSim setup ----
dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, cache_dir=".cache")
dreamsim_model = dreamsim_model.to(device).eval()

def safe_read_video(path):
    """Read a video (or animated image) to float32 [0,1], shape [F,H,W,C]; return None on failure."""
    try:
        if not os.path.exists(path):
            return None
        arr = iio.imread(path)
        # imageio returns [H,W,C] for single-frame images; make it [1,H,W,C] for consistency
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4:
            return None
        return arr.astype("float32") / 255.0
    except Exception:
        return None

def compute_dreamsim_video(gt_video, pred_video, dreamsim_model, dreamsim_preprocess, device):
    """
    Compute mean DreamSim distance for an entire video in batch mode.
    Expects gt_video/pred_video as numpy arrays in [0,1], shape [F,H,W,C].
    """
    gt_imgs, pred_imgs = [], []
    F = gt_video.shape[0]
    for f in range(F):
        gt_img = Image.fromarray((gt_video[f] * 255).astype("uint8"))
        pred_img = Image.fromarray((pred_video[f] * 255).astype("uint8"))
        gt_imgs.append(dreamsim_preprocess(gt_img))
        pred_imgs.append(dreamsim_preprocess(pred_img))

    gt_batch = torch.stack(gt_imgs).to(device)   # [F,3,224,224]
    pred_batch = torch.stack(pred_imgs).to(device)

    with torch.no_grad():
        dists = dreamsim_model(pred_batch.squeeze(1), gt_batch.squeeze(1))  # [F] or scalar reduced; assume [F]
        if dists.ndim == 0:
            return float(dists.item())
        return float(dists.mean().item())

per_video_results = []
skipped = []  # list of tuples: (name, reason)

for validation_video in tqdm(validation_video_list):
    gt_video_path = os.path.join(data_root, validation_video)
    pred_video_path = os.path.join(args.output_dir, validation_video.replace("/", "_"))

    gt_video = safe_read_video(gt_video_path)
    if gt_video is None:
        skipped.append((validation_video, f"missing/unreadable GT: {gt_video_path}"))
        continue

    pred_video = safe_read_video(pred_video_path)
    if pred_video is None:
        skipped.append((validation_video, f"missing/unreadable PRED: {pred_video_path}"))
        continue

    if gt_video.shape != pred_video.shape:
        skipped.append((validation_video, f"shape mismatch GT {gt_video.shape} vs PRED {pred_video.shape}"))
        continue

    F, H, W, C = gt_video.shape
    if C == 1:
        gt_video = gt_video.repeat(1, 1, 1, 3)
        pred_video = pred_video.repeat(1, 1, 1, 3)
        C = 3

    gt = torch.from_numpy(gt_video).permute(0, 3, 1, 2).to(device)     # [F,C,H,W] in [0,1]
    pred = torch.from_numpy(pred_video).permute(0, 3, 1, 2).to(device) # [F,C,H,W] in [0,1]

    with torch.no_grad():
        # PSNR / SSIM averaged across frames
        psnr_val = float(psnr_metric(pred, gt).item())
        ssim_val = float(ssim_metric(pred, gt).item())

        # LPIPS expects [-1,1], 3ch
        gt_lp = gt[:, :3].mul(2.0).sub(1.0)
        pred_lp = pred[:, :3].mul(2.0).sub(1.0)
        lpips_val = float(lpips_metric(pred_lp, gt_lp).item())

        # DreamSim (batched)
        dreamsim_val = compute_dreamsim_video(
            gt_video, pred_video, dreamsim_model, dreamsim_preprocess, device
        )

    per_video_results.append({
        "name": validation_video,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "lpips": lpips_val,
        "dreamsim": dreamsim_val,
    })

# ---- Write results ----
with open(save_path, "w") as f:
    if per_video_results:
        avg_psnr = sum(r["psnr"] for r in per_video_results) / len(per_video_results)
        avg_ssim = sum(r["ssim"] for r in per_video_results) / len(per_video_results)
        avg_lpips = sum(r["lpips"] for r in per_video_results) / len(per_video_results)
        avg_dreamsim = sum(r["dreamsim"] for r in per_video_results) / len(per_video_results)

        for r in per_video_results:
            f.write(f"{r['name']}\tPSNR: {r['psnr']:.4f}\tSSIM: {r['ssim']:.4f}\tLPIPS: {r['lpips']:.6f}\tDreamSim: {r['dreamsim']:.6f}\n")
        f.write("\n")
        f.write(f"AVERAGE\tPSNR: {avg_psnr:.4f}\tSSIM: {avg_ssim:.4f}\tLPIPS: {avg_lpips:.6f}\tDreamSim: {avg_dreamsim:.6f}\n")
    else:
        f.write("No valid video pairs were found; nothing to average.\n")

    if skipped:
        f.write("\nSKIPPED ENTRIES:\n")
        for name, reason in skipped:
            f.write(f"{name}\t{reason}\n")

print(f"Wrote results to {save_path}")
if skipped:
    print(f"Skipped {len(skipped)} item(s); see reasons in {save_path}")
