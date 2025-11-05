import os
import glob
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

os.makedirs(args.output_dir, exist_ok=True)
save_path = os.path.join(args.output_dir, "results.txt")

with open(args.txt_path, "r") as f:
    validation_video_list = [line.strip() for line in f if line.strip()]

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
        if arr.ndim == 3:  # single frame -> [1,H,W,C]
            arr = arr[None, ...]
        if arr.ndim != 4:
            return None
        return arr.astype("float32") / 255.0
    except Exception:
        return None

def compute_dreamsim_video(gt_video, pred_video, dreamsim_model, dreamsim_preprocess, device):
    """Mean DreamSim distance for an entire video (batched). Inputs: numpy [F,H,W,C] in [0,1]."""
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
        dists = dreamsim_model(pred_batch.squeeze(1), gt_batch.squeeze(1))  # [F] or scalar
        return float(dists.mean().item() if dists.ndim > 0 else dists.item())

def find_prediction_paths(output_dir, original_name):
    """
    Return a sorted list of candidate prediction files for a given GT video.
    Supports:
      - exact flattened name: original.replace('/', '_')
      - numbered variants: <base_noext>_0<ext>, <base_noext>_1<ext>, ...
    """
    base = original_name.replace("/", "_")        # e.g., "dir/foo.mp4" -> "dir_foo.mp4"
    root, ext = os.path.splitext(base)
    candidates = set()

    # exact file (as before)
    p_exact = os.path.join(output_dir, base)
    if os.path.exists(p_exact):
        candidates.add(p_exact)

    # patterns for numbered variants (with or without ext)
    patterns = [
        os.path.join(output_dir, f"{root}_*{ext}") if ext else None,
        os.path.join(output_dir, f"{root}_*"),
    ]
    for pat in patterns:
        if pat is None: 
            continue
        for p in glob.glob(pat):
            if os.path.isfile(p):
                candidates.add(p)

    # sort by numeric suffix if present (…_3.mp4 -> 3), otherwise lexicographic
    def suffix_key(p):
        fname = os.path.basename(p)
        stem, _e = os.path.splitext(fname)
        if "_" in stem:
            maybe = stem.split("_")[-1]
            if maybe.isdigit():
                return (0, int(maybe), fname)
        return (1, 0, fname)

    return sorted(candidates, key=suffix_key)

per_pair_results = []  # store per GT–PRED pair
skipped = []           # (name, reason)

for validation_video in tqdm(validation_video_list):
    gt_video_path = os.path.join(data_root, validation_video)
    gt_video = safe_read_video(gt_video_path)
    if gt_video is None:
        skipped.append((validation_video, f"missing/unreadable GT: {gt_video_path}"))
        continue

    pred_paths = find_prediction_paths(args.output_dir, validation_video)
    if not pred_paths:
        skipped.append((validation_video, "no prediction files found"))
        continue

    for pred_video_path in pred_paths:
        pred_video = safe_read_video(pred_video_path)
        if pred_video is None:
            skipped.append((f"{validation_video} :: {os.path.basename(pred_video_path)}", "missing/unreadable PRED"))
            continue

        if gt_video.shape != pred_video.shape:
            skipped.append((f"{validation_video} :: {os.path.basename(pred_video_path)}",
                            f"shape mismatch GT {gt_video.shape} vs PRED {pred_video.shape}"))
            continue

        F, H, W, C = gt_video.shape
        if C == 1:
            gt_video = gt_video.repeat(1, 1, 1, 3)
            pred_video = pred_video.repeat(1, 1, 1, 3)

        gt = torch.from_numpy(gt_video).permute(0, 3, 1, 2).to(device)     # [F,C,H,W] in [0,1]
        pred = torch.from_numpy(pred_video).permute(0, 3, 1, 2).to(device) # [F,C,H,W] in [0,1]

        with torch.no_grad():
            psnr_val = float(psnr_metric(pred, gt).item())
            ssim_val = float(ssim_metric(pred, gt).item())
            gt_lp = gt[:, :3].mul(2.0).sub(1.0)
            pred_lp = pred[:, :3].mul(2.0).sub(1.0)
            lpips_val = float(lpips_metric(pred_lp, gt_lp).item())
            dreamsim_val = compute_dreamsim_video(gt_video, pred_video, dreamsim_model, dreamsim_preprocess, device)

        per_pair_results.append({
            "gt": validation_video,
            "pred": os.path.basename(pred_video_path),
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "dreamsim": dreamsim_val,
        })

# ---- Write results ----
with open(save_path, "w") as f:
    if per_pair_results:
        # overall averages across ALL GT–PRED pairs
        avg_psnr = sum(r["psnr"] for r in per_pair_results) / len(per_pair_results)
        avg_ssim = sum(r["ssim"] for r in per_pair_results) / len(per_pair_results)
        avg_lpips = sum(r["lpips"] for r in per_pair_results) / len(per_pair_results)
        avg_dreamsim = sum(r["dreamsim"] for r in per_pair_results) / len(per_pair_results)

        current_gt = None
        for r in per_pair_results:
            if r["gt"] != current_gt:
                current_gt = r["gt"]
                f.write(f"\nGT: {current_gt}\n")
            f.write(f"  PRED: {r['pred']}\tPSNR: {r['psnr']:.4f}\tSSIM: {r['ssim']:.4f}\tLPIPS: {r['lpips']:.6f}\tDreamSim: {r['dreamsim']:.6f}\n")

        f.write("\nOVERALL AVERAGES (across all GT–PRED pairs)\n")
        f.write(f"PSNR: {avg_psnr:.4f}\tSSIM: {avg_ssim:.4f}\tLPIPS: {avg_lpips:.6f}\tDreamSim: {avg_dreamsim:.6f}\n")
    else:
        f.write("No valid GT–PRED pairs were found; nothing to average.\n")

    if skipped:
        f.write("\nSKIPPED ENTRIES:\n")
        for name, reason in skipped:
            f.write(f"{name}\t{reason}\n")

print(f"Wrote results to {save_path}")
if skipped:
    print(f"Skipped {len(skipped)} item(s); see reasons in {save_path}")
