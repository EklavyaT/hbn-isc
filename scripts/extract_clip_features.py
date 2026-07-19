"""
Extract per-second CLIP image embeddings from a movie file.

For each second t in the movie, sample 1 frame at the midpoint (t + 0.5 sec).
Encode it with CLIP ViT-L/14 (openai/clip-vit-large-patch14) -> 768-dim L2-normalized embedding.

Output: outputs/movie_features_clip/{movie_stem}_clip.csv
Columns: time_sec, clip_0, clip_1, ..., clip_767

Special case: ThePresent.mp4 is cropped to time_sec < 203.133 to match the EEG viewing window.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import decord
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

decord.bridge.set_bridge("native")
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "openai/clip-vit-large-patch14"
TRUNCATE_SEC = {"ThePresent": 203.133}


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_features(video_path: Path, batch_size: int = 16) -> pd.DataFrame:
    vr = decord.VideoReader(str(video_path))
    fps = vr.get_avg_fps()
    n_frames = len(vr)
    n_seconds = n_frames // int(round(fps))

    cap = TRUNCATE_SEC.get(video_path.stem)
    if cap is not None:
        n_seconds = sum(1 for t in range(n_seconds) if t < cap)

    seconds = list(range(n_seconds))
    frame_indices = [min(int(round((t + 0.5) * fps)), n_frames - 1) for t in seconds]

    device = select_device()
    print(f"  device: {device}", flush=True)
    print(f"  loading {MODEL_NAME} ...", flush=True)
    t0 = time.time()
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)

    feats = np.zeros((n_seconds, 768), dtype=np.float32)
    t_enc = time.time()
    with torch.no_grad():
        for b_start in range(0, n_seconds, batch_size):
            b_end = min(b_start + batch_size, n_seconds)
            idx = frame_indices[b_start:b_end]
            batch_np = vr.get_batch(idx).asnumpy()  # (B, H, W, 3)
            pil_imgs = [Image.fromarray(arr) for arr in batch_np]
            inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
            out = model.get_image_features(**inputs)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats[b_start:b_end] = emb.cpu().numpy()
            if b_start == 0:
                t_first = time.time() - t_enc
                print(f"  first batch (B={b_end - b_start}) took {t_first:.2f}s", flush=True)

    print(f"  encoded {n_seconds} frames in {time.time() - t_enc:.1f}s", flush=True)

    cols = ["time_sec"] + [f"clip_{i}" for i in range(768)]
    out = np.concatenate([np.array(seconds, dtype=np.float32)[:, None], feats], axis=1)
    return pd.DataFrame(out, columns=cols).astype({"time_sec": int})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str)
    parser.add_argument("--out-dir", type=str, default="outputs/movie_features_clip")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{video_path.stem}_clip.csv"

    t0 = time.time()
    print(f"[{video_path.name}] extracting CLIP features ...", flush=True)
    df = extract_features(video_path, batch_size=args.batch_size)
    df.to_csv(out_csv, index=False)
    print(f"[{video_path.name}] wrote {out_csv} ({len(df)} rows, total {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
