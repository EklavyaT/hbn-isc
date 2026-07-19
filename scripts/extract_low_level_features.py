"""
Extract per-second low-level visual + audio features from a movie file.

Output: outputs/movie_features_lowlevel/{movie_stem}_lowlevel.csv
Columns:
    time_sec,
    mean_luminance, luminance_change, motion_energy, spatial_frequency, scene_change_indicator,
    envelope_mean, envelope_peak, spectral_centroid, spectral_rolloff, pitch_mean

Visual features are computed from 30 frames in each second (FPS rounded to nearest int).
Audio features are computed from the corresponding 1-second waveform window.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import cv2
import decord
import librosa
import numpy as np
import pandas as pd

decord.bridge.set_bridge("native")
warnings.filterwarnings("ignore", category=UserWarning)

SCENE_CHANGE_THRESH = 30.0  # mean abs frame diff (0-255 scale) heuristic for hard cut


def extract_audio_to_wav(video_path: Path, target_sr: int = 22050) -> Path:
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-ac", "1", "-ar", str(target_sr),
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    return tmp


def visual_features_per_second(video_path: Path) -> pd.DataFrame:
    vr = decord.VideoReader(str(video_path))
    fps_float = vr.get_avg_fps()
    fps = int(round(fps_float))
    n_frames = len(vr)
    n_seconds = n_frames // fps  # drop trailing partial second

    rows = []
    prev_last_gray = None  # last frame of previous second, for cross-second diff/flow

    for sec in range(n_seconds):
        idx_start = sec * fps
        idx_end = idx_start + fps
        batch = vr.get_batch(list(range(idx_start, idx_end))).asnumpy()  # (fps, H, W, 3)

        gray = np.empty(batch.shape[:3], dtype=np.float32)
        for i in range(batch.shape[0]):
            gray[i] = cv2.cvtColor(batch[i], cv2.COLOR_RGB2GRAY).astype(np.float32)

        mean_lum = float(gray.mean())

        if prev_last_gray is not None:
            stack = np.concatenate([prev_last_gray[None], gray], axis=0)
        else:
            stack = gray
        diffs = np.abs(np.diff(stack, axis=0))
        lum_change = float(diffs.mean())
        scene_change = int((diffs.mean(axis=(1, 2)) > SCENE_CHANGE_THRESH).any())

        flow_mags = []
        prev = prev_last_gray if prev_last_gray is not None else gray[0]
        for i in range(gray.shape[0]):
            cur = gray[i]
            flow = cv2.calcOpticalFlowFarneback(
                prev, cur, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_mags.append(float(mag.mean()))
            prev = cur
        motion_energy = float(np.mean(flow_mags))

        lap_vars = []
        for i in range(gray.shape[0]):
            lap = cv2.Laplacian(gray[i], cv2.CV_32F)
            lap_vars.append(float(lap.var()))
        spatial_freq = float(np.mean(lap_vars))

        rows.append({
            "time_sec": sec,
            "mean_luminance": mean_lum,
            "luminance_change": lum_change,
            "motion_energy": motion_energy,
            "spatial_frequency": spatial_freq,
            "scene_change_indicator": scene_change,
        })

        prev_last_gray = gray[-1]

    return pd.DataFrame(rows)


def audio_features_per_second(video_path: Path, n_seconds: int) -> pd.DataFrame:
    wav_path = extract_audio_to_wav(video_path)
    try:
        sr_target = 22050
        y, sr = librosa.load(str(wav_path), sr=sr_target, mono=True)

        rows = []
        for sec in range(n_seconds):
            start = sec * sr
            end = start + sr
            chunk = y[start:end]
            if chunk.size < sr // 2:
                rows.append({
                    "time_sec": sec,
                    "envelope_mean": np.nan,
                    "envelope_peak": np.nan,
                    "spectral_centroid": np.nan,
                    "spectral_rolloff": np.nan,
                    "pitch_mean": np.nan,
                })
                continue

            rms_full = float(np.sqrt(np.mean(chunk ** 2)))

            win = int(0.1 * sr)
            hop = win
            n_win = max(1, len(chunk) // hop)
            rms_windows = []
            for w in range(n_win):
                seg = chunk[w * hop : w * hop + win]
                if seg.size == 0:
                    continue
                rms_windows.append(float(np.sqrt(np.mean(seg ** 2))))
            envelope_peak = float(max(rms_windows)) if rms_windows else np.nan

            cent = librosa.feature.spectral_centroid(y=chunk, sr=sr)
            roll = librosa.feature.spectral_rolloff(y=chunk, sr=sr, roll_percent=0.85)
            spec_centroid = float(np.mean(cent))
            spec_rolloff = float(np.mean(roll))

            try:
                f0 = librosa.yin(
                    chunk, fmin=80, fmax=400, sr=sr,
                    frame_length=2048,
                )
                voiced = f0[(f0 > 80) & (f0 < 400) & np.isfinite(f0)]
                pitch_mean = float(np.mean(voiced)) if voiced.size > 0 else np.nan
            except Exception:
                pitch_mean = np.nan

            rows.append({
                "time_sec": sec,
                "envelope_mean": rms_full,
                "envelope_peak": envelope_peak,
                "spectral_centroid": spec_centroid,
                "spectral_rolloff": spec_rolloff,
                "pitch_mean": pitch_mean,
            })
        return pd.DataFrame(rows)
    finally:
        try:
            wav_path.unlink()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Path to input mp4")
    parser.add_argument(
        "--out-dir", type=str,
        default="outputs/movie_features_lowlevel",
        help="Directory to write CSV into",
    )
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{video_path.stem}_lowlevel.csv"

    t0 = time.time()
    print(f"[{video_path.name}] extracting visual features ...", flush=True)
    vis = visual_features_per_second(video_path)
    t_vis = time.time() - t0
    print(f"  visual: {len(vis)} rows in {t_vis:.1f}s", flush=True)

    t1 = time.time()
    print(f"[{video_path.name}] extracting audio features ...", flush=True)
    aud = audio_features_per_second(video_path, n_seconds=len(vis))
    t_aud = time.time() - t1
    print(f"  audio: {len(aud)} rows in {t_aud:.1f}s", flush=True)

    df = vis.merge(aud, on="time_sec", how="outer").sort_values("time_sec").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    total = time.time() - t0
    print(f"[{video_path.name}] wrote {out_csv}  ({len(df)} rows, total {total:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
