"""v3 of the HBN-EEG preprocessing pipeline.

Adds a 60/120 Hz notch filter on the original raw before splitting into the
1-100 Hz ICA stream and the 1-20 Hz analysis stream. v2 confirmed that HBN
data ships with line-noise contamination ~14x larger than alpha power, which
caused ICALabel to label 17/20 components as line noise and the pipeline to
destroy brain signal. v3 removes the line-noise contamination at the start.
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import mne
from mne_icalabel import label_components
from pyprep.find_noisy_channels import NoisyChannels

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBJECT_DIR_NAME = "sub_NDARAC904DMU"
SUBJECT_BIDS = "sub-NDARAC904DMU"
TASK = "RestingState"
EEG_PATH = PROJECT_ROOT / "data" / SUBJECT_DIR_NAME / "eeg" / f"{SUBJECT_BIDS}_task-{TASK}_eeg.set"
OUT_PATH = PROJECT_ROOT / "outputs" / f"{SUBJECT_DIR_NAME}_{TASK}_preproc_v3_raw.fif"

REJECT_LABELS = {"eye blink", "muscle artifact", "heart beat", "channel noise", "line noise"}
NOTCH_FREQS = [60.0, 120.0]


def run_step(name, fn, *args, **kwargs):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
    except Exception:
        print(f"[{name}] FAILED", flush=True)
        traceback.print_exc()
        raise
    print(f"[{name}] OK ({time.time() - t0:.1f}s)", flush=True)
    return out


def band_power(raw, lo, hi, fmax=None):
    fmax = fmax if fmax is not None else hi + 5
    psd = raw.compute_psd(fmin=max(0.5, lo - 5), fmax=fmax, n_fft=2048, verbose="WARNING")
    freqs = psd.freqs
    mask = (freqs >= lo) & (freqs <= hi)
    return float(psd.get_data().mean(axis=0)[mask].mean())


def load_raw(path):
    raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="WARNING")
    print(f"  sfreq={raw.info['sfreq']} Hz, n_ch={len(raw.ch_names)}, dur={raw.times[-1]:.1f}s")
    return raw


def set_montage(raw):
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    raw.set_montage(montage, on_missing="warn", match_case=False)
    return raw


def notch(raw):
    p_before_60 = band_power(raw, 55, 65)
    p_before_120 = band_power(raw, 115, 125, fmax=130)
    p_alpha_before = band_power(raw, 8, 13)
    print(f"  PRE-notch  55-65 Hz: {p_before_60:.3e}, 115-125 Hz: {p_before_120:.3e}, alpha 8-13 Hz: {p_alpha_before:.3e}")
    raw.notch_filter(freqs=NOTCH_FREQS, verbose="WARNING")
    p_after_60 = band_power(raw, 55, 65)
    p_after_120 = band_power(raw, 115, 125, fmax=130)
    p_alpha_after = band_power(raw, 8, 13)
    print(f"  POST-notch 55-65 Hz: {p_after_60:.3e}, 115-125 Hz: {p_after_120:.3e}, alpha 8-13 Hz: {p_alpha_after:.3e}")
    print(f"  60 Hz attenuation: {p_before_60 / max(p_after_60, 1e-30):.1f}x")
    print(f"  120 Hz attenuation: {p_before_120 / max(p_after_120, 1e-30):.1f}x")
    alpha_ratio = (p_alpha_after / p_alpha_before) if p_alpha_before else float("nan")
    print(f"  alpha post/pre ratio (should be ~1.0): {alpha_ratio:.3f}")
    return raw


def make_two_streams(raw):
    raw_ica = raw.copy().filter(l_freq=1.0, h_freq=100.0, fir_design="firwin", verbose="WARNING")
    raw_analysis = raw.copy().filter(l_freq=1.0, h_freq=20.0, fir_design="firwin", verbose="WARNING")
    print(f"  raw_ica: {raw_ica.info['highpass']}-{raw_ica.info['lowpass']} Hz")
    print(f"  raw_analysis: {raw_analysis.info['highpass']}-{raw_analysis.info['lowpass']} Hz")
    return raw_ica, raw_analysis


def car(raw):
    raw.set_eeg_reference("average", projection=False, verbose="WARNING")
    return raw


def find_bads_pyprep(raw_ica):
    nc = NoisyChannels(raw_ica, random_state=42)
    nc.find_all_bads()
    bads = [str(b) for b in nc.get_bads()]
    print(f"  PyPREP NoisyChannels flagged {len(bads)} bad(s): {bads}")
    by_method = {
        "by_nan": list(getattr(nc, "bad_by_nan", []) or []),
        "by_flat": list(getattr(nc, "bad_by_flat", []) or []),
        "by_deviation": list(getattr(nc, "bad_by_deviation", []) or []),
        "by_hf_noise": list(getattr(nc, "bad_by_hf_noise", []) or []),
        "by_correlation": list(getattr(nc, "bad_by_correlation", []) or []),
        "by_SNR": list(getattr(nc, "bad_by_SNR", []) or []),
        "by_dropout": list(getattr(nc, "bad_by_dropout", []) or []),
        "by_ransac": list(getattr(nc, "bad_by_ransac", []) or []),
    }
    for method, items in by_method.items():
        if items:
            print(f"    {method}: {len(items)} -> {[str(c) for c in items]}")
    return bads


def apply_bads(raw_ica, raw_analysis, bads):
    raw_ica.info["bads"] = list(bads)
    raw_analysis.info["bads"] = list(bads)
    if bads:
        raw_ica.interpolate_bads(reset_bads=True, verbose="WARNING")
        raw_analysis.interpolate_bads(reset_bads=True, verbose="WARNING")
    return raw_ica, raw_analysis


def fit_ica(raw_ica):
    ica = mne.preprocessing.ICA(
        n_components=20,
        method="infomax",
        fit_params={"extended": True},
        max_iter=500,
        random_state=42,
    )
    ica.fit(raw_ica, picks="eeg", verbose="WARNING")
    print(f"  fitted n_components={ica.n_components_}")
    return ica


def run_icalabel(raw_ica, ica):
    labels = label_components(raw_ica, ica, method="iclabel")
    probs = labels["y_pred_proba"]
    if hasattr(probs, "tolist"):
        probs = probs.tolist()
    probs = [float(p) for p in probs]
    counts = {lab: int(sum(1 for x in labels["labels"] if x == lab)) for lab in set(labels["labels"])}
    print(f"  label distribution: {counts}")
    print(f"  median max-prob: {float(np.median(probs)):.3f}")
    print("  IC# label                  prob")
    for i, (lab, p) in enumerate(zip(labels["labels"], probs)):
        print(f"  {i:>3}  {lab:<22} {p:.3f}")
    return labels, probs


def select_bad_ics(labels, probs):
    idxs = [i for i, lab in enumerate(labels["labels"]) if lab in REJECT_LABELS]
    info = [(i, labels["labels"][i], probs[i]) for i in idxs]
    print(f"  excluding {len(idxs)} component(s):")
    for i, lab, p in info:
        print(f"    IC{i:>2}  {lab:<22}  p={p:.3f}")
    return idxs, info


def apply_ica_to_analysis(raw_analysis, ica, bad_ic_idx):
    ica.exclude = list(bad_ic_idx)
    raw_clean = ica.apply(raw_analysis.copy(), verbose="WARNING")
    return raw_clean


def resample(raw):
    raw.resample(sfreq=200.0, verbose="WARNING")
    return raw


def save(raw, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(out_path), overwrite=True, verbose="WARNING")
    sz = out_path.stat().st_size / (1024 * 1024)
    print(f"  wrote {out_path.name} ({sz:.1f} MB)")


def main():
    if not EEG_PATH.exists():
        sys.exit(f"EEG file not found: {EEG_PATH}")

    t_start = time.time()
    summary = {}

    raw = run_step("1. Load raw", load_raw, EEG_PATH)
    summary["orig_duration_s"] = float(raw.times[-1])
    summary["orig_n_channels"] = len(raw.ch_names)
    summary["orig_sfreq"] = float(raw.info["sfreq"])

    raw = run_step("2. Set montage GSN-HydroCel-129", set_montage, raw)
    raw = run_step("3. Notch 60/120 Hz", notch, raw)

    raw_ica, raw_analysis = run_step("4. Build two filter streams (1-100 / 1-20)", make_two_streams, raw)
    raw_ica = run_step("5a. CAR on raw_ica", car, raw_ica)
    raw_analysis = run_step("5b. CAR on raw_analysis", car, raw_analysis)

    bads = run_step("6a. PyPREP NoisyChannels on raw_ica", find_bads_pyprep, raw_ica)
    summary["n_bad_channels"] = len(bads)
    summary["bad_channels"] = bads
    raw_ica, raw_analysis = run_step("6b. Interpolate bads on both streams", apply_bads, raw_ica, raw_analysis, bads)

    ica = run_step("7. Fit ICA on raw_ica", fit_ica, raw_ica)
    labels, probs = run_step("8. ICALabel", run_icalabel, raw_ica, ica)
    summary["ic_label_counts"] = {lab: int(sum(1 for x in labels["labels"] if x == lab)) for lab in set(labels["labels"])}
    summary["ic_median_max_prob"] = float(np.median(probs))

    bad_ic_idx, bad_ic_info = run_step("9. Select bad ICs", select_bad_ics, labels, probs)
    summary["n_components_rejected"] = len(bad_ic_idx)
    summary["rejected_ics"] = bad_ic_info

    raw_after_ica = run_step("10. Apply ICA exclusions to raw_analysis", apply_ica_to_analysis, raw_analysis, ica, bad_ic_idx)
    raw_after_ica = run_step("11. Re-apply CAR after ICA", car, raw_after_ica)
    raw_final = run_step("12. Resample to 200 Hz", resample, raw_after_ica)
    run_step("13. Save", save, raw_final, OUT_PATH)

    summary["final_n_samples"] = int(raw_final.n_times)
    summary["final_sfreq"] = float(raw_final.info["sfreq"])
    summary["final_n_channels"] = len(raw_final.ch_names)
    summary["final_duration_s"] = float(raw_final.times[-1])
    summary["wall_clock_s"] = round(time.time() - t_start, 1)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nDONE")


if __name__ == "__main__":
    main()
