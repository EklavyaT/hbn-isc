"""Shared figure style + palette for the IEEE TAFFC paper."""
import os
import matplotlib as mpl


def apply_paper_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


MOVIE_ORDER = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]
MOVIE_COLORS = {
    "DespicableMe": "#E69F00",
    "DiaryOfAWimpyKid": "#56B4E9",
    "FunwithFractals": "#009E73",
    "ThePresent": "#CC79A7",
}
MOVIE_LABELS = {
    "DespicableMe": "Despicable Me",
    "DiaryOfAWimpyKid": "Diary of a Wimpy Kid",
    "FunwithFractals": "Fun with Fractals",
    "ThePresent": "The Present",
}
SEX_COLORS = {"M": "#0072B2", "F": "#D55E00"}
BAND_COLORS = {"theta": "#08519c", "alpha": "#3182bd", "beta": "#9ecae1"}

PAPER_DIR = "figures/paper"
os.makedirs(PAPER_DIR, exist_ok=True)


def get_montage_info():
    """Load an MNE info object with the GSN-HydroCel-129 montage attached."""
    import mne
    import os as _os
    fdir = "outputs/preprocessed_movies_R1"
    fifs = [f for f in _os.listdir(fdir) if "ThePresent" in f and "v3concat" in f]
    raw = mne.io.read_raw_fif(_os.path.join(fdir, fifs[0]), preload=False, verbose="ERROR")
    if raw.get_montage() is None:
        raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-129"), on_missing="warn")
    return raw.info, list(raw.ch_names)


def save_pdf_png(fig, name):
    """Save fig as both PDF and PNG to PAPER_DIR/name.{pdf,png}."""
    fig.savefig(f"{PAPER_DIR}/{name}.pdf")
    fig.savefig(f"{PAPER_DIR}/{name}.png")
