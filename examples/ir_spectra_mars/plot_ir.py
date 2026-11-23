"""Compare MARS IR spectra of Ac-Ala5-LysH+ with the experimental IRMPD spectrum.

Usage:
    python plot_ir.py rossi2010_ala5lysh_ir.csv "SO3LR Tauri-M=md_so3lr.dat" "ff19SB=md_ff19sb.dat"

Each MARS file is the two-column output of `mars ir` (wavenumber, intensity).
Computed wavenumbers are scaled by 0.97, smoothed with a Gaussian and
normalised in the 1300-1800 cm-1 window together with the experiment.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

WINDOW = (1300.0, 1800.0)
AMIDE_I = (1600.0, 1750.0)
AMIDE_II = (1450.0, 1600.0)
COLORS = ["#3b6e8f", "#a4473d", "#5b7f4a", "#7a5c8e"]


def load_mars(path, scale, sigma_cm):
    x, y = np.loadtxt(path, comments="#", unpack=True)
    x = x * scale
    grid = np.arange(WINDOW[0] - 100, WINDOW[1] + 100, 1.0)
    if len(x) > 1 and np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-3):
        y = np.interp(grid, x, y)  # MD spectrum on a regular grid
    else:
        y = np.histogram(x, bins=np.append(grid, grid[-1] + 1.0), weights=y)[0]  # harmonic sticks
    return grid, gaussian_filter1d(y, sigma_cm)


def normalise(x, y):
    mask = (x >= WINDOW[0]) & (x <= WINDOW[1])
    y = y - y[mask].min()
    return y / y[mask].max()


def band_max(x, y, band):
    mask = (x >= band[0]) & (x <= band[1])
    return x[mask][np.argmax(y[mask])]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", help="CSV with wavenumber_cm-1,intensity")
    parser.add_argument("spectra", nargs="+", help="LABEL=mars_output.dat")
    parser.add_argument("--scale", type=float, default=0.97, help="frequency scaling factor (default 0.97)")
    parser.add_argument("--sigma", type=float, default=8.0, help="Gaussian smoothing sigma in cm-1 (default 8)")
    parser.add_argument("-o", "--output", default="ir_comparison.png")
    args = parser.parse_args()

    rows = [line.split(",") for line in open(args.experiment) if line[0].isdigit()]
    xe, ye = np.array(rows, dtype=float).T
    ye = normalise(xe, ye)

    fig, axes = plt.subplots(len(args.spectra), 1, figsize=(5.5, 1.9 * len(args.spectra) + 0.4),
                             sharex=True, squeeze=False)
    print(f"{'':16s}{'amide I':>10s}{'amide II':>10s}{'Pearson r':>11s}")
    print(f"{'experiment':16s}{band_max(xe, ye, AMIDE_I):10.0f}{band_max(xe, ye, AMIDE_II):10.0f}{'':>11s}")
    for ax, spec, color in zip(axes[:, 0], args.spectra, COLORS):
        label, path = spec.split("=", 1)
        x, y = load_mars(path, args.scale, args.sigma)
        y = normalise(x, y)
        mask = (xe >= WINDOW[0]) & (xe <= WINDOW[1])
        r = np.corrcoef(ye[mask], np.interp(xe[mask], x, y))[0, 1]
        print(f"{label:16s}{band_max(x, y, AMIDE_I):10.0f}{band_max(x, y, AMIDE_II):10.0f}{r:11.2f}")

        ax.fill_between(xe, ye, color="0.82", lw=0, label="Experiment")
        ax.plot(xe, ye, color="0.45", lw=0.8)
        ax.plot(x, y, color=color, lw=1.6, label=label)
        ax.text(0.02, 0.88, f"r = {r:.2f}", transform=ax.transAxes, fontsize=8, color=color)
        ax.set_yticks([])
        ax.set_ylim(0, 1.12)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        for band, name in ((AMIDE_I, "amide I"), (AMIDE_II, "amide II")):
            ax.axvspan(*band, color="0.5", alpha=0.06, lw=0)
            if ax is axes[0, 0]:
                ax.text(sum(band) / 2, 1.13, name, ha="center", va="bottom", fontsize=8, color="0.35")
    axes[-1, 0].set_xlim(*WINDOW)
    axes[-1, 0].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    fig.supylabel("Intensity (normalised)", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Figure written to {args.output}")


if __name__ == "__main__":
    main()
