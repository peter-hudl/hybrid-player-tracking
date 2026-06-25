"""
Plot the similarity matrix from an AlignmentResult as an annotated heatmap.

Usage (script):
    python tools/plot_similarity.py            # saves similarity_heatmap.png
    python tools/plot_similarity.py out.png    # saves to specified path

Usage (library):
    from alignment_method import align_identities
    from tools.plot_similarity import plot_similarity_heatmap

    result = align_identities(...)
    fig = plot_similarity_heatmap(result)
    fig.savefig("heatmap.png", dpi=130)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from alignment_method import AlignmentResult


# Ground truth for the Halmstadt U19 dataset:
#   confirmed: jersey number matches optical player (verified via GPS cross-correlation)
#   unverified: outliers in time alignment — treat assignment result with caution
HALMSTADT_GT_JERSEYS = [3, 6, 7, 8, 12, 14, 17]
HALMSTADT_UNVERIFIED_JERSEYS = [2, 9, 16]


def plot_similarity_heatmap(
    result: AlignmentResult,
    title: str = "IMU-optical identity alignment: Pearson r similarity",
    gt_jerseys: list[int] | None = None,
    unverified_jerseys: list[int] | None = None,
    out_path: str | Path | None = None,
    dpi: int = 130,
) -> plt.Figure:
    """
    Annotated heatmap of the similarity matrix from align_identities().

    Parameters
    ----------
    result : AlignmentResult from alignment_method.align_identities()
    title : plot title
    gt_jerseys : jersey numbers where WIMU jersey == optical jersey (confirmed GT).
        Marked with a green '+'. Pass [] to disable. Defaults to the Halmstadt set.
    unverified_jerseys : jersey numbers where time alignment is unreliable.
        Marked with an orange dashed box on the diagonal. Defaults to the Halmstadt set.
    out_path : save figure to this path if provided
    dpi : output resolution

    Returns
    -------
    matplotlib Figure
    """
    if gt_jerseys is None:
        gt_jerseys = HALMSTADT_GT_JERSEYS
    if unverified_jerseys is None:
        unverified_jerseys = HALMSTADT_UNVERIFIED_JERSEYS

    sim = result.similarity_matrix
    jerseys = result.jerseys_imu
    opt_players = result.jerseys_optical
    asgn = result.assignments

    n_imu = len(jerseys)
    n_opt = len(opt_players)

    fig, ax = plt.subplots(
        figsize=(max(9, n_opt * 0.85), max(6, n_imu * 0.75))
    )

    im = ax.imshow(sim, aspect="auto", vmin=-0.2, vmax=1.0, cmap="RdYlGn")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")

    ax.set_xticks(range(n_opt))
    ax.set_xticklabels([f"opt{p}" for p in opt_players], rotation=90, fontsize=9)
    ax.set_yticks(range(n_imu))
    ax.set_yticklabels([f"J{j}" for j in jerseys], fontsize=10)
    ax.set_xlabel("Optical player (merged jersey)", fontsize=10)
    ax.set_ylabel("WIMU jersey", fontsize=10)

    n_correct = sum(1 for j, p in asgn.items() if j == p and j in gt_jerseys)
    n_verifiable = sum(1 for j in asgn if j in gt_jerseys)
    acc_str = f"{n_correct}/{n_verifiable}" if n_verifiable > 0 else "N/A"

    ax.set_title(
        f"{title}\n"
        f"Verified accuracy: {acc_str}  |  Gold: assignment  |  Green+: confirmed GT  |  Orange--: unverified",
        fontsize=9,
    )

    # Cell r values
    for ri, j in enumerate(jerseys):
        for ci, p in enumerate(opt_players):
            val = sim[ri, ci]
            ax.text(ci, ri, f"{val:.2f}", ha="center", va="center", fontsize=6,
                    color="black" if abs(val) < 0.7 else "white")

    # Gold box: Hungarian assignment
    for j, p in asgn.items():
        if j in jerseys and p in opt_players:
            ri = jerseys.index(j)
            ci = opt_players.index(p)
            ax.add_patch(plt.Rectangle(
                (ci - 0.5, ri - 0.5), 1, 1,
                fill=False, edgecolor="gold", linewidth=3.0,
            ))

    # Green + on confirmed GT diagonal cells
    for j in gt_jerseys:
        if j in jerseys and j in opt_players:
            ri = jerseys.index(j)
            ci = opt_players.index(j)
            ax.text(ci + 0.38, ri - 0.38, "+", ha="right", va="top",
                    fontsize=7, color="limegreen", fontweight="bold", zorder=5)

    # Orange dashed box on unverified expected diagonal cells
    for j in unverified_jerseys:
        if j in jerseys and j in opt_players:
            ri = jerseys.index(j)
            ci = opt_players.index(j)
            ax.add_patch(plt.Rectangle(
                (ci - 0.5, ri - 0.5), 1, 1,
                fill=False, edgecolor="orange", linewidth=2.0, linestyle="--",
            ))

    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=dpi)
        print(f"Saved {out_path}")

    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from alignment_method import align_identities

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("similarity_heatmap.png")

    print("Running alignment...")
    result = align_identities(
        imu_dir="data/wimu",
        tracking_parquet="data/tracking_full.parquet",
        system_delta_s=-796.97,
    )

    plot_similarity_heatmap(result, out_path=out)
