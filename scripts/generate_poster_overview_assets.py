from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.linalg import svd

import benchmark_vortex_shedding as bench


DEEPXDE_DMD_CONFIG = bench.MethodConfig("dmd_sindy", 10, "poly2", "finite_difference", 0.215)
KUTZ_DMD_CONFIG = bench.MethodConfig("dmd_sindy", 9, "meanfield_poly2", "finite_difference", 0.1)


def point_scalar_field(bundle: bench.DatasetBundle, snapshot: np.ndarray) -> np.ndarray:
    n_points = bundle.coords.shape[0]
    return snapshot[:n_points]


def point_speed_field(bundle: bench.DatasetBundle, snapshot: np.ndarray) -> np.ndarray:
    n_points = bundle.coords.shape[0]
    return np.hypot(snapshot[:n_points], snapshot[n_points:])


def plot_field(axis, bundle: bench.DatasetBundle, field: np.ndarray, title: str, cmap: str = "RdBu_r"):
    vmax = float(np.nanmax(np.abs(field)))
    vmax = max(vmax, 1.0e-12)
    if bundle.plot_kind == "grid":
        image = axis.imshow(field.reshape(bundle.grid_shape), cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
    else:
        image = axis.tricontourf(
            bundle.coords[:, 0],
            bundle.coords[:, 1],
            field,
            levels=32,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
        )
        axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def build_dmd_result(bundle: bench.DatasetBundle, config: bench.MethodConfig) -> dict[str, object]:
    train, val, test = bench.split_dataset(bundle)
    return bench.refit_and_evaluate(bundle, train, val, test, config, compute_long_horizon=False)


def make_mean_subtraction_figure(bundle: bench.DatasetBundle, slug: str, snapshot_index: int) -> None:
    bench.apply_plot_style()
    mean_field = bundle.snapshots.mean(axis=0)
    snapshot = bundle.snapshots[snapshot_index]
    fluctuation = snapshot - mean_field

    titles = ("Snapshot", "Mean Flow", "Fluctuation")
    colorbar_label = "Amplitude"
    if bundle.plot_kind == "point_cloud":
        mean_field = point_scalar_field(bundle, mean_field)
        snapshot = point_scalar_field(bundle, snapshot)
        fluctuation = point_scalar_field(bundle, fluctuation)
        titles = (r"$u$ snapshot", r"Mean $\bar{u}$", r"Fluctuation $u'$")
        colorbar_label = r"$u$ velocity"

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), constrained_layout=True)
    image = plot_field(axes[0], bundle, snapshot, titles[0])
    plot_field(axes[1], bundle, mean_field, titles[1])
    plot_field(axes[2], bundle, fluctuation, titles[2])
    cbar = fig.colorbar(image, ax=axes, shrink=0.80, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label(colorbar_label)

    fig.savefig(bench.FIGURES_DIR / f"poster_mean_subtraction_{slug}.png")
    fig.savefig(bench.FIGURES_DIR / f"poster_mean_subtraction_{slug}.pdf")
    plt.close(fig)


def make_scree_plot_single(bundle: bench.DatasetBundle, slug: str) -> None:
    bench.apply_plot_style()
    train, _, _ = bench.split_dataset(bundle)
    centered = train.snapshots - train.snapshots.mean(axis=0, keepdims=True)
    _, singular_values, _ = svd(centered, full_matrices=False)
    normalized = singular_values[:12] / singular_values[0]
    rank = max(3, min(10, bench.svht_rank(singular_values, *centered.shape)))

    fig, axis = plt.subplots(1, 1, figsize=(5.4, 4.0), constrained_layout=True)
    axis.plot(np.arange(1, normalized.size + 1), normalized, color="#1D4ED8", marker="o", lw=2.0)
    axis.axvline(rank, color="#C2410C", ls="--", lw=1.7, label=rf"keep first $r={rank}$ modes")
    axis.set_yscale("log")
    axis.set_title(f"{bundle.title.replace(' (vorticity)', '').replace(' (u,v)', '')}: rank selection")
    axis.set_xlabel("Mode index")
    axis.set_ylabel(r"Normalized singular value $\sigma_i / \sigma_1$")
    axis.grid(True, alpha=0.18)
    axis.legend(frameon=False, loc="upper right")

    fig.savefig(bench.FIGURES_DIR / f"poster_singular_value_scree_{slug}.png")
    fig.savefig(bench.FIGURES_DIR / f"poster_singular_value_scree_{slug}.pdf")
    plt.close(fig)


def make_scree_plot() -> None:
    bench.apply_plot_style()
    bundles = [bench.load_kutz_dataset(), bench.load_deepxde_dataset()]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), constrained_layout=True, sharey=True)

    for axis, bundle in zip(axes, bundles):
        train, _, _ = bench.split_dataset(bundle)
        centered = train.snapshots - train.snapshots.mean(axis=0, keepdims=True)
        _, singular_values, _ = svd(centered, full_matrices=False)
        normalized = singular_values[:12] / singular_values[0]
        rank = max(3, min(10, bench.svht_rank(singular_values, *centered.shape)))

        axis.plot(np.arange(1, normalized.size + 1), normalized, color="#1D4ED8", marker="o", lw=2.0)
        axis.axvline(rank, color="#C2410C", ls="--", lw=1.6, label=rf"chosen $r={rank}$")
        axis.set_yscale("log")
        axis.set_title(bundle.title.replace(" (vorticity)", "").replace(" (u,v)", ""))
        axis.set_xlabel("Mode index")
        axis.grid(True, alpha=0.18)
        axis.legend(frameon=False, loc="upper right")

    axes[0].set_ylabel(r"Normalized singular value $\sigma_i / \sigma_1$")
    fig.savefig(bench.FIGURES_DIR / "poster_singular_value_scree.png")
    fig.savefig(bench.FIGURES_DIR / "poster_singular_value_scree.pdf")
    plt.close(fig)


def make_deepxde_phase_figures() -> None:
    bench.apply_plot_style()
    bundle = bench.load_deepxde_dataset()
    result = build_dmd_result(bundle, DEEPXDE_DMD_CONFIG)
    truth = result["truth_states"]
    rollout = result["rollout_states"]

    fig_truth = plt.figure(figsize=(7.2, 5.7), constrained_layout=True)
    axis_truth = fig_truth.add_subplot(111, projection="3d")
    axis_truth.plot(truth[:, 0], truth[:, 1], truth[:, 2], color="#0F172A", lw=2.2)
    axis_truth.scatter(*truth[0], color="#0F172A", s=28, marker="o")
    axis_truth.set_xlabel(r"$a_1$")
    axis_truth.set_ylabel(r"$a_2$")
    axis_truth.set_zlabel(r"$a_3$")
    axis_truth.set_title("DeepXDE Truth Coordinates")
    axis_truth.view_init(elev=23, azim=-58)
    axis_truth.xaxis.pane.fill = False
    axis_truth.yaxis.pane.fill = False
    axis_truth.zaxis.pane.fill = False
    fig_truth.savefig(bench.FIGURES_DIR / "poster_deepxde_truth_phase.png")
    fig_truth.savefig(bench.FIGURES_DIR / "poster_deepxde_truth_phase.pdf")
    plt.close(fig_truth)

    fig_compare = plt.figure(figsize=(7.2, 5.7), constrained_layout=True)
    axis_compare = fig_compare.add_subplot(111, projection="3d")
    axis_compare.plot(truth[:, 0], truth[:, 1], truth[:, 2], color="#0F172A", lw=2.2, label="Truth")
    axis_compare.plot(
        rollout[:, 0],
        rollout[:, 1],
        rollout[:, 2],
        color=bench.METHOD_COLORS["dmd_sindy"],
        lw=2.0,
        label=f"DMD+SINDy (NRMSE={result['field_nrmse']:.3f})",
    )
    axis_compare.scatter(*truth[0], color="#0F172A", s=28, marker="o")
    axis_compare.scatter(*rollout[-1], color=bench.METHOD_COLORS["dmd_sindy"], s=32, marker="^")
    axis_compare.set_xlabel(r"$a_1$")
    axis_compare.set_ylabel(r"$a_2$")
    axis_compare.set_zlabel(r"$a_3$")
    axis_compare.set_title("DeepXDE Truth vs DMD+SINDy")
    axis_compare.view_init(elev=23, azim=-58)
    axis_compare.xaxis.pane.fill = False
    axis_compare.yaxis.pane.fill = False
    axis_compare.zaxis.pane.fill = False
    axis_compare.legend(frameon=False, loc="upper left")
    fig_compare.savefig(bench.FIGURES_DIR / "poster_deepxde_truth_vs_dmd_phase.png")
    fig_compare.savefig(bench.FIGURES_DIR / "poster_deepxde_truth_vs_dmd_phase.pdf")
    plt.close(fig_compare)


def make_sparse_equation_figure(bundle: bench.DatasetBundle, config: bench.MethodConfig, slug: str) -> None:
    bench.apply_plot_style()
    result = build_dmd_result(bundle, config)
    coefficients = np.asarray(result["model"].model.coefficients(), dtype=float)
    feature_names = result["model"].model.get_feature_names()

    fig, axis = plt.subplots(1, 1, figsize=(8.3, 4.6), constrained_layout=True)
    image = axis.imshow(coefficients, cmap="PuOr", aspect="auto")
    axis.set_title(f"{bundle.title.replace(' (vorticity)', '').replace(' (u,v)', '')}: sparse equation discovery")
    axis.set_xlabel("Library term")
    axis.set_ylabel("State equation")
    axis.set_xticks(np.arange(len(feature_names)), labels=feature_names, rotation=40, ha="right")
    axis.set_yticks([0, 1, 2], labels=[r"$\dot{a}_1$", r"$\dot{a}_2$", r"$\dot{a}_3$"])
    axis.text(0.03, -0.22, r"$\dot{a}=\Theta(a)\Xi$", transform=axis.transAxes, fontsize=11)
    cbar = fig.colorbar(image, ax=axis, shrink=0.90, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label("Coefficient")

    fig.savefig(bench.FIGURES_DIR / f"poster_sparse_equation_discovery_{slug}.png")
    fig.savefig(bench.FIGURES_DIR / f"poster_sparse_equation_discovery_{slug}.pdf")
    plt.close(fig)


def add_goal_card(axis, title: str, body: str, accent: str) -> None:
    axis.set_facecolor("#F8FAFC")
    axis.text(0.05, 0.88, title, ha="left", va="top", fontsize=12, fontweight="bold", color="#0F172A")
    axis.text(0.05, 0.58, body, ha="left", va="top", fontsize=10.5, color="#334155", linespacing=1.35)
    axis.add_patch(plt.Rectangle((0.03, 0.10), 0.94, 0.05, color=accent, alpha=0.95, transform=axis.transAxes))
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def make_goal_block_figure() -> None:
    bench.apply_plot_style()
    kutz = bench.load_kutz_dataset()
    kutz_result = build_dmd_result(kutz, KUTZ_DMD_CONFIG)

    mean_field = kutz.snapshots.mean(axis=0)
    snapshot = kutz.snapshots[40] - mean_field
    mode_one = kutz_result["reduced_model"].basis[:, 0]
    coefficients = np.asarray(kutz_result["model"].model.coefficients(), dtype=float)

    fig = plt.figure(figsize=(14.5, 4.8), constrained_layout=True)
    grid = GridSpec(1, 5, figure=fig, width_ratios=[1.1, 0.18, 1.1, 0.18, 1.25])

    ax0 = fig.add_subplot(grid[0, 0])
    plot_field(ax0, kutz, snapshot, "Challenge: high-dimensional wake data")
    ax0.text(
        0.03,
        -0.16,
        "Thousands of spatial degrees of freedom\nmake direct equation discovery difficult.",
        transform=ax0.transAxes,
        fontsize=9.8,
        color="#334155",
    )

    ax_arrow0 = fig.add_subplot(grid[0, 1])
    ax_arrow0.axis("off")
    ax_arrow0.text(0.5, 0.56, r"$\Longrightarrow$", ha="center", va="center", fontsize=30, color="#F59E0B")
    ax_arrow0.text(0.5, 0.28, "DMD", ha="center", va="center", fontsize=11, fontweight="bold", color="#0F172A")

    ax1 = fig.add_subplot(grid[0, 2])
    plot_field(ax1, kutz, mode_one, "Need coherent latent modes")
    ax1.text(
        0.03,
        -0.16,
        "DMD extracts the dominant oscillatory structure\nand low-dimensional coordinates.",
        transform=ax1.transAxes,
        fontsize=9.8,
        color="#334155",
    )

    ax_arrow1 = fig.add_subplot(grid[0, 3])
    ax_arrow1.axis("off")
    ax_arrow1.text(0.5, 0.56, r"$\Longrightarrow$", ha="center", va="center", fontsize=30, color="#8B5CF6")
    ax_arrow1.text(0.5, 0.28, "SINDy", ha="center", va="center", fontsize=11, fontweight="bold", color="#0F172A")

    ax2 = fig.add_subplot(grid[0, 4])
    ax2.set_facecolor("#F8FAFC")
    im = ax2.imshow(coefficients, cmap="PuOr", aspect="auto")
    ax2.set_title("Goal: sparse interpretable ODE")
    ax2.set_xlabel("Library term")
    ax2.set_ylabel("State")
    ax2.set_yticks([0, 1, 2], labels=[r"$\dot{a}_1$", r"$\dot{a}_2$", r"$\dot{a}_3$"])
    ax2.text(0.03, -0.18, r"$x(t)\rightarrow a(t)\rightarrow \dot{a}=\Theta(a)\Xi$", transform=ax2.transAxes, fontsize=10, color="#0F172A")
    ax2.text(
        0.03,
        -0.34,
        "Use DMD for reduced coordinates and SINDy for\ncompact governing-equation discovery.",
        transform=ax2.transAxes,
        fontsize=9.8,
        color="#334155",
    )
    cbar = fig.colorbar(im, ax=ax2, shrink=0.84, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label("Coefficient")

    fig.savefig(bench.FIGURES_DIR / "poster_goal_block_why_dmd_sindy.png")
    fig.savefig(bench.FIGURES_DIR / "poster_goal_block_why_dmd_sindy.pdf")
    plt.close(fig)


def make_overview_algorithm_flow() -> None:
    bench.apply_plot_style()
    kutz = bench.load_kutz_dataset()
    deepxde = bench.load_deepxde_dataset()
    deepxde_result = build_dmd_result(deepxde, DEEPXDE_DMD_CONFIG)
    kutz_result = build_dmd_result(kutz, KUTZ_DMD_CONFIG)

    mean_field = kutz.snapshots.mean(axis=0)
    snapshot = kutz.snapshots[40]
    fluctuation = snapshot - mean_field
    truth = deepxde_result["truth_states"]
    coefficients = np.asarray(kutz_result["model"].model.coefficients(), dtype=float)

    fig = plt.figure(figsize=(16.0, 4.7), constrained_layout=True)
    grid = GridSpec(1, 5, figure=fig, width_ratios=[1.0, 1.0, 1.15, 1.0, 1.0])
    axes = [fig.add_subplot(grid[0, idx], projection="3d" if idx == 3 else None) for idx in range(5)]

    vmax = float(np.max(np.abs(snapshot)))
    axes[0].imshow(snapshot.reshape(kutz.grid_shape), cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    axes[0].set_title("1. Snapshots")
    axes[0].text(0.03, -0.12, r"$X=[x_1,x_2,\ldots,x_m]$", transform=axes[0].transAxes, fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    subgrid = grid[0, 1].subgridspec(2, 1, hspace=0.15)
    ax_mean = fig.add_subplot(subgrid[0, 0])
    ax_scree = fig.add_subplot(subgrid[1, 0])
    plot_field(ax_mean, kutz, mean_field, r"Mean subtraction: $x'(t)=x(t)-\bar{x}$")
    centered = kutz.snapshots[:90] - kutz.snapshots[:90].mean(axis=0, keepdims=True)
    _, singular_values, _ = svd(centered, full_matrices=False)
    normalized = singular_values[:10] / singular_values[0]
    rank = max(3, min(10, bench.svht_rank(singular_values, *centered.shape)))
    ax_scree.plot(np.arange(1, normalized.size + 1), normalized, color="#1D4ED8", marker="o", lw=1.8)
    ax_scree.axvline(rank, color="#C2410C", ls="--", lw=1.5)
    ax_scree.set_yscale("log")
    ax_scree.set_title("2. Rank selection")
    ax_scree.set_xlabel("Mode")
    ax_scree.set_ylabel(r"$\sigma_i / \sigma_1$")

    mode_grid = grid[0, 2].subgridspec(1, 3, wspace=0.02)
    mode_axes = [fig.add_subplot(mode_grid[0, idx]) for idx in range(3)]
    bench.plot_mode_panel(mode_axes, kutz, kutz_result["reduced_model"])
    mode_axes[0].set_title("3. DMD latent modes")

    axes[3].plot(truth[:, 0], truth[:, 1], truth[:, 2], color="#0F172A", lw=2.0)
    axes[3].set_title("4. Reduced coordinates")
    axes[3].set_xlabel(r"$a_1$")
    axes[3].set_ylabel(r"$a_2$")
    axes[3].set_zlabel(r"$a_3$")
    axes[3].view_init(elev=23, azim=-58)
    axes[3].xaxis.pane.fill = False
    axes[3].yaxis.pane.fill = False
    axes[3].zaxis.pane.fill = False
    axes[3].text2D(0.08, -0.10, r"$x(t)\approx \bar{x}+\Phi a(t)$", transform=axes[3].transAxes, fontsize=10)

    im = axes[4].imshow(coefficients, cmap="PuOr", aspect="auto")
    axes[4].set_title("5. Sparse equation discovery")
    axes[4].set_xlabel("Library term")
    axes[4].set_ylabel("State equation")
    axes[4].set_yticks([0, 1, 2], labels=[r"$\dot{a}_1$", r"$\dot{a}_2$", r"$\dot{a}_3$"])
    axes[4].text(0.03, -0.16, r"$\dot{a}=\Theta(a)\Xi$", transform=axes[4].transAxes, fontsize=10)
    cbar = fig.colorbar(im, ax=axes[4], shrink=0.80, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label("Coefficient")

    fig.savefig(bench.FIGURES_DIR / "poster_overview_algorithm_flow.png")
    fig.savefig(bench.FIGURES_DIR / "poster_overview_algorithm_flow.pdf")
    plt.close(fig)


def make_overview_algorithm_flow_deepxde() -> None:
    bench.apply_plot_style()
    bundle = bench.load_deepxde_dataset()
    result = build_dmd_result(bundle, DEEPXDE_DMD_CONFIG)
    train, _, _ = bench.split_dataset(bundle)

    snapshot = bundle.snapshots[70]
    mean_field = bundle.snapshots.mean(axis=0)
    fluctuation = snapshot - mean_field
    truth = result["truth_states"]
    coefficients = np.asarray(result["model"].model.coefficients(), dtype=float)

    u_snapshot = point_scalar_field(bundle, snapshot)
    u_mean = point_scalar_field(bundle, mean_field)
    u_fluctuation = point_scalar_field(bundle, fluctuation)

    centered = train.snapshots - train.snapshots.mean(axis=0, keepdims=True)
    _, singular_values, _ = svd(centered, full_matrices=False)
    normalized = singular_values[:10] / singular_values[0]
    rank = max(3, min(10, bench.svht_rank(singular_values, *centered.shape)))

    fig = plt.figure(figsize=(16.2, 4.9), constrained_layout=True)
    grid = GridSpec(1, 5, figure=fig, width_ratios=[1.0, 1.05, 1.15, 1.0, 1.05])
    axes = [fig.add_subplot(grid[0, idx], projection="3d" if idx == 3 else None) for idx in range(5)]

    image0 = plot_field(axes[0], bundle, u_snapshot, r"1. Snapshot: $u(x,y,t)$")
    axes[0].text(0.03, -0.12, r"$X=[x_1,x_2,\ldots,x_m]$", transform=axes[0].transAxes, fontsize=10)
    cbar0 = fig.colorbar(image0, ax=axes[0], shrink=0.84, pad=0.02)
    cbar0.outline.set_visible(False)
    cbar0.set_label(r"$u$ velocity")

    subgrid = grid[0, 1].subgridspec(2, 1, hspace=0.18)
    ax_mean = fig.add_subplot(subgrid[0, 0])
    ax_scree = fig.add_subplot(subgrid[1, 0])
    plot_field(ax_mean, bundle, u_fluctuation, r"2. Mean subtraction: $u'=u-\bar{u}$")
    ax_scree.plot(np.arange(1, normalized.size + 1), normalized, color="#1D4ED8", marker="o", lw=1.8)
    ax_scree.axvline(rank, color="#C2410C", ls="--", lw=1.5, label=rf"chosen $r={rank}$")
    ax_scree.set_yscale("log")
    ax_scree.set_title("Rank selection")
    ax_scree.set_xlabel("Mode")
    ax_scree.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax_scree.legend(frameon=False, loc="upper right")

    mode_grid = grid[0, 2].subgridspec(1, 3, wspace=0.02)
    mode_axes = [fig.add_subplot(mode_grid[0, idx]) for idx in range(3)]
    bench.plot_mode_panel(mode_axes, bundle, result["reduced_model"])
    mode_axes[0].set_title("3. DMD latent modes")

    axes[3].plot(truth[:, 0], truth[:, 1], truth[:, 2], color="#0F172A", lw=2.0)
    axes[3].set_title("4. Reduced coordinates")
    axes[3].set_xlabel(r"$a_1$")
    axes[3].set_ylabel(r"$a_2$")
    axes[3].set_zlabel(r"$a_3$")
    axes[3].view_init(elev=23, azim=-58)
    axes[3].xaxis.pane.fill = False
    axes[3].yaxis.pane.fill = False
    axes[3].zaxis.pane.fill = False
    axes[3].text2D(0.08, -0.10, r"$x(t)\approx \bar{x}+\Phi a(t)$", transform=axes[3].transAxes, fontsize=10)

    feature_names = result["model"].model.get_feature_names()
    im = axes[4].imshow(coefficients, cmap="PuOr", aspect="auto")
    axes[4].set_title("5. Sparse equation discovery")
    axes[4].set_xlabel("Library term")
    axes[4].set_ylabel("State equation")
    axes[4].set_xticks(np.arange(len(feature_names)), labels=feature_names, rotation=40, ha="right")
    axes[4].set_yticks([0, 1, 2], labels=[r"$\dot{a}_1$", r"$\dot{a}_2$", r"$\dot{a}_3$"])
    axes[4].text(0.03, -0.22, r"$\dot{a}=\Theta(a)\Xi$", transform=axes[4].transAxes, fontsize=10)
    cbar = fig.colorbar(im, ax=axes[4], shrink=0.82, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label("Coefficient")

    fig.savefig(bench.FIGURES_DIR / "poster_overview_algorithm_flow_deepxde.png")
    fig.savefig(bench.FIGURES_DIR / "poster_overview_algorithm_flow_deepxde.pdf")
    plt.close(fig)


def main() -> None:
    bench.ensure_output_dirs()
    make_mean_subtraction_figure(bench.load_kutz_dataset(), "kutz", snapshot_index=40)
    make_mean_subtraction_figure(bench.load_deepxde_dataset(), "deepxde", snapshot_index=70)
    make_scree_plot()
    make_scree_plot_single(bench.load_deepxde_dataset(), "deepxde")
    make_deepxde_phase_figures()
    make_sparse_equation_figure(bench.load_deepxde_dataset(), DEEPXDE_DMD_CONFIG, "deepxde")
    make_goal_block_figure()
    make_overview_algorithm_flow()
    make_overview_algorithm_flow_deepxde()


if __name__ == "__main__":
    main()
