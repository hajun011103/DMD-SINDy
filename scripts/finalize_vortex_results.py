from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gplearn.genetic import SymbolicRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import benchmark_vortex_shedding as bench


FINAL_CONFIGS: dict[str, dict[str, bench.MethodConfig]] = {
    "kutz_cylinder": {
        "raw_sindy": bench.MethodConfig("raw_sindy", 3, "poly3", "tvregdiff", 1.0),
        "pod_sindy": bench.MethodConfig("pod_sindy", 10, "poly3", "tvregdiff", 1.0),
        "dmd_sindy": bench.MethodConfig("dmd_sindy", 9, "meanfield_poly2", "finite_difference", 0.1),
    },
    "deepxde_cylinder": {
        "raw_sindy": bench.MethodConfig("raw_sindy", 3, "poly2", "finite_difference", 0.215),
        "pod_sindy": bench.MethodConfig("pod_sindy", 6, "poly2", "finite_difference", 0.00464),
        "dmd_sindy": bench.MethodConfig("dmd_sindy", 10, "poly2", "finite_difference", 0.215),
    },
}

MODEL_ORDER = ["raw_sindy", "pod_sindy", "dmd_sindy", "genetic_sr"]
REDUCED_FIGURE_MODEL_ORDER = ["raw_sindy", "dmd_sindy", "genetic_sr"]
MODEL_LABELS = {
    "raw_sindy": bench.METHOD_LABELS["raw_sindy"],
    "pod_sindy": bench.METHOD_LABELS["pod_sindy"],
    "dmd_sindy": bench.METHOD_LABELS["dmd_sindy"],
    "genetic_sr": "Genetic SR",
}

TIME_SERIES_COLORS = {
    "truth": "#0F172A",
    "raw_sindy": bench.METHOD_COLORS["raw_sindy"],
    "pod_sindy": bench.METHOD_COLORS["pod_sindy"],
    "dmd_sindy": bench.METHOD_COLORS["dmd_sindy"],
    "genetic_sr": "#7C3AED",
}
LINE_STYLES = {
    "truth": "-",
    "raw_sindy": "-.",
    "pod_sindy": "--",
    "dmd_sindy": "-",
    "genetic_sr": ":",
}
LINE_WIDTHS = {
    "truth": 2.3,
    "raw_sindy": 1.8,
    "pod_sindy": 2.1,
    "dmd_sindy": 1.9,
    "genetic_sr": 1.9,
}
MODEL_MARKERS = {
    "truth": "o",
    "raw_sindy": "s",
    "pod_sindy": "D",
    "dmd_sindy": "^",
    "genetic_sr": "P",
}

DATASET_ORDER = ["kutz_cylinder", "deepxde_cylinder"]
DATASET_SLUGS = {
    "kutz_cylinder": "kutz",
    "deepxde_cylinder": "deepxde",
}
SNAPSHOT_FRACTIONS = np.linspace(0.05, 0.95, 6)


@dataclass
class GeneticSRModel:
    regressors: list[SymbolicRegressor]
    state_mean: np.ndarray
    state_scale: np.ndarray


def fit_genetic_sr_model(train_states: np.ndarray, time: np.ndarray, seed: int) -> GeneticSRModel:
    state_mean = train_states.mean(axis=0, keepdims=True)
    state_scale = train_states.std(axis=0, ddof=1, keepdims=True)
    state_scale[state_scale == 0.0] = 1.0
    scaled_states = (train_states - state_mean) / state_scale
    derivatives = np.gradient(scaled_states, time, axis=0, edge_order=2)

    regressors: list[SymbolicRegressor] = []
    feature_names = bench.STATE_NAMES[: scaled_states.shape[1]]
    for target_idx in range(scaled_states.shape[1]):
        regressor = SymbolicRegressor(
            population_size=350,
            generations=14,
            tournament_size=10,
            stopping_criteria=1.0e-4,
            const_range=(-2.0, 2.0),
            init_depth=(2, 5),
            function_set=("add", "sub", "mul", "div"),
            metric="mean absolute error",
            parsimony_coefficient=5.0e-3,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            feature_names=feature_names,
            low_memory=True,
            random_state=seed + target_idx,
            verbose=0,
        )
        regressor.fit(scaled_states, derivatives[:, target_idx])
        regressors.append(regressor)

    return GeneticSRModel(regressors=regressors, state_mean=state_mean, state_scale=state_scale)


def simulate_genetic_sr_model(
    model: GeneticSRModel,
    initial_state: np.ndarray,
    time: np.ndarray,
    clip_norm: float,
) -> np.ndarray:
    initial_scaled = ((np.asarray(initial_state, dtype=float).reshape(1, -1) - model.state_mean) / model.state_scale)[0]

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        state_2d = state.reshape(1, -1)
        return np.array([regressor.predict(state_2d)[0] for regressor in model.regressors], dtype=float)

    solution = solve_ivp(
        rhs,
        (float(time[0]), float(time[-1])),
        initial_scaled,
        t_eval=time,
        method="RK45",
        max_step=max(float(np.median(np.diff(time))), 1.0e-9),
        rtol=1.0e-7,
        atol=1.0e-9,
    )
    if not solution.success or solution.y.shape[1] != time.size:
        return np.full((time.size, initial_scaled.size), np.nan, dtype=float)

    trajectory = solution.y.T * model.state_scale + model.state_mean
    norms = np.linalg.norm(trajectory, axis=1)
    if np.any(~np.isfinite(trajectory)) or float(np.nanmax(norms)) > clip_norm:
        return np.full((time.size, initial_scaled.size), np.nan, dtype=float)
    return trajectory


def genetic_sr_equations(model: GeneticSRModel) -> list[str]:
    equations = []
    for idx, regressor in enumerate(model.regressors):
        equations.append(f"d{bench.STATE_NAMES[idx]}/dt = {regressor._program}")
    return equations


def write_genetic_sr_report(genetic_payload: dict[str, dict[str, object]]) -> None:
    lines = ["# Genetic Symbolic Regression Report", ""]
    for dataset_name in DATASET_ORDER:
        payload = genetic_payload[dataset_name]
        lines.append(f"## {payload['dataset_label']}")
        lines.append("")
        lines.append(f"- Test field NRMSE: {payload['field_nrmse']:.4f}")
        lines.append(f"- Stable long-horizon integration: {payload['stable']}")
        lines.append("- Equations:")
        for equation in payload["equations"]:
            lines.append(f"  - {equation}")
        lines.append("")
    (bench.RESULTS_DIR / "genetic_sr_equations.md").write_text("\n".join(lines), encoding="utf-8")


def make_long_time_grid(dt: float, training_states: np.ndarray) -> np.ndarray:
    n_points = max(240, 3 * int(training_states.shape[0]))
    return np.linspace(0.0, dt * (n_points - 1), n_points)


def simulate_model_in_common_basis(
    result: dict[str, object],
    common_reduced_model: bench.ReducedModel,
    initial_field: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    if result["model"] is None or result["reduced_model"] is None:
        return np.full((time.size, len(bench.STATE_NAMES)), np.nan, dtype=float)

    reduced_model = result["reduced_model"]
    initial_state = bench.project_fields(np.asarray(initial_field, dtype=float).reshape(1, -1), reduced_model)[0]
    clip_norm = max(15.0, 4.0 * float(np.max(np.linalg.norm(reduced_model.training_states, axis=1))))
    rollout_states = bench.simulate_sindy_model_solve_ivp(result["model"], initial_state, time, clip_norm=clip_norm)
    predicted_fields = bench.reconstruct_fields(rollout_states, reduced_model)
    return bench.project_fields(predicted_fields, common_reduced_model)


def build_comparison_payload(
    bundle: bench.DatasetBundle,
    test: bench.SplitData,
    dataset_results: dict[str, dict[str, object]],
    genetic_states: np.ndarray,
    genetic_model: GeneticSRModel,
) -> dict[str, object]:
    dmd_result = dataset_results["dmd_sindy"]
    pod_result = dataset_results["pod_sindy"]
    raw_result = dataset_results["raw_sindy"]
    common_reduced_model = dmd_result["reduced_model"]

    clip_norm = max(15.0, 4.0 * float(np.max(np.linalg.norm(common_reduced_model.training_states, axis=1))))
    long_time = make_long_time_grid(bundle.dt, common_reduced_model.training_states)
    initial_field = test.snapshots[0]

    short_states = {
        "dmd_sindy": dmd_result["rollout_states"],
        "pod_sindy": bench.project_fields(pod_result["predicted_fields"], common_reduced_model),
        "raw_sindy": bench.project_fields(raw_result["predicted_fields"], common_reduced_model),
        "genetic_sr": genetic_states,
    }
    short_field_nrmse = {
        "dmd_sindy": float(dmd_result["field_nrmse"]),
        "pod_sindy": float(pod_result["field_nrmse"]),
        "raw_sindy": float(raw_result["field_nrmse"]),
        "genetic_sr": float(bench.relative_field_error(bench.reconstruct_fields(genetic_states, common_reduced_model), test.snapshots)),
    }

    long_states = {
        "dmd_sindy": simulate_model_in_common_basis(dmd_result, common_reduced_model, initial_field, long_time),
        "pod_sindy": simulate_model_in_common_basis(pod_result, common_reduced_model, initial_field, long_time),
        "raw_sindy": simulate_model_in_common_basis(raw_result, common_reduced_model, initial_field, long_time),
        "genetic_sr": simulate_genetic_sr_model(genetic_model, dmd_result["truth_states"][0], long_time, clip_norm=clip_norm),
    }
    long_stable = {method_name: bool(np.all(np.isfinite(states))) for method_name, states in long_states.items()}

    return {
        "dataset_label": bundle.title,
        "time": test.time,
        "truth_states": dmd_result["truth_states"],
        "short_states": short_states,
        "short_field_nrmse": short_field_nrmse,
        "long_time": long_time,
        "long_states": long_states,
        "long_stable": long_stable,
        "truth_end": float(test.time[-1]),
    }


def plot_snapshot_field(axis, bundle: bench.DatasetBundle, snapshot: np.ndarray):
    if bundle.plot_kind == "grid":
        vmax = float(np.nanmax(np.abs(bundle.snapshots)))
        image = axis.imshow(
            snapshot.reshape(bundle.grid_shape),
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            origin="lower",
        )
        colorbar_label = "Vorticity"
    else:
        coords = bundle.coords
        n_points = coords.shape[0]
        speed = np.hypot(snapshot[:n_points], snapshot[n_points:])
        vmax = float(np.nanmax(np.hypot(bundle.snapshots[:, :n_points], bundle.snapshots[:, n_points:])))
        image = axis.tricontourf(
            coords[:, 0],
            coords[:, 1],
            speed,
            levels=36,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        colorbar_label = "Speed"
        axis.set_aspect("equal")

    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image, colorbar_label


def make_snapshot_figures(bundles: dict[str, bench.DatasetBundle]) -> None:
    bench.apply_plot_style()
    frame_indices = {
        dataset_name: np.clip(
            np.round(SNAPSHOT_FRACTIONS * (bundle.time.size - 1)).astype(int),
            0,
            bundle.time.size - 1,
        )
        for dataset_name, bundle in bundles.items()
    }

    for frame_number in range(len(SNAPSHOT_FRACTIONS)):
        fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8), constrained_layout=True)
        for axis, dataset_name in zip(axes, DATASET_ORDER):
            bundle = bundles[dataset_name]
            sample_idx = int(frame_indices[dataset_name][frame_number])
            image, colorbar_label = plot_snapshot_field(axis, bundle, bundle.snapshots[sample_idx])
            progress = 100.0 * sample_idx / max(bundle.time.size - 1, 1)
            axis.set_title(f"{bundle.title}\n$t = {bundle.time[sample_idx]:.2f}$ ({progress:.0f}% of record)")
            cbar = fig.colorbar(image, ax=axis, shrink=0.86, pad=0.02)
            cbar.outline.set_visible(False)
            cbar.set_label(colorbar_label)

        fig.savefig(bench.FIGURES_DIR / f"vortex_snapshot_{frame_number + 1:02d}.png")
        fig.savefig(bench.FIGURES_DIR / f"vortex_snapshot_{frame_number + 1:02d}.pdf")
        plt.close(fig)


def make_latent_mode_figures(
    bundles: dict[str, bench.DatasetBundle],
    dataset_results: dict[str, dict[str, object]],
) -> None:
    bench.apply_plot_style()

    for dataset_name in DATASET_ORDER:
        bundle = bundles[dataset_name]
        reduced_model = dataset_results[dataset_name]["dmd_sindy"]["reduced_model"]
        fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), constrained_layout=True)
        image = bench.plot_mode_panel(list(axes), bundle, reduced_model)
        fig.suptitle(f"{bundle.title}\nLatent Modes Used by DMD+SINDy", y=1.08, fontsize=15)
        cbar = fig.colorbar(image, ax=axes, shrink=0.84, pad=0.02)
        cbar.outline.set_visible(False)
        cbar.set_label("Mode amplitude")

        slug = DATASET_SLUGS[dataset_name]
        fig.savefig(bench.FIGURES_DIR / f"vortex_latent_modes_{slug}.png")
        fig.savefig(bench.FIGURES_DIR / f"vortex_latent_modes_{slug}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.6), constrained_layout=True)
    for row_idx, dataset_name in enumerate(DATASET_ORDER):
        bundle = bundles[dataset_name]
        reduced_model = dataset_results[dataset_name]["dmd_sindy"]["reduced_model"]
        row_axes = list(axes[row_idx])
        image = bench.plot_mode_panel(row_axes, bundle, reduced_model)
        row_axes[0].set_ylabel(bundle.title, fontsize=12)
        cbar = fig.colorbar(image, ax=row_axes, shrink=0.84, pad=0.02)
        cbar.outline.set_visible(False)
        cbar.set_label("Mode amplitude")

    fig.suptitle("Latent Modal Structures Used by DMD+SINDy", y=1.03, fontsize=15)
    fig.savefig(bench.FIGURES_DIR / "vortex_latent_modes.png")
    fig.savefig(bench.FIGURES_DIR / "vortex_latent_modes.pdf")
    plt.close(fig)


def add_model_curve(axis, time: np.ndarray, states: np.ndarray, state_idx: int, method_name: str, label: str) -> None:
    if np.all(np.isfinite(states)):
        axis.plot(
            time,
            states[:, state_idx],
            color=TIME_SERIES_COLORS[method_name],
            lw=LINE_WIDTHS[method_name],
            ls=LINE_STYLES[method_name],
            label=label,
        )
        axis.scatter(
            time[-1],
            states[-1, state_idx],
            color=TIME_SERIES_COLORS[method_name],
            s=20,
            marker=MODEL_MARKERS[method_name],
            zorder=5,
        )
        return
    axis.plot(
        [],
        [],
        color=TIME_SERIES_COLORS[method_name],
        lw=LINE_WIDTHS[method_name],
        ls=LINE_STYLES[method_name],
        label=f"{label} (unstable)",
    )


def add_phase_curve(axis, states: np.ndarray, method_name: str, label: str) -> None:
    if np.all(np.isfinite(states)):
        axis.plot(
            states[:, 0],
            states[:, 1],
            states[:, 2],
            color=TIME_SERIES_COLORS[method_name],
            lw=LINE_WIDTHS[method_name],
            ls=LINE_STYLES[method_name],
            label=label,
        )
        axis.scatter(*states[-1], color=TIME_SERIES_COLORS[method_name], s=22, marker=MODEL_MARKERS[method_name])
        return
    axis.plot(
        [],
        [],
        [],
        color=TIME_SERIES_COLORS[method_name],
        lw=LINE_WIDTHS[method_name],
        ls=LINE_STYLES[method_name],
        label=f"{label} (unstable)",
    )


def make_phase_comparison_figures(comparison_payload: dict[str, dict[str, object]]) -> None:
    bench.apply_plot_style()
    for dataset_name in DATASET_ORDER:
        payload = comparison_payload[dataset_name]
        fig = plt.figure(figsize=(9.0, 7.2), constrained_layout=True)
        axis = fig.add_subplot(111, projection="3d")

        truth_states = payload["truth_states"]
        axis.plot(
            truth_states[:, 0],
            truth_states[:, 1],
            truth_states[:, 2],
            color=TIME_SERIES_COLORS["truth"],
            lw=LINE_WIDTHS["truth"],
            ls=LINE_STYLES["truth"],
            label="Truth",
        )
        axis.scatter(*truth_states[0], color=TIME_SERIES_COLORS["truth"], s=26, marker=MODEL_MARKERS["truth"])

        for method_name in REDUCED_FIGURE_MODEL_ORDER:
            label = f"{MODEL_LABELS[method_name]} (NRMSE={payload['short_field_nrmse'][method_name]:.3f})"
            add_phase_curve(axis, payload["short_states"][method_name], method_name, label)

        axis.set_xlabel(r"$a_1$")
        axis.set_ylabel(r"$a_2$")
        axis.set_zlabel(r"$a_3$")
        axis.set_title(f"{payload['dataset_label']}\n3D Phase Portrait Comparison")
        axis.view_init(elev=23, azim=-58)
        axis.xaxis.pane.fill = False
        axis.yaxis.pane.fill = False
        axis.zaxis.pane.fill = False
        axis.legend(frameon=False, loc="upper left")

        slug = DATASET_SLUGS[dataset_name]
        fig.savefig(bench.FIGURES_DIR / f"vortex_phase_compare_{slug}.png")
        fig.savefig(bench.FIGURES_DIR / f"vortex_phase_compare_{slug}.pdf")
        plt.close(fig)


def make_separate_mode_benchmark_figures(comparison_payload: dict[str, dict[str, object]]) -> None:
    bench.apply_plot_style()
    for state_idx, state_name in enumerate(bench.STATE_NAMES):
        # Keep these benchmark panels narrow enough for the poster results block.
        fig, axes = plt.subplots(2, 1, figsize=(7.8, 4.9), constrained_layout=True, sharex=False)

        for row_idx, dataset_name in enumerate(DATASET_ORDER):
            payload = comparison_payload[dataset_name]
            axis = axes[row_idx]

            axis.plot(
                payload["time"],
                payload["truth_states"][:, state_idx],
                color=TIME_SERIES_COLORS["truth"],
                lw=LINE_WIDTHS["truth"],
                ls=LINE_STYLES["truth"],
                label="Truth",
            )

            for method_name in REDUCED_FIGURE_MODEL_ORDER:
                label = MODEL_LABELS[method_name]
                add_model_curve(
                    axis,
                    payload["time"],
                    payload["short_states"][method_name],
                    state_idx,
                    method_name,
                    label,
                )

            axis.set_ylabel(payload["dataset_label"])
            axis.set_title(rf"${state_name}(t)$", fontsize=13, pad=6)

            if row_idx == len(DATASET_ORDER) - 1:
                axis.set_xlabel("Time from test start")

            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.margins(x=0.02)

            axis.legend(frameon=False, loc="upper right", fontsize=8.2)
        fig.savefig(bench.FIGURES_DIR / f"vortex_benchmark_{state_name}.png")
        fig.savefig(bench.FIGURES_DIR / f"vortex_benchmark_{state_name}.pdf")
        plt.close(fig)


def make_modal_timeseries_figure(
    dataset_results: dict[str, dict[str, object]],
    genetic_payload: dict[str, dict[str, object]],
) -> None:
    bench.apply_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.4), sharex=False, constrained_layout=True)

    for row_idx, dataset_name in enumerate(DATASET_ORDER):
        dmd_result = dataset_results[dataset_name]["dmd_sindy"]
        pod_result = dataset_results[dataset_name]["pod_sindy"]
        raw_result = dataset_results[dataset_name]["raw_sindy"]
        truth_states = dmd_result["truth_states"]
        dmd_states = dmd_result["rollout_states"]
        pod_states_common = bench.project_fields(pod_result["predicted_fields"], dmd_result["reduced_model"])
        raw_states_common = bench.project_fields(raw_result["predicted_fields"], dmd_result["reduced_model"])
        genetic_states = genetic_payload[dataset_name]["states"]
        time = genetic_payload[dataset_name]["time"]
        dataset_label = dmd_result["dataset_label"]

        for col_idx, state_name in enumerate(bench.STATE_NAMES):
            axis = axes[row_idx, col_idx]
            axis.plot(
                time,
                truth_states[:, col_idx],
                color=TIME_SERIES_COLORS["truth"],
                lw=LINE_WIDTHS["truth"],
                ls=LINE_STYLES["truth"],
                label="Truth",
            )
            axis.plot(
                time,
                raw_states_common[:, col_idx],
                color=TIME_SERIES_COLORS["raw_sindy"],
                lw=LINE_WIDTHS["raw_sindy"],
                ls=LINE_STYLES["raw_sindy"],
                label="Pure SINDy",
            )
            axis.plot(
                time,
                pod_states_common[:, col_idx],
                color=TIME_SERIES_COLORS["pod_sindy"],
                lw=LINE_WIDTHS["pod_sindy"],
                ls=LINE_STYLES["pod_sindy"],
                label="POD+SINDy",
            )
            axis.plot(
                time,
                dmd_states[:, col_idx],
                color=TIME_SERIES_COLORS["dmd_sindy"],
                lw=LINE_WIDTHS["dmd_sindy"],
                ls=LINE_STYLES["dmd_sindy"],
                label="DMD+SINDy",
            )
            axis.plot(
                time,
                genetic_states[:, col_idx],
                color=TIME_SERIES_COLORS["genetic_sr"],
                lw=LINE_WIDTHS["genetic_sr"],
                ls=LINE_STYLES["genetic_sr"],
                label="Genetic SR",
            )
            if row_idx == 0:
                axis.set_title(rf"${state_name}(t)$")
            if col_idx == 0:
                axis.set_ylabel(dataset_label)
            if row_idx == len(DATASET_ORDER) - 1:
                axis.set_xlabel("Time")

            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        axes[row_idx, 2].legend(frameon=False, loc="upper right")

    fig.savefig(bench.FIGURES_DIR / "vortex_modal_timeseries.png")
    fig.savefig(bench.FIGURES_DIR / "vortex_modal_timeseries.pdf")
    plt.close(fig)


def main() -> None:
    bench.ensure_output_dirs()
    bundles = {bundle.name: bundle for bundle in bench.load_datasets()}

    final_payload: dict[str, dict[str, object]] = {}
    genetic_payload: dict[str, dict[str, object]] = {}
    comparison_payload: dict[str, dict[str, object]] = {}
    rows = []

    for dataset_name, configs in FINAL_CONFIGS.items():
        bundle = bundles[dataset_name]
        train, val, test = bench.split_dataset(bundle)
        final_payload[dataset_name] = {}

        for method_name, config in configs.items():
            result = bench.refit_and_evaluate(bundle, train, val, test, config, compute_long_horizon=False)
            final_payload[dataset_name][method_name] = result
            rows.append(
                {
                    "dataset": result["dataset"],
                    "dataset_label": result["dataset_label"],
                    "method": result["method"],
                    "method_label": result["method_label"],
                    "filter_rank": result["filter_rank"],
                    "library_name": result["library_name"],
                    "differentiation_name": result["differentiation_name"],
                    "threshold": result["threshold"],
                    "field_nrmse": result["field_nrmse"],
                    "state_rmse": result["state_rmse"],
                    "active_terms": result["active_terms"],
                    "interpretability_penalty": result["interpretability_penalty"],
                    "shift_corr": result["shift_corr"],
                    "reference_shift_corr": result["reference_shift_corr"],
                    "long_horizon_max_norm": result["long_horizon_max_norm"],
                    "stable": result["stable"],
                }
            )

        dmd_result = final_payload[dataset_name]["dmd_sindy"]
        raw_result = final_payload[dataset_name]["raw_sindy"]
        reduced_model = dmd_result["reduced_model"]
        combined_train_states = reduced_model.training_states
        combined_time = np.arange(combined_train_states.shape[0], dtype=float) * bundle.dt
        truth_states = dmd_result["truth_states"]
        clip_norm = max(15.0, 4.0 * float(np.max(np.linalg.norm(combined_train_states, axis=1))))
        sr_model = fit_genetic_sr_model(combined_train_states, combined_time, seed=37 if dataset_name == "kutz_cylinder" else 73)
        sr_states = simulate_genetic_sr_model(sr_model, truth_states[0], test.time, clip_norm=clip_norm)
        sr_fields = bench.reconstruct_fields(sr_states, reduced_model)
        sr_field_nrmse = bench.relative_field_error(sr_fields, test.snapshots)
        sr_stable = bool(np.all(np.isfinite(sr_states)))
        genetic_payload[dataset_name] = {
            "dataset_label": bundle.title,
            "time": test.time,
            "states": sr_states,
            "field_nrmse": sr_field_nrmse,
            "stable": sr_stable,
            "equations": genetic_sr_equations(sr_model),
        }
        comparison_payload[dataset_name] = build_comparison_payload(bundle, test, final_payload[dataset_name], sr_states, sr_model)

    summary_df = pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)
    summary_df.to_csv(bench.RESULTS_DIR / "benchmark_summary.csv", index=False)
    bench.write_equation_report(final_payload)
    write_genetic_sr_report(genetic_payload)
    bench.make_poster_figure(final_payload, summary_df)
    make_latent_mode_figures(bundles, final_payload)
    make_modal_timeseries_figure(final_payload, genetic_payload)
    make_snapshot_figures(bundles)
    make_phase_comparison_figures(comparison_payload)
    make_separate_mode_benchmark_figures(comparison_payload)


if __name__ == "__main__":
    main()
