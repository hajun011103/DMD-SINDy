from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps
from pydmd import DMD
from sklearn.exceptions import ConvergenceWarning
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from scipy.linalg import svd


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Input data condition number")
warnings.filterwarnings("ignore", message="Sparsity parameter is too big")

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
DATA_ROOT = Path(os.environ.get("VORTEX_BENCHMARK_DATA_ROOT", str(REPO_ROOT / "data" / "benchmarks")))

METHOD_LABELS = {
    "raw_sindy": "Pure SINDy",
    "pod_sindy": "POD+SINDy",
    "dmd_sindy": "DMD+SINDy",
}

METHOD_COLORS = {
    "raw_sindy": "#C2410C",
    "pod_sindy": "#0F766E",
    "dmd_sindy": "#1D4ED8",
}

STATE_NAMES = ["a1", "a2", "a3"]
LAMBDA_GRID = np.geomspace(1.0e-3, 1.0, 10)
REDUCED_MODEL_CACHE: dict[tuple[str, str, int, int], ReducedModel] = {}
MEANFIELD_ALLOWED_FEATURES = [
    {"a1", "a2", "a1 a3"},
    {"a1", "a2", "a2 a3"},
    {"a3", "a1^2", "a2^2"},
]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    title: str
    time: np.ndarray
    snapshots: np.ndarray
    dt: float
    plot_kind: str
    grid_shape: tuple[int, int] | None = None
    coords: np.ndarray | None = None
    reference_shift_mode: np.ndarray | None = None
    reference_pair_basis: np.ndarray | None = None


@dataclass(frozen=True)
class SplitData:
    time: np.ndarray
    snapshots: np.ndarray


@dataclass
class ReducedModel:
    mean_field: np.ndarray
    basis: np.ndarray
    training_states: np.ndarray
    filtered_training_fields: np.ndarray
    shift_correlation: float
    reference_shift_correlation: float


@dataclass
class FittedSINDyModel:
    model: ps.SINDy
    state_mean: np.ndarray
    state_scale: np.ndarray


@dataclass(frozen=True)
class MethodConfig:
    method_name: str
    filter_rank: int
    library_name: str
    differentiation_name: str
    threshold: float


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)


def load_kutz_dataset() -> DatasetBundle:
    payload = loadmat(
        DATA_ROOT / "kutz_cylinder" / "CYLINDER_ALL.mat",
        variable_names=["VORTALL", "nx", "ny"],
    )
    basis_payload = loadmat(
        DATA_ROOT / "kutz_cylinder" / "CYLINDER_basis.mat",
        variable_names=["vortAVG", "vortUNSTEADY", "vortPHI"],
    )
    snapshots = payload["VORTALL"].T.astype(float)
    time = np.arange(snapshots.shape[0], dtype=float)
    nx = int(payload["nx"].ravel()[0])
    ny = int(payload["ny"].ravel()[0])
    reference_shift = (basis_payload["vortAVG"] - basis_payload["vortUNSTEADY"]).ravel().astype(float)
    reference_pair_basis = basis_payload["vortPHI"][:, :2].astype(float)
    return DatasetBundle(
        name="kutz_cylinder",
        title="Kutz Cylinder (vorticity)",
        time=time,
        snapshots=snapshots,
        dt=1.0,
        plot_kind="grid",
        grid_shape=(ny, nx),
        reference_shift_mode=reference_shift,
        reference_pair_basis=reference_pair_basis,
    )


def load_deepxde_dataset() -> DatasetBundle:
    payload = loadmat(
        DATA_ROOT / "deepxde_cylinder" / "cylinder_nektar_wake.mat",
        variable_names=["U_star", "t", "X_star"],
    )
    velocity = payload["U_star"]
    snapshots = np.concatenate([velocity[:, 0, :].T, velocity[:, 1, :].T], axis=1).astype(float)
    time = payload["t"].ravel().astype(float)
    dt = float(np.median(np.diff(time)))
    coords = payload["X_star"].astype(float)
    return DatasetBundle(
        name="deepxde_cylinder",
        title="DeepXDE Cylinder (u,v)",
        time=time,
        snapshots=snapshots,
        dt=dt,
        plot_kind="point_cloud",
        coords=coords,
    )


def load_datasets() -> list[DatasetBundle]:
    return [load_kutz_dataset(), load_deepxde_dataset()]


def split_dataset(bundle: DatasetBundle) -> tuple[SplitData, SplitData, SplitData]:
    n_samples = bundle.time.size
    train_end = max(8, int(round(0.6 * n_samples)))
    val_end = max(train_end + 8, int(round(0.8 * n_samples)))
    val_end = min(val_end, n_samples - 8)
    train = SplitData(
        time=bundle.time[:train_end] - bundle.time[0],
        snapshots=bundle.snapshots[:train_end],
    )
    val = SplitData(
        time=bundle.time[train_end:val_end] - bundle.time[train_end],
        snapshots=bundle.snapshots[train_end:val_end],
    )
    test = SplitData(
        time=bundle.time[val_end:] - bundle.time[val_end],
        snapshots=bundle.snapshots[val_end:],
    )
    return train, val, test


def relative_field_error(prediction: np.ndarray, truth: np.ndarray) -> float:
    if not np.all(np.isfinite(prediction)):
        return float("inf")
    numerator = np.linalg.norm(prediction - truth)
    denominator = max(np.linalg.norm(truth), 1.0e-12)
    return float(numerator / denominator)


def coefficient_rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    if not np.all(np.isfinite(prediction)):
        return float("inf")
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def orthonormalize(candidate: np.ndarray, basis_vectors: list[np.ndarray]) -> np.ndarray:
    vector = candidate.astype(float).copy()
    for basis in basis_vectors:
        vector -= np.dot(vector, basis) * basis
    norm = np.linalg.norm(vector)
    if norm < 1.0e-10:
        raise ValueError("Degenerate mode encountered during orthonormalization.")
    return vector / norm


def svht_rank(singular_values: np.ndarray, n_rows: int, n_cols: int) -> int:
    beta = min(n_rows, n_cols) / max(n_rows, n_cols)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    threshold = omega * np.median(singular_values)
    rank = int(np.sum(singular_values > threshold))
    return max(rank, 0)


def singular_spectrum_summary(bundle: DatasetBundle, train: SplitData) -> tuple[pd.DataFrame, int]:
    centered = train.snapshots - train.snapshots.mean(axis=0, keepdims=True)
    _, singular_values, vh = svd(centered, full_matrices=False)
    energy = singular_values**2
    energy_fraction = energy / max(np.sum(energy), 1.0e-12)
    cumulative_energy = np.cumsum(energy_fraction)

    coefficients = centered @ vh[:3].T
    rho2 = coefficients[:, 0] ** 2 + coefficients[:, 1] ** 2
    shift_corr = np.corrcoef(coefficients[:, 2], rho2)[0, 1] if coefficients.shape[1] >= 3 else np.nan

    baseline_rank = max(3, min(10, svht_rank(singular_values, *centered.shape)))
    rows = []
    for mode_idx, sigma in enumerate(singular_values[:10], start=1):
        rows.append(
            {
                "dataset": bundle.name,
                "dataset_label": bundle.title,
                "mode": mode_idx,
                "singular_value": float(sigma),
                "energy_fraction": float(energy_fraction[mode_idx - 1]),
                "cumulative_energy": float(cumulative_energy[mode_idx - 1]),
                "baseline_rank_svht": baseline_rank,
                "mode3_shift_corr": float(shift_corr),
            }
        )
    return pd.DataFrame(rows), baseline_rank


def pod_filter(centered_fields: np.ndarray, rank: int) -> np.ndarray:
    u, singular_values, vh = svd(centered_fields, full_matrices=False)
    retained_rank = max(1, min(rank, singular_values.size))
    return (u[:, :retained_rank] * singular_values[:retained_rank]) @ vh[:retained_rank]


def dmd_filter(centered_fields: np.ndarray, rank: int) -> np.ndarray:
    retained_rank = max(1, min(rank, centered_fields.shape[0] - 1))
    try:
        dmd = DMD(
            svd_rank=retained_rank,
            tlsq_rank=retained_rank,
            exact=False,
            opt=True,
            forward_backward=True,
        )
        dmd.fit(centered_fields.T)
        reconstruction = np.real(np.asarray(dmd.reconstructed_data).T)
        if reconstruction.shape != centered_fields.shape or not np.all(np.isfinite(reconstruction)):
            return pod_filter(centered_fields, retained_rank)
        return reconstruction
    except Exception:
        return pod_filter(centered_fields, retained_rank)


def preprocess_training_fields(centered_fields: np.ndarray, method_name: str, rank: int) -> np.ndarray:
    if method_name == "raw_sindy":
        return centered_fields.copy()
    if method_name == "pod_sindy":
        return pod_filter(centered_fields, rank)
    if method_name == "dmd_sindy":
        return dmd_filter(centered_fields, rank)
    raise ValueError(f"Unknown method: {method_name}")


def construct_shift_mode(
    raw_centered_fields: np.ndarray,
    oscillatory_basis: np.ndarray,
    smoothed_fields: np.ndarray,
) -> tuple[np.ndarray, float]:
    oscillatory_states = smoothed_fields @ oscillatory_basis
    rho2 = oscillatory_states[:, 0] ** 2 + oscillatory_states[:, 1] ** 2
    rho2_centered = rho2 - rho2.mean()

    residual = raw_centered_fields - (raw_centered_fields @ oscillatory_basis) @ oscillatory_basis.T
    numerator = rho2_centered @ residual
    denominator = max(float(rho2_centered @ rho2_centered), 1.0e-12)
    shift_mode = numerator / denominator
    shift_mode = orthonormalize(shift_mode, [oscillatory_basis[:, 0], oscillatory_basis[:, 1]])

    shift_states = raw_centered_fields @ shift_mode
    shift_corr = np.corrcoef(shift_states, rho2)[0, 1]
    if shift_corr < 0.0:
        shift_mode *= -1.0
        shift_states *= -1.0
        shift_corr *= -1.0
    return shift_mode, float(shift_corr)


def build_reduced_model(bundle: DatasetBundle, training_fields: np.ndarray, method_name: str, rank: int) -> ReducedModel:
    cache_key = (bundle.name, method_name, int(rank), int(training_fields.shape[0]))
    if cache_key in REDUCED_MODEL_CACHE:
        return REDUCED_MODEL_CACHE[cache_key]

    mean_field = training_fields.mean(axis=0, keepdims=True)
    raw_centered = training_fields - mean_field
    filtered_centered = preprocess_training_fields(raw_centered, method_name, rank)

    if bundle.reference_pair_basis is not None:
        ref_mode_1 = orthonormalize(bundle.reference_pair_basis[:, 0], [])
        ref_mode_2 = orthonormalize(bundle.reference_pair_basis[:, 1], [ref_mode_1])
        oscillatory_basis = np.column_stack([ref_mode_1, ref_mode_2])
    else:
        _, _, vh = svd(filtered_centered, full_matrices=False)
        oscillatory_basis = vh[:2].T
        for idx in range(2):
            oscillatory_basis[:, idx] = orthonormalize(oscillatory_basis[:, idx], [oscillatory_basis[:, j] for j in range(idx)])

    if bundle.reference_shift_mode is not None:
        shift_mode = orthonormalize(bundle.reference_shift_mode, [oscillatory_basis[:, 0], oscillatory_basis[:, 1]])
        rho2 = np.sum((filtered_centered @ oscillatory_basis) ** 2, axis=1)
        shift_states = raw_centered @ shift_mode
        shift_corr = float(np.corrcoef(shift_states, rho2)[0, 1])
        if shift_corr < 0.0:
            shift_mode *= -1.0
            shift_corr *= -1.0
        reference_corr = 1.0
    else:
        shift_mode, shift_corr = construct_shift_mode(raw_centered, oscillatory_basis, filtered_centered)
        reference_corr = float("nan")

    basis = np.column_stack([oscillatory_basis[:, 0], oscillatory_basis[:, 1], shift_mode])
    oscillatory_states = filtered_centered @ oscillatory_basis
    if method_name == "dmd_sindy":
        # Keep DMD for the shedding pair, but preserve the raw shift-mode amplitude.
        shift_states = raw_centered @ shift_mode
        training_states = np.column_stack([oscillatory_states, shift_states])
    else:
        training_states = np.column_stack([oscillatory_states, filtered_centered @ shift_mode])
    reduced_model = ReducedModel(
        mean_field=mean_field,
        basis=basis,
        training_states=training_states,
        filtered_training_fields=mean_field + training_states @ basis.T,
        shift_correlation=shift_corr,
        reference_shift_correlation=reference_corr,
    )
    REDUCED_MODEL_CACHE[cache_key] = reduced_model
    return reduced_model


def project_fields(fields: np.ndarray, reduced_model: ReducedModel) -> np.ndarray:
    return (fields - reduced_model.mean_field) @ reduced_model.basis


def reconstruct_fields(states: np.ndarray, reduced_model: ReducedModel) -> np.ndarray:
    return reduced_model.mean_field + states @ reduced_model.basis.T


def build_library(library_name: str):
    if library_name == "poly2":
        return ps.PolynomialLibrary(degree=2, include_interaction=True, include_bias=True)
    if library_name == "poly3":
        return ps.PolynomialLibrary(degree=3, include_interaction=True, include_bias=True)
    if library_name == "meanfield_poly2":
        return ps.PolynomialLibrary(degree=2, include_interaction=True, include_bias=True)
    if library_name == "poly2_trig":
        return ps.ConcatLibrary(
            [
                ps.PolynomialLibrary(degree=2, include_interaction=True, include_bias=True),
                ps.FourierLibrary(n_frequencies=1),
            ]
        )
    if library_name == "poly3_trig":
        return ps.ConcatLibrary(
            [
                ps.PolynomialLibrary(degree=3, include_interaction=True, include_bias=True),
                ps.FourierLibrary(n_frequencies=1),
            ]
        )
    raise ValueError(f"Unsupported library: {library_name}")


def build_differentiator(name: str):
    if name == "finite_difference":
        return ps.FiniteDifference(order=2, is_uniform=True)
    if name == "spectral":
        return ps.SpectralDerivative()
    if name == "tvregdiff":
        return ps.SINDyDerivative(kind="trend_filtered", order=0, alpha=1.0e-2)
    raise ValueError(f"Unsupported differentiator: {name}")


def support_prior_indices(feature_library, states: np.ndarray, library_name: str) -> list[int] | None:
    if library_name != "meanfield_poly2":
        return None

    feature_library.fit(states)
    feature_names = feature_library.get_feature_names(input_features=STATE_NAMES)
    required_terms = set().union(*MEANFIELD_ALLOWED_FEATURES)
    return [idx for idx, feature_name in enumerate(feature_names) if feature_name not in required_terms]


def refit_meanfield_support(
    model: ps.SINDy,
    scaled_states: np.ndarray,
    dt: float,
    differentiation_name: str,
) -> None:
    feature_names = model.get_feature_names()
    derivatives = np.asarray(build_differentiator(differentiation_name)._differentiate(scaled_states, t=dt), dtype=float)
    design_matrix = np.asarray(model.feature_library.fit_transform(scaled_states), dtype=float)
    constrained = np.zeros_like(model.coefficients())
    ridge = 1.0e-6

    for state_idx, allowed in enumerate(MEANFIELD_ALLOWED_FEATURES):
        keep = [idx for idx, name in enumerate(feature_names) if name in allowed]
        if not keep:
            continue
        theta = design_matrix[:, keep]
        gram = theta.T @ theta + ridge * np.eye(theta.shape[1])
        rhs = theta.T @ derivatives[:, state_idx]
        constrained[state_idx, keep] = np.linalg.solve(gram, rhs)

    model.optimizer.coef_ = constrained


def fit_sindy_model(states: np.ndarray, dt: float, library_name: str, differentiation_name: str, threshold: float) -> FittedSINDyModel:
    state_mean = states.mean(axis=0, keepdims=True)
    state_scale = states.std(axis=0, ddof=1, keepdims=True)
    state_scale[state_scale == 0.0] = 1.0
    scaled_states = (states - state_mean) / state_scale
    feature_library = build_library(library_name)
    sparse_ind = None

    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=ps.STLSQ(
            threshold=threshold,
            alpha=1.0e-4,
            max_iter=60,
            normalize_columns=sparse_ind is None,
            sparse_ind=sparse_ind,
            unbias=sparse_ind is None,
        ),
        differentiation_method=build_differentiator(differentiation_name),
    )
    model.fit(scaled_states, t=dt, feature_names=STATE_NAMES)
    if library_name == "meanfield_poly2":
        refit_meanfield_support(model, scaled_states, dt, differentiation_name)
    return FittedSINDyModel(model=model, state_mean=state_mean, state_scale=state_scale)


def predict_scaled_derivative(fitted_model: FittedSINDyModel, state: np.ndarray) -> np.ndarray:
    return np.asarray(fitted_model.model.predict(state.reshape(1, -1))[0], dtype=float)


def simulate_sindy_model_solve_ivp(
    fitted_model: FittedSINDyModel,
    initial_state: np.ndarray,
    time: np.ndarray,
    clip_norm: float,
) -> np.ndarray:
    if time.size == 0:
        return np.empty((0, initial_state.size))

    initial_state_scaled = ((np.asarray(initial_state, dtype=float).reshape(1, -1) - fitted_model.state_mean) / fitted_model.state_scale)[0]

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        return predict_scaled_derivative(fitted_model, state)

    solution = solve_ivp(
        rhs,
        (float(time[0]), float(time[-1])),
        initial_state_scaled,
        method="RK45",
        t_eval=time,
        max_step=max(float(np.median(np.diff(time))), 1.0e-9),
        rtol=1.0e-7,
        atol=1.0e-9,
    )
    if not solution.success or solution.y.shape[1] != time.size:
        return np.full((time.size, initial_state.size), np.nan, dtype=float)
    trajectory = solution.y.T * fitted_model.state_scale + fitted_model.state_mean
    state_norm = np.linalg.norm(trajectory, axis=1)
    if np.any(~np.isfinite(trajectory)) or float(np.nanmax(state_norm)) > clip_norm:
        return np.full((time.size, initial_state.size), np.nan, dtype=float)
    return trajectory


def simulate_sindy_model_rk4(
    fitted_model: FittedSINDyModel,
    initial_state: np.ndarray,
    time: np.ndarray,
    clip_norm: float,
) -> np.ndarray:
    if time.size == 0:
        return np.empty((0, initial_state.size))

    initial_state_scaled = ((np.asarray(initial_state, dtype=float).reshape(1, -1) - fitted_model.state_mean) / fitted_model.state_scale)[0]
    states = np.empty((time.size, initial_state_scaled.size), dtype=float)
    states[0] = initial_state_scaled

    for idx in range(1, time.size):
        dt = float(time[idx] - time[idx - 1])
        current = states[idx - 1]
        k1 = predict_scaled_derivative(fitted_model, current)
        k2 = predict_scaled_derivative(fitted_model, current + 0.5 * dt * k1)
        k3 = predict_scaled_derivative(fitted_model, current + 0.5 * dt * k2)
        k4 = predict_scaled_derivative(fitted_model, current + dt * k3)
        next_state = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        unscaled = next_state * fitted_model.state_scale[0] + fitted_model.state_mean[0]
        if not np.all(np.isfinite(unscaled)) or float(np.linalg.norm(unscaled)) > clip_norm:
            return np.full((time.size, initial_state.size), np.nan, dtype=float)
        states[idx] = next_state

    trajectory = states * fitted_model.state_scale + fitted_model.state_mean
    if np.any(~np.isfinite(trajectory)):
        return np.full((time.size, initial_state.size), np.nan, dtype=float)
    return trajectory


def count_active_terms(fitted_model: FittedSINDyModel, tol: float = 1.0e-10) -> int:
    coefficients = np.asarray(fitted_model.model.coefficients(), dtype=float)
    return int(np.sum(np.abs(coefficients) > tol))


def coefficient_lookup(fitted_model: FittedSINDyModel) -> dict[str, np.ndarray]:
    feature_names = fitted_model.model.get_feature_names()
    coefficients = np.asarray(fitted_model.model.coefficients(), dtype=float)
    return {feature: coefficients[:, idx] for idx, feature in enumerate(feature_names)}


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def shift_closure_metrics(states: np.ndarray) -> tuple[float, float]:
    if states.shape[1] < 3:
        return float("nan"), float("nan")
    rho2 = states[:, 0] ** 2 + states[:, 1] ** 2
    return safe_correlation(states[:, 2], rho2), float(np.std(states[:, 2], ddof=1))


def safe_relative_difference(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1.0e-8)
    return abs(a - b) / scale


def interpretability_penalty(fitted_model: FittedSINDyModel) -> float:
    lookup = coefficient_lookup(fitted_model)
    feature_names = fitted_model.model.get_feature_names()
    coefficients = np.asarray(fitted_model.model.coefficients(), dtype=float)
    active_mask = np.abs(coefficients) > 1.0e-8

    allowed = [
        {"a1", "a2", "a1 a3"},
        {"a1", "a2", "a2 a3"},
        {"a3", "a1^2", "a2^2"},
    ]
    core = [
        {"a1", "a2", "a1 a3"},
        {"a1", "a2", "a2 a3"},
        {"a3", "a1^2", "a2^2"},
    ]

    extra_terms = 0
    missing_core = 0
    for state_idx in range(3):
        active_features = {name for name, is_active in zip(feature_names, active_mask[state_idx]) if is_active}
        extra_terms += len(active_features - allowed[state_idx])
        missing_core += len(core[state_idx] - active_features)

    eq1 = lookup.get("a1", np.zeros(3))[0]
    eq2 = lookup.get("a2", np.zeros(3))[1]
    cross12 = lookup.get("a2", np.zeros(3))[0]
    cross21 = lookup.get("a1", np.zeros(3))[1]
    shift12 = lookup.get("a1 a3", np.zeros(3))[0]
    shift21 = lookup.get("a2 a3", np.zeros(3))[1]
    quad31 = lookup.get("a1^2", np.zeros(3))[2]
    quad32 = lookup.get("a2^2", np.zeros(3))[2]
    shift33 = lookup.get("a3", np.zeros(3))[2]

    symmetry_penalty = 0.0
    symmetry_penalty += safe_relative_difference(eq1, eq2)
    symmetry_penalty += safe_relative_difference(cross12, -cross21)
    symmetry_penalty += safe_relative_difference(shift12, shift21)
    symmetry_penalty += safe_relative_difference(quad31, quad32)

    sign_penalty = 0.0
    if shift12 >= 0.0:
        sign_penalty += 1.0
    if shift21 >= 0.0:
        sign_penalty += 1.0
    if shift33 >= 0.0:
        sign_penalty += 1.0
    if quad31 <= 0.0:
        sign_penalty += 1.0
    if quad32 <= 0.0:
        sign_penalty += 1.0

    return float(extra_terms + missing_core + symmetry_penalty + sign_penalty)


def long_horizon_stability(
    fitted_model: FittedSINDyModel,
    initial_state: np.ndarray,
    dt: float,
    training_states: np.ndarray,
) -> tuple[float, bool]:
    training_norm = float(np.max(np.linalg.norm(training_states, axis=1)))
    clip_norm = max(15.0, 4.0 * training_norm)
    n_points = max(240, 3 * training_states.shape[0])
    horizon = dt * (n_points - 1)
    time = np.linspace(0.0, horizon, n_points)
    trajectory = simulate_sindy_model_solve_ivp(fitted_model, initial_state, time, clip_norm=clip_norm)
    stable = np.all(np.isfinite(trajectory))
    max_norm = float(np.nanmax(np.linalg.norm(trajectory, axis=1))) if stable else float("inf")
    return max_norm, stable


def pareto_efficient(errors: np.ndarray, complexity: np.ndarray) -> np.ndarray:
    n_points = errors.size
    efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not efficient[i]:
            continue
        dominates = (errors <= errors[i]) & (complexity <= complexity[i]) & (
            (errors < errors[i]) | (complexity < complexity[i])
        )
        dominates[i] = False
        efficient[dominates] = False
    return efficient


def evaluate_configuration(
    bundle: DatasetBundle,
    train: SplitData,
    evaluation_split: SplitData,
    method_name: str,
    filter_rank: int,
    library_name: str,
    differentiation_name: str,
    threshold: float,
    compute_long_horizon: bool = True,
) -> dict[str, object]:
    try:
        reduced = build_reduced_model(bundle, train.snapshots, method_name, filter_rank)
        fitted_model = fit_sindy_model(reduced.training_states, bundle.dt, library_name, differentiation_name, threshold)

        truth_states = project_fields(evaluation_split.snapshots, reduced)
        initial_state = truth_states[0]
        training_norm = float(np.max(np.linalg.norm(reduced.training_states, axis=1)))
        simulator = simulate_sindy_model_solve_ivp if compute_long_horizon else simulate_sindy_model_rk4
        rollout_states = simulator(
            fitted_model,
            initial_state,
            evaluation_split.time,
            clip_norm=max(15.0, 4.0 * training_norm),
        )
        predicted_fields = reconstruct_fields(rollout_states, reduced)

        stable = bool(np.all(np.isfinite(rollout_states)))
        max_norm = float(np.nanmax(np.linalg.norm(rollout_states, axis=1))) if stable else float("inf")
        if compute_long_horizon and stable:
            max_norm, stable = long_horizon_stability(fitted_model, initial_state, bundle.dt, reduced.training_states)
        active_terms = count_active_terms(fitted_model)
        penalty = interpretability_penalty(fitted_model)
        field_error = relative_field_error(predicted_fields, evaluation_split.snapshots)
        state_error = coefficient_rmse(rollout_states, truth_states)
        truth_shift_corr, truth_a3_scale = shift_closure_metrics(truth_states)
        model_shift_corr, _ = shift_closure_metrics(rollout_states)
        a3_rmse = float(
            np.sqrt(np.mean((rollout_states[:, 2] - truth_states[:, 2]) ** 2)) / max(truth_a3_scale, 1.0e-6)
        )
        shift_gap = 1.0
        if np.isfinite(truth_shift_corr) and np.isfinite(model_shift_corr):
            shift_gap = abs(model_shift_corr - truth_shift_corr)
        shift_corr_penalty = max(0.0, 0.75 - float(reduced.shift_correlation))
        score = (
            field_error
            + 0.015 * active_terms
            + 0.15 * penalty
            + 0.18 * a3_rmse
            + 0.15 * shift_gap
            + 0.20 * shift_corr_penalty
            + (0.0 if stable else 100.0)
        )

        return {
            "dataset": bundle.name,
            "dataset_label": bundle.title,
            "method": method_name,
            "method_label": METHOD_LABELS[method_name],
            "filter_rank": filter_rank,
            "library_name": library_name,
            "differentiation_name": differentiation_name,
            "threshold": threshold,
            "field_nrmse": field_error,
            "state_rmse": state_error,
            "a3_nrmse": a3_rmse,
            "active_terms": active_terms,
            "interpretability_penalty": penalty,
            "shift_corr": reduced.shift_correlation,
            "reference_shift_corr": reduced.reference_shift_correlation,
            "closure_truth_corr": truth_shift_corr,
            "closure_model_corr": model_shift_corr,
            "long_horizon_max_norm": max_norm,
            "stable": stable,
            "score": score,
            "reduced_model": reduced,
            "model": fitted_model,
            "truth_states": truth_states,
            "rollout_states": rollout_states,
            "predicted_fields": predicted_fields,
        }
    except Exception:
        zero_states = np.full((evaluation_split.time.size, 3), np.nan, dtype=float)
        return {
            "dataset": bundle.name,
            "dataset_label": bundle.title,
            "method": method_name,
            "method_label": METHOD_LABELS[method_name],
            "filter_rank": filter_rank,
            "library_name": library_name,
            "differentiation_name": differentiation_name,
            "threshold": threshold,
            "field_nrmse": float("inf"),
            "state_rmse": float("inf"),
            "a3_nrmse": float("inf"),
            "active_terms": 999,
            "interpretability_penalty": 999.0,
            "shift_corr": float("nan"),
            "reference_shift_corr": float("nan"),
            "closure_truth_corr": float("nan"),
            "closure_model_corr": float("nan"),
            "long_horizon_max_norm": float("inf"),
            "stable": False,
            "score": float("inf"),
            "reduced_model": None,
            "model": None,
            "truth_states": zero_states,
            "rollout_states": zero_states,
            "predicted_fields": np.full_like(evaluation_split.snapshots, np.nan),
        }


def choose_baseline_threshold(
    bundle: DatasetBundle,
    train: SplitData,
    val: SplitData,
    method_name: str,
    filter_rank: int,
) -> tuple[MethodConfig, pd.DataFrame]:
    rows = []
    for threshold in LAMBDA_GRID:
        result = evaluate_configuration(
            bundle=bundle,
            train=train,
            evaluation_split=val,
            method_name=method_name,
            filter_rank=filter_rank,
            library_name="poly3",
            differentiation_name="tvregdiff",
            threshold=float(threshold),
            compute_long_horizon=False,
        )
        row = {
            "dataset": bundle.name,
            "method": method_name,
            "stage": "baseline",
            "filter_rank": filter_rank,
            "library_name": "poly3",
            "differentiation_name": "tvregdiff",
            "threshold": float(threshold),
            "field_nrmse": result["field_nrmse"],
            "state_rmse": result["state_rmse"],
            "active_terms": result["active_terms"],
            "interpretability_penalty": result["interpretability_penalty"],
            "shift_corr": result["shift_corr"],
            "reference_shift_corr": result["reference_shift_corr"],
            "long_horizon_max_norm": result["long_horizon_max_norm"],
            "stable": result["stable"],
            "score": result["score"],
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    best_idx = int(summary["score"].astype(float).idxmin())
    best = summary.loc[best_idx]
    config = MethodConfig(
        method_name=method_name,
        filter_rank=int(best["filter_rank"]),
        library_name=str(best["library_name"]),
        differentiation_name=str(best["differentiation_name"]),
        threshold=float(best["threshold"]),
    )
    return config, summary


def optimize_dmd_method(bundle: DatasetBundle, train: SplitData, val: SplitData) -> tuple[MethodConfig, pd.DataFrame]:
    search_rows = []
    library_names = ["poly2", "poly3", "meanfield_poly2", "poly2_trig", "poly3_trig"]
    differentiation_names = ["finite_difference", "tvregdiff", "spectral"]
    max_rank = min(20, max(3, train.snapshots.shape[0] - 2))

    for filter_rank in range(3, max_rank + 1):
        print(f"[search] {bundle.name} rank={filter_rank}", flush=True)
        for library_name in library_names:
            for differentiation_name in differentiation_names:
                for threshold in LAMBDA_GRID:
                    result = evaluate_configuration(
                        bundle=bundle,
                        train=train,
                        evaluation_split=val,
                        method_name="dmd_sindy",
                        filter_rank=filter_rank,
                        library_name=library_name,
                        differentiation_name=differentiation_name,
                        threshold=float(threshold),
                        compute_long_horizon=False,
                    )
                    search_rows.append(
                        {
                            "dataset": bundle.name,
                            "method": "dmd_sindy",
                            "stage": "optimization",
                            "filter_rank": filter_rank,
                            "library_name": library_name,
                            "differentiation_name": differentiation_name,
                            "threshold": float(threshold),
                            "field_nrmse": result["field_nrmse"],
                            "state_rmse": result["state_rmse"],
                            "active_terms": result["active_terms"],
                            "interpretability_penalty": result["interpretability_penalty"],
                            "shift_corr": result["shift_corr"],
                            "reference_shift_corr": result["reference_shift_corr"],
                            "long_horizon_max_norm": result["long_horizon_max_norm"],
                            "stable": result["stable"],
                            "score": result["score"],
                        }
                    )

    summary = pd.DataFrame(search_rows).sort_values(["score", "field_nrmse", "active_terms"]).reset_index(drop=True)
    finite = np.isfinite(summary["field_nrmse"].to_numpy(dtype=float))
    complexity = summary["active_terms"].to_numpy(dtype=float)
    pareto_mask = np.zeros(summary.shape[0], dtype=bool)
    pareto_mask[finite] = pareto_efficient(summary.loc[finite, "field_nrmse"].to_numpy(dtype=float), complexity[finite])
    summary["pareto_efficient"] = pareto_mask

    candidates = summary[summary["stable"] & summary["pareto_efficient"]]
    if candidates.empty:
        candidates = summary[summary["stable"]]
    if candidates.empty:
        candidates = summary

    best = candidates.sort_values(["score", "field_nrmse", "active_terms"]).iloc[0]
    config = MethodConfig(
        method_name="dmd_sindy",
        filter_rank=int(best["filter_rank"]),
        library_name=str(best["library_name"]),
        differentiation_name=str(best["differentiation_name"]),
        threshold=float(best["threshold"]),
    )
    return config, summary


def refit_and_evaluate(
    bundle: DatasetBundle,
    train: SplitData,
    val: SplitData,
    test: SplitData,
    config: MethodConfig,
    compute_long_horizon: bool = True,
) -> dict[str, object]:
    combined_time = np.concatenate([train.time, train.time[-1] + bundle.dt + val.time])
    del combined_time
    combined_snapshots = np.vstack([train.snapshots, val.snapshots])
    combined_train = SplitData(time=np.arange(combined_snapshots.shape[0], dtype=float) * bundle.dt, snapshots=combined_snapshots)
    return evaluate_configuration(
        bundle=bundle,
        train=combined_train,
        evaluation_split=test,
        method_name=config.method_name,
        filter_rank=config.filter_rank,
        library_name=config.library_name,
        differentiation_name=config.differentiation_name,
        threshold=config.threshold,
        compute_long_horizon=compute_long_horizon,
    )


def equation_strings(fitted_model: FittedSINDyModel) -> list[str]:
    feature_names = fitted_model.model.get_feature_names()
    coefficients = np.asarray(fitted_model.model.coefficients(), dtype=float)
    equations = []
    for state_idx, state_name in enumerate(STATE_NAMES):
        terms = []
        for feature_name, coefficient in zip(feature_names, coefficients[state_idx]):
            if abs(coefficient) < 1.0e-8:
                continue
            if feature_name == "1":
                terms.append(f"{coefficient:+.4f}")
            else:
                terms.append(f"{coefficient:+.4f} {feature_name}")
        rhs = " ".join(terms).replace("+ -", "- ").lstrip("+ ").strip()
        equations.append(f"d{state_name}/dt = {rhs if rhs else '0'}")
    return equations


def write_equation_report(dataset_results: dict[str, dict[str, object]]) -> None:
    lines = ["# DMD+SINDy Equation Report", ""]
    for dataset_name, payload in dataset_results.items():
        result = payload["dmd_sindy"]
        if result["model"] is None:
            continue
        lines.append(f"## {result['dataset_label']}")
        lines.append("")
        lines.append(
            f"- Hyperparameters: rank={result['filter_rank']}, library={result['library_name']}, "
            f"diff={result['differentiation_name']}, lambda={result['threshold']:.4g}"
        )
        lines.append(
            f"- Test field NRMSE: {result['field_nrmse']:.4f}; active terms: {result['active_terms']}; "
            f"shift correlation: {result['shift_corr']:.3f}"
        )
        if np.isfinite(result["reference_shift_corr"]):
            lines.append(f"- Reference shift-mode alignment: {result['reference_shift_corr']:.3f}")
        lines.append("- Equations:")
        for equation in equation_strings(result["model"]):
            lines.append(f"  - {equation}")
        lines.append("")
    (RESULTS_DIR / "dmd_equations.md").write_text("\n".join(lines), encoding="utf-8")


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
        }
    )


def plot_mode_panel(ax_modes: list[plt.Axes], bundle: DatasetBundle, reduced_model: ReducedModel) -> None:
    mode_labels = ["Mode 1", "Mode 2", "Shift Mode"]
    for idx, axis in enumerate(ax_modes):
        mode = reduced_model.basis[:, idx]
        vmax = np.max(np.abs(mode))
        if bundle.plot_kind == "grid":
            field = mode.reshape(bundle.grid_shape)
            image = axis.imshow(field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        else:
            coords = bundle.coords
            image = axis.tricontourf(
                coords[:, 0],
                coords[:, 1],
                mode[: coords.shape[0]],
                levels=30,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
        axis.set_title(mode_labels[idx])
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    return image


def plot_phase_panel(axis, truth_states: np.ndarray, rollout_states: np.ndarray, title: str) -> None:
    axis.plot(truth_states[:, 0], truth_states[:, 1], truth_states[:, 2], color="#0F172A", lw=2.0, label="Truth")
    axis.plot(
        rollout_states[:, 0],
        rollout_states[:, 1],
        rollout_states[:, 2],
        color=METHOD_COLORS["dmd_sindy"],
        lw=2.0,
        label="DMD+SINDy",
    )
    axis.scatter(*truth_states[0], color="#0F172A", s=24, marker="o")
    axis.scatter(*rollout_states[-1], color=METHOD_COLORS["dmd_sindy"], s=28, marker="^")
    axis.set_xlabel(r"$a_1$")
    axis.set_ylabel(r"$a_2$")
    axis.set_zlabel(r"$a_3$")
    axis.set_title(title)
    axis.xaxis.pane.fill = False
    axis.yaxis.pane.fill = False
    axis.zaxis.pane.fill = False
    axis.legend(frameon=False, loc="upper left")


def plot_benchmark_panel(axis, benchmark_df: pd.DataFrame) -> None:
    datasets = benchmark_df["dataset_label"].drop_duplicates().tolist()
    method_order = ["raw_sindy", "pod_sindy", "dmd_sindy"]
    x = np.arange(len(datasets))
    width = 0.22

    for idx, method_name in enumerate(method_order):
        subset = benchmark_df[benchmark_df["method"] == method_name]
        values = [float(subset[subset["dataset_label"] == label]["field_nrmse"].iloc[0]) for label in datasets]
        axis.bar(
            x + (idx - 1) * width,
            values,
            width=width,
            label=METHOD_LABELS[method_name],
            color=METHOD_COLORS[method_name],
        )

    finite_values = benchmark_df["field_nrmse"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_values.max() / max(finite_values.min(), 1.0e-6) > 3.0:
        axis.set_yscale("log")

    axis.set_xticks(x)
    axis.set_xticklabels(datasets, rotation=12, ha="right")
    axis.set_ylabel("Field NRMSE")
    axis.set_title("Benchmark Comparison")
    axis.legend(frameon=False)


def make_poster_figure(dataset_results: dict[str, dict[str, object]], benchmark_df: pd.DataFrame) -> None:
    apply_plot_style()
    fig = plt.figure(figsize=(16.5, 9.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.7, 1.15, 1.0], height_ratios=[1.0, 1.0])
    benchmark_axis = fig.add_subplot(grid[:, 2])
    plot_benchmark_panel(benchmark_axis, benchmark_df)

    bundle_lookup = {bundle.name: bundle for bundle in load_datasets()}
    dataset_order = ["kutz_cylinder", "deepxde_cylinder"]

    for row_idx, dataset_name in enumerate(dataset_order):
        bundle = bundle_lookup[dataset_name]
        result = dataset_results[dataset_name]["dmd_sindy"]

        left = grid[row_idx, 0].subgridspec(1, 3, wspace=0.02)
        mode_axes = [fig.add_subplot(left[0, idx]) for idx in range(3)]
        phase_axis = fig.add_subplot(grid[row_idx, 1], projection="3d")

        image = plot_mode_panel(mode_axes, bundle, result["reduced_model"])
        mode_axes[0].set_ylabel(bundle.title, fontsize=12)

        phase_title = f"{bundle.title}\nDMD+SINDy vs Truth (NRMSE={result['field_nrmse']:.3f})"
        plot_phase_panel(phase_axis, result["truth_states"], result["rollout_states"], phase_title)

        cbar = fig.colorbar(image, ax=mode_axes, shrink=0.78, pad=0.02)
        cbar.outline.set_visible(False)
        cbar.set_label("Mode amplitude")

    fig.savefig(FIGURES_DIR / "vortex_benchmark_poster.png")
    fig.savefig(FIGURES_DIR / "vortex_benchmark_poster.pdf")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()

    spectrum_rows = []
    baseline_rows = []
    optimization_rows = []
    final_rows = []
    final_payload: dict[str, dict[str, object]] = {}

    for bundle in load_datasets():
        train, val, test = split_dataset(bundle)
        spectrum_df, baseline_rank = singular_spectrum_summary(bundle, train)
        spectrum_rows.append(spectrum_df)

        selected_configs: dict[str, MethodConfig] = {}
        for method_name in ["raw_sindy", "pod_sindy", "dmd_sindy"]:
            filter_rank = baseline_rank if method_name != "raw_sindy" else 3
            config, search_df = choose_baseline_threshold(bundle, train, val, method_name, filter_rank)
            selected_configs[method_name] = config
            baseline_rows.append(search_df)

        raw_score = baseline_rows[-3]["field_nrmse"].min()
        pod_score = baseline_rows[-2]["field_nrmse"].min()
        dmd_score = baseline_rows[-1]["field_nrmse"].min()

        optimized_config, optimization_df = optimize_dmd_method(bundle, train, val)
        selected_configs["dmd_sindy"] = optimized_config
        optimization_rows.append(optimization_df)

        final_payload[bundle.name] = {}
        for method_name in ["raw_sindy", "pod_sindy", "dmd_sindy"]:
            result = refit_and_evaluate(bundle, train, val, test, selected_configs[method_name])
            final_payload[bundle.name][method_name] = result
            final_rows.append(
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

    spectrum_df = pd.concat(spectrum_rows, ignore_index=True)
    baseline_df = pd.concat(baseline_rows, ignore_index=True)
    final_df = pd.DataFrame(final_rows).sort_values(["dataset", "method"]).reset_index(drop=True)

    spectrum_df.to_csv(RESULTS_DIR / "singular_spectrum_summary.csv", index=False)
    baseline_df.to_csv(RESULTS_DIR / "baseline_lambda_sweeps.csv", index=False)
    if optimization_rows:
        pd.concat(optimization_rows, ignore_index=True).to_csv(RESULTS_DIR / "dmd_optimization_search.csv", index=False)
    final_df.to_csv(RESULTS_DIR / "benchmark_summary.csv", index=False)
    write_equation_report(final_payload)
    make_poster_figure(final_payload, final_df)


if __name__ == "__main__":
    main()
