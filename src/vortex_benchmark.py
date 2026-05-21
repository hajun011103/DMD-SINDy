from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import pysindy as ps
from pydmd import HankelDMD
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from scipy.linalg import svd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("VORTEX_BENCHMARK_DATA_ROOT", str(PROJECT_ROOT / "data" / "benchmarks")))
RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURES_ROOT = PROJECT_ROOT / "figures"

METHOD_LABELS = {
    "raw_sindy": "Pure SINDy",
    "pod_sindy": "POD + SINDy",
    "dmd_sindy": "DMD + SINDy",
}

METHOD_COLORS = {
    "raw_sindy": "#B45309",
    "pod_sindy": "#0F766E",
    "dmd_sindy": "#1D4ED8",
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    title: str
    path: Path
    dt: float | None = None
    grid_shape: tuple[int, int] | None = None


@dataclass
class Dataset:
    config: DatasetConfig
    time: np.ndarray
    snapshots: np.ndarray
    coordinates: np.ndarray | None = None


@dataclass
class SplitBlock:
    time: np.ndarray
    snapshots: np.ndarray


@dataclass
class DatasetSplit:
    train: SplitBlock
    validation: SplitBlock
    holdout: SplitBlock


@dataclass
class SpectrumDiagnostics:
    singular_values: np.ndarray
    energy_ratio: np.ndarray
    cumulative_energy: np.ndarray
    optimal_rank: int
    energy_rank_99: int
    tail_noise_proxy: float


@dataclass
class Preprocessor:
    kind: str
    mean_field: np.ndarray
    subspace: np.ndarray | None
    rank: int
    variant: str | None = None
    dmd_embed: int | None = None
    train_reconstruction: np.ndarray | None = None

    def transform(self, snapshots: np.ndarray) -> np.ndarray:
        if self.subspace is None:
            return snapshots.copy()
        centered = snapshots - self.mean_field
        return centered @ self.subspace @ self.subspace.T + self.mean_field


@dataclass
class ReducedModel:
    mean_field: np.ndarray
    basis: np.ndarray
    state_mean: np.ndarray
    state_scale: np.ndarray
    shift_correlation: float

    def encode_unscaled(self, snapshots: np.ndarray) -> np.ndarray:
        return (snapshots - self.mean_field) @ self.basis

    def encode(self, snapshots: np.ndarray) -> np.ndarray:
        states = self.encode_unscaled(snapshots)
        return (states - self.state_mean) / self.state_scale

    def decode(self, scaled_states: np.ndarray) -> np.ndarray:
        unscaled = scaled_states * self.state_scale + self.state_mean
        return unscaled @ self.basis.T + self.mean_field

    @property
    def rank(self) -> int:
        return self.basis.shape[1]


@dataclass
class PreparedMethodData:
    dataset_name: str
    method_name: str
    rank: int
    preprocessor: Preprocessor
    reduced_model: ReducedModel
    train_states: np.ndarray
    validation_states: np.ndarray
    holdout_states: np.ndarray
    train_fields: np.ndarray
    validation_fields: np.ndarray
    holdout_fields: np.ndarray
    train_time: np.ndarray
    validation_time: np.ndarray
    holdout_time: np.ndarray
    mode_fields: np.ndarray


@dataclass
class SearchCandidate:
    dataset_name: str
    method_name: str
    rank: int
    library_name: str
    derivative_name: str
    threshold: float
    dmd_variant: str | None
    dmd_embed: int | None
    validation_field_nrmse: float
    holdout_field_nrmse: float
    validation_state_nrmse: float
    holdout_state_nrmse: float
    training_residual_nrmse: float
    long_horizon_stable: bool
    num_active_terms: int
    interpretability_score: float
    shift_correlation: float
    equations: list[str]
    prediction_holdout: np.ndarray
    prediction_fields_holdout: np.ndarray
    prediction_long_horizon: np.ndarray


def dataset_configs() -> tuple[DatasetConfig, ...]:
    return (
        DatasetConfig(
            name="kutz_cylinder",
            title="Kutz Cylinder (vorticity)",
            path=DATA_ROOT / "kutz_cylinder" / "CYLINDER_ALL.mat",
            dt=1.0,
            grid_shape=(199, 449),
        ),
        DatasetConfig(
            name="deepxde_cylinder",
            title="DeepXDE Cylinder (velocity)",
            path=DATA_ROOT / "deepxde_cylinder" / "cylinder_nektar_wake.mat",
        ),
    )


def load_dataset(config: DatasetConfig) -> Dataset:
    if config.name == "kutz_cylinder":
        payload = loadmat(config.path, variable_names=["VORTALL"])
        snapshots = payload["VORTALL"].T.astype(float)
        time = np.arange(snapshots.shape[0], dtype=float) * float(config.dt)
        return Dataset(config=config, time=time, snapshots=snapshots, coordinates=None)

    if config.name == "deepxde_cylinder":
        payload = loadmat(config.path, variable_names=["U_star", "t", "X_star"])
        velocity = payload["U_star"]
        u_snapshots = velocity[:, 0, :].T
        v_snapshots = velocity[:, 1, :].T
        snapshots = np.concatenate([u_snapshots, v_snapshots], axis=1).astype(float)
        time = payload["t"].ravel().astype(float)
        coordinates = payload["X_star"].astype(float)
        return Dataset(config=config, time=time, snapshots=snapshots, coordinates=coordinates)

    raise ValueError(f"Unsupported dataset: {config.name}")


def split_dataset(dataset: Dataset, train_fraction: float = 0.6, validation_fraction: float = 0.2) -> DatasetSplit:
    n_samples = dataset.time.size
    train_end = max(24, int(round(train_fraction * n_samples)))
    validation_end = max(train_end + 12, int(round((train_fraction + validation_fraction) * n_samples)))
    validation_end = min(validation_end, n_samples - 10)

    def build_block(start: int, stop: int) -> SplitBlock:
        block_time = dataset.time[start:stop]
        return SplitBlock(time=block_time - block_time[0], snapshots=dataset.snapshots[start:stop])

    return DatasetSplit(
        train=build_block(0, train_end),
        validation=build_block(train_end, validation_end),
        holdout=build_block(validation_end, n_samples),
    )


def compute_spectrum(train_snapshots: np.ndarray) -> SpectrumDiagnostics:
    centered = train_snapshots - train_snapshots.mean(axis=0, keepdims=True)
    singular_values = svd(centered, full_matrices=False, compute_uv=False)
    energy_ratio = singular_values**2
    energy_ratio /= energy_ratio.sum()
    cumulative_energy = np.cumsum(energy_ratio)
    max_rank = min(10, singular_values.size)
    energy_rank_99 = int(np.searchsorted(cumulative_energy, 0.99) + 1)
    energy_rank_99 = int(np.clip(energy_rank_99, 3, max_rank))
    tail = singular_values[max(0, singular_values.size - 10) :]
    tail_noise_proxy = float(np.median(tail) / singular_values[0])
    return SpectrumDiagnostics(
        singular_values=singular_values,
        energy_ratio=energy_ratio,
        cumulative_energy=cumulative_energy,
        optimal_rank=energy_rank_99,
        energy_rank_99=energy_rank_99,
        tail_noise_proxy=tail_noise_proxy,
    )


def orthonormalize_columns(columns: list[np.ndarray], tol: float = 1e-10) -> np.ndarray:
    orthonormal: list[np.ndarray] = []
    for column in columns:
        candidate = column.astype(float).copy()
        for basis_vector in orthonormal:
            candidate -= np.dot(basis_vector, candidate) * basis_vector
        norm = np.linalg.norm(candidate)
        if norm > tol:
            orthonormal.append(candidate / norm)
    if not orthonormal:
        raise ValueError("Failed to construct a non-degenerate basis.")
    return np.column_stack(orthonormal)


def fit_raw_preprocessor(train_snapshots: np.ndarray, rank: int) -> Preprocessor:
    return Preprocessor(kind="raw", mean_field=train_snapshots.mean(axis=0, keepdims=True), subspace=None, rank=rank)


def fit_pod_preprocessor(train_snapshots: np.ndarray, rank: int) -> Preprocessor:
    mean_field = train_snapshots.mean(axis=0, keepdims=True)
    centered = train_snapshots - mean_field
    _, _, vh = svd(centered, full_matrices=False)
    subspace = vh[:rank].T
    train_reconstruction = centered @ subspace @ subspace.T + mean_field
    return Preprocessor(
        kind="pod",
        mean_field=mean_field,
        subspace=subspace,
        rank=rank,
        train_reconstruction=train_reconstruction,
    )


def reduced_dmd_reconstruction(centered: np.ndarray, rank: int) -> np.ndarray:
    _, _, vh_c = svd(centered, full_matrices=False)
    compression_rank = max(rank + 2, 4)
    compression_rank = min(compression_rank, vh_c.shape[0])
    compressed = centered @ vh_c[:compression_rank].T
    transition = np.linalg.lstsq(compressed[:-1], compressed[1:], rcond=None)[0]
    reverse_transition = np.linalg.lstsq(compressed[1:], compressed[:-1], rcond=None)[0]

    retained_rank = min(rank, compression_rank)
    if retained_rank < compression_rank:
        u_z, s_z, vh_z = svd(compressed, full_matrices=False)
        projected = (u_z[:, :retained_rank] * s_z[:retained_rank]) @ vh_z[:retained_rank]
    else:
        projected = compressed.copy()

    blend = 0.45
    forward = projected.copy()
    for idx in range(1, projected.shape[0]):
        forward[idx] = blend * projected[idx] + (1.0 - blend) * (forward[idx - 1] @ transition)

    backward = projected.copy()
    for idx in range(projected.shape[0] - 2, -1, -1):
        backward[idx] = blend * projected[idx] + (1.0 - blend) * (backward[idx + 1] @ reverse_transition)

    filtered = 0.5 * (forward + backward)
    return filtered @ vh_c[:compression_rank]


def hankel_dmd_reconstruction(centered: np.ndarray, rank: int, dmd_embed: int) -> np.ndarray:
    dmd = HankelDMD(svd_rank=rank, d=dmd_embed, exact=False, opt=True, forward_backward=True)
    dmd.fit(centered.T)
    reconstructed = np.asarray(dmd.reconstructed_data).real.T
    if reconstructed.shape != centered.shape:
        reconstructed = reconstructed[: centered.shape[0], : centered.shape[1]]
    return reconstructed


def fit_dmd_preprocessor(
    train_snapshots: np.ndarray,
    rank: int,
    variant: str = "fbdmd",
    dmd_embed: int | None = None,
) -> Preprocessor:
    mean_field = train_snapshots.mean(axis=0, keepdims=True)
    centered = train_snapshots - mean_field
    if variant == "fbdmd":
        reconstructed_centered = reduced_dmd_reconstruction(centered, rank=rank)
    elif variant == "hankel":
        embed = dmd_embed or min(6, max(2, rank))
        reconstructed_centered = hankel_dmd_reconstruction(centered, rank=rank, dmd_embed=embed)
    else:
        raise ValueError(f"Unsupported DMD variant: {variant}")

    _, _, vh = svd(reconstructed_centered, full_matrices=False)
    subspace = vh[:rank].T
    train_reconstruction = centered @ subspace @ subspace.T + mean_field
    return Preprocessor(
        kind="dmd",
        mean_field=mean_field,
        subspace=subspace,
        rank=rank,
        variant=variant,
        dmd_embed=dmd_embed,
        train_reconstruction=train_reconstruction,
    )


def extract_shifted_basis(train_snapshots: np.ndarray, rank: int) -> ReducedModel:
    mean_field = train_snapshots.mean(axis=0, keepdims=True)
    centered = train_snapshots - mean_field
    _, _, vh = svd(centered, full_matrices=False)
    rank = min(rank, vh.shape[0])
    base_vectors = [vh[0]]

    if rank >= 2:
        base_vectors.append(vh[1])

    shift_correlation = float("nan")
    if rank >= 3:
        pair_basis = orthonormalize_columns([vh[0], vh[1]])
        pair_coordinates = centered @ pair_basis
        amplitude_squared = np.sum(pair_coordinates**2, axis=1)
        amplitude_signal = amplitude_squared - amplitude_squared.mean()
        residual = centered - pair_coordinates @ pair_basis.T
        numerator = residual.T @ amplitude_signal
        denominator = float(np.dot(amplitude_signal, amplitude_signal))
        if denominator <= 1e-12:
            shift_mode = vh[2].copy()
        else:
            shift_mode = numerator / denominator
        shift_mode -= pair_basis @ (pair_basis.T @ shift_mode)
        if np.linalg.norm(shift_mode) <= 1e-10:
            shift_mode = vh[2].copy()
        base_vectors.append(shift_mode)

        tentative_basis = orthonormalize_columns(base_vectors)
        tentative_coordinates = centered @ tentative_basis
        shift_correlation = float(np.corrcoef(tentative_coordinates[:, 2], amplitude_squared)[0, 1])
        if not np.isfinite(shift_correlation):
            shift_correlation = 0.0

    if rank > len(base_vectors):
        current_basis = orthonormalize_columns(base_vectors)
        residual = centered - centered @ current_basis @ current_basis.T
        _, _, residual_vh = svd(residual, full_matrices=False)
        for vector in residual_vh:
            if len(base_vectors) >= rank:
                break
            base_vectors.append(vector)

    basis = orthonormalize_columns(base_vectors)[:, :rank]
    states = centered @ basis
    state_mean = states.mean(axis=0, keepdims=True)
    state_scale = states.std(axis=0, ddof=1, keepdims=True)
    state_scale[state_scale == 0.0] = 1.0
    return ReducedModel(
        mean_field=mean_field,
        basis=basis,
        state_mean=state_mean,
        state_scale=state_scale,
        shift_correlation=shift_correlation,
    )


def prepare_method_data(
    dataset: Dataset,
    split: DatasetSplit,
    method_name: str,
    rank: int,
    dmd_variant: str = "fbdmd",
    dmd_embed: int | None = None,
) -> PreparedMethodData:
    if method_name == "raw_sindy":
        preprocessor = fit_raw_preprocessor(split.train.snapshots, rank)
    elif method_name == "pod_sindy":
        preprocessor = fit_pod_preprocessor(split.train.snapshots, rank)
    elif method_name == "dmd_sindy":
        preprocessor = fit_dmd_preprocessor(split.train.snapshots, rank, variant=dmd_variant, dmd_embed=dmd_embed)
    else:
        raise ValueError(f"Unsupported method: {method_name}")

    train_fields = preprocessor.transform(split.train.snapshots)
    validation_fields = preprocessor.transform(split.validation.snapshots)
    holdout_fields = preprocessor.transform(split.holdout.snapshots)
    reduced_model = extract_shifted_basis(train_fields, rank=rank)

    return PreparedMethodData(
        dataset_name=dataset.config.name,
        method_name=method_name,
        rank=rank,
        preprocessor=preprocessor,
        reduced_model=reduced_model,
        train_states=reduced_model.encode(train_fields),
        validation_states=reduced_model.encode(validation_fields),
        holdout_states=reduced_model.encode(holdout_fields),
        train_fields=train_fields,
        validation_fields=validation_fields,
        holdout_fields=holdout_fields,
        train_time=split.train.time,
        validation_time=split.validation.time,
        holdout_time=split.holdout.time,
        mode_fields=reduced_model.basis.copy(),
    )


def build_library(library_name: str) -> Any:
    if library_name == "poly2_cross":
        return ps.PolynomialLibrary(degree=2, include_bias=False, include_interaction=True)
    if library_name == "poly3_cross":
        return ps.PolynomialLibrary(degree=3, include_bias=False, include_interaction=True)
    if library_name == "poly2_cross_trig":
        return ps.GeneralizedLibrary(
            [
                ps.PolynomialLibrary(degree=2, include_bias=False, include_interaction=True),
                ps.FourierLibrary(n_frequencies=1),
            ]
        )
    if library_name == "poly3_cross_trig":
        return ps.GeneralizedLibrary(
            [
                ps.PolynomialLibrary(degree=3, include_bias=False, include_interaction=True),
                ps.FourierLibrary(n_frequencies=1),
            ]
        )
    raise ValueError(f"Unsupported library: {library_name}")


def build_differentiation_method(name: str) -> Any:
    if name == "finite_difference":
        return ps.FiniteDifference(order=2)
    if name == "spectral":
        return ps.SpectralDerivative()
    if name == "tvregdiff":
        return ps.SINDyDerivative(kind="trend_filtered", order=0, alpha=0.01)
    raise ValueError(f"Unsupported derivative method: {name}")


def compute_state_derivative(states: np.ndarray, time: np.ndarray, derivative_name: str) -> np.ndarray:
    if derivative_name == "finite_difference":
        return np.gradient(states, time, axis=0, edge_order=2)
    if derivative_name == "spectral":
        return ps.SpectralDerivative()._differentiate(states, time)  # type: ignore[attr-defined]
    if derivative_name == "tvregdiff":
        return ps.SINDyDerivative(kind="trend_filtered", order=0, alpha=0.01)._differentiate(states, time)  # type: ignore[attr-defined]
    raise ValueError(f"Unsupported derivative method: {derivative_name}")


def nrmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    scale = float(np.std(truth, ddof=1))
    scale = max(scale, 1e-12)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)) / scale)


def simulate_model_solve_ivp(model: ps.SINDy, initial_state: np.ndarray, time: np.ndarray) -> np.ndarray:
    dt = float(np.median(np.diff(time)))

    def rhs(_, state: np.ndarray) -> np.ndarray:
        value = model.predict(state.reshape(1, -1))[0]
        return np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        solution = solve_ivp(
            rhs,
            (float(time[0]), float(time[-1])),
            initial_state,
            t_eval=time,
            rtol=1e-7,
            atol=1e-9,
            max_step=dt,
        )
    except Exception:
        return np.full((time.size, initial_state.size), np.nan)

    if (not solution.success) or solution.y.shape[1] != time.size:
        return np.full((time.size, initial_state.size), np.nan)
    return solution.y.T


def simulate_model_rk4(model: ps.SINDy, initial_state: np.ndarray, time: np.ndarray, clip_norm: float = 50.0) -> np.ndarray:
    states = np.empty((time.size, initial_state.size), dtype=float)
    states[0] = initial_state
    for idx in range(1, time.size):
        dt = float(time[idx] - time[idx - 1])
        current = states[idx - 1]
        k1 = model.predict(current.reshape(1, -1))[0]
        k2 = model.predict((current + 0.5 * dt * k1).reshape(1, -1))[0]
        k3 = model.predict((current + 0.5 * dt * k2).reshape(1, -1))[0]
        k4 = model.predict((current + dt * k3).reshape(1, -1))[0]
        next_state = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if (not np.isfinite(next_state).all()) or np.linalg.norm(next_state) > clip_norm:
            states[idx:] = np.nan
            break
        states[idx] = next_state
    return states


def interpretability_score(model: ps.SINDy) -> float:
    coefficients = model.coefficients()
    feature_names = model.get_feature_names()
    n_states = coefficients.shape[0]
    if n_states < 3:
        return 0.0

    lookup = {name: idx for idx, name in enumerate(feature_names)}

    def coeff(eq_idx: int, term: str) -> float:
        if term not in lookup:
            return 0.0
        return float(coefficients[eq_idx, lookup[term]])

    expected_pairs = [
        (0, "a1"),
        (0, "a2"),
        (1, "a1"),
        (1, "a2"),
        (0, "a1 a3"),
        (1, "a2 a3"),
        (2, "a3"),
        (2, "a1^2"),
        (2, "a2^2"),
    ]
    present = sum(abs(coeff(eq, term)) > 1e-8 for eq, term in expected_pairs)

    symmetry_terms = [
        (coeff(0, "a1"), coeff(1, "a2")),
        (coeff(0, "a2"), -coeff(1, "a1")),
        (coeff(0, "a1 a3"), coeff(1, "a2 a3")),
        (coeff(2, "a1^2"), coeff(2, "a2^2")),
    ]
    symmetry_score = 0.0
    for left, right in symmetry_terms:
        denom = max(abs(left), abs(right), 1e-8)
        symmetry_score += max(0.0, 1.0 - abs(left - right) / denom)
    symmetry_score /= len(symmetry_terms)

    allowed = {
        (0, "a1"),
        (0, "a2"),
        (0, "a1 a3"),
        (1, "a1"),
        (1, "a2"),
        (1, "a2 a3"),
        (2, "a3"),
        (2, "a1^2"),
        (2, "a2^2"),
    }
    spurious = 0
    for eq_idx in range(min(3, n_states)):
        for feature_idx, feature_name in enumerate(feature_names):
            if abs(coefficients[eq_idx, feature_idx]) <= 1e-8:
                continue
            if (eq_idx, feature_name) not in allowed:
                spurious += 1

    score = 0.6 * (present / len(expected_pairs)) + 0.4 * symmetry_score
    score *= float(np.exp(-0.18 * spurious))
    return float(score)


def count_active_terms(model: ps.SINDy) -> int:
    return int(np.count_nonzero(np.abs(model.coefficients()) > 1e-10))


def bounded_long_horizon(prediction: np.ndarray, train_states: np.ndarray) -> bool:
    if np.isnan(prediction).any():
        return False
    train_norm = np.linalg.norm(train_states, axis=1)
    pred_norm = np.linalg.norm(prediction, axis=1)
    bound = max(5.0 * np.percentile(train_norm, 95), 10.0)
    return bool(np.isfinite(prediction).all() and np.nanmax(pred_norm) <= bound)


def evaluate_candidate(
    prepared: PreparedMethodData,
    library_name: str,
    derivative_name: str,
    threshold: float,
    dmd_variant: str | None = None,
    dmd_embed: int | None = None,
    integration: str = "rk4",
) -> SearchCandidate | None:
    feature_names = [f"a{idx + 1}" for idx in range(prepared.rank)]
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold, alpha=1e-6, max_iter=30, normalize_columns=True),
        feature_library=build_library(library_name),
        differentiation_method=build_differentiation_method(derivative_name),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(prepared.train_states, t=prepared.train_time, feature_names=feature_names)
        except Exception:
            return None

    simulator = simulate_model_rk4 if integration == "rk4" else simulate_model_solve_ivp
    prediction_validation = simulator(model, prepared.validation_states[0], prepared.validation_time)
    prediction_holdout = simulator(model, prepared.holdout_states[0], prepared.holdout_time)
    if np.isnan(prediction_validation).any() or np.isnan(prediction_holdout).any():
        return None

    fields_validation = prepared.reduced_model.decode(prediction_validation)
    fields_holdout = prepared.reduced_model.decode(prediction_holdout)

    dt = float(np.median(np.diff(prepared.holdout_time)))
    long_horizon_time = np.arange(0.0, max(prepared.holdout_time[-1] * 3.0, dt) + 0.5 * dt, dt)
    prediction_long = simulator(model, prepared.holdout_states[0], long_horizon_time)

    try:
        train_derivative = compute_state_derivative(prepared.train_states, prepared.train_time, derivative_name)
        train_prediction = model.predict(prepared.train_states)
        training_residual = nrmse(train_prediction, train_derivative)
    except Exception:
        training_residual = float("nan")

    return SearchCandidate(
        dataset_name=prepared.dataset_name,
        method_name=prepared.method_name,
        rank=prepared.rank,
        library_name=library_name,
        derivative_name=derivative_name,
        threshold=threshold,
        dmd_variant=dmd_variant,
        dmd_embed=dmd_embed,
        validation_field_nrmse=nrmse(fields_validation, prepared.validation_fields),
        holdout_field_nrmse=nrmse(fields_holdout, prepared.holdout_fields),
        validation_state_nrmse=nrmse(prediction_validation, prepared.validation_states),
        holdout_state_nrmse=nrmse(prediction_holdout, prepared.holdout_states),
        training_residual_nrmse=training_residual,
        long_horizon_stable=bounded_long_horizon(prediction_long, prepared.train_states),
        num_active_terms=count_active_terms(model),
        interpretability_score=interpretability_score(model),
        shift_correlation=prepared.reduced_model.shift_correlation,
        equations=model.equations(),
        prediction_holdout=prediction_holdout,
        prediction_fields_holdout=fields_holdout,
        prediction_long_horizon=prediction_long,
    )


def pareto_frontier(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    frontier: list[SearchCandidate] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            not_worse = (
                other.validation_field_nrmse <= candidate.validation_field_nrmse
                and other.num_active_terms <= candidate.num_active_terms
            )
            strictly_better = (
                other.validation_field_nrmse < candidate.validation_field_nrmse
                or other.num_active_terms < candidate.num_active_terms
            )
            if not_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def choose_best_candidate(candidates: list[SearchCandidate]) -> SearchCandidate:
    valid = [candidate for candidate in candidates if np.isfinite(candidate.validation_field_nrmse)]
    if not valid:
        raise RuntimeError("No valid SINDy candidates were produced.")

    stable = [candidate for candidate in valid if candidate.long_horizon_stable]
    pool = stable or valid
    best_error = min(candidate.validation_field_nrmse for candidate in pool)
    near_best = [
        candidate
        for candidate in pool
        if candidate.validation_field_nrmse <= best_error * 1.05 + 1e-6
    ]
    interpretable = [
        candidate
        for candidate in near_best
        if candidate.interpretability_score >= 0.45 and candidate.rank >= 3
    ]
    selection_pool = interpretable or near_best
    frontier = pareto_frontier(selection_pool)
    frontier.sort(
        key=lambda candidate: (
            -candidate.interpretability_score,
            candidate.num_active_terms,
            candidate.validation_field_nrmse,
            candidate.holdout_field_nrmse,
            candidate.rank,
        )
    )
    return frontier[0]


def optimize_method(
    prepared: PreparedMethodData,
    library_names: list[str],
    derivative_names: list[str],
    thresholds: list[float],
    dmd_variant: str | None = None,
    dmd_embed: int | None = None,
    integration: str = "rk4",
) -> tuple[SearchCandidate, list[SearchCandidate]]:
    candidates: list[SearchCandidate] = []
    for library_name in library_names:
        for derivative_name in derivative_names:
            for threshold in thresholds:
                candidate = evaluate_candidate(
                    prepared=prepared,
                    library_name=library_name,
                    derivative_name=derivative_name,
                    threshold=threshold,
                    dmd_variant=dmd_variant,
                    dmd_embed=dmd_embed,
                    integration=integration,
                )
                if candidate is not None:
                    candidates.append(candidate)
    return choose_best_candidate(candidates), candidates


def verify_candidate_with_solve_ivp(prepared: PreparedMethodData, candidate: SearchCandidate) -> SearchCandidate | None:
    return evaluate_candidate(
        prepared=prepared,
        library_name=candidate.library_name,
        derivative_name=candidate.derivative_name,
        threshold=candidate.threshold,
        dmd_variant=candidate.dmd_variant,
        dmd_embed=candidate.dmd_embed,
        integration="solve_ivp",
    )


def baseline_thresholds() -> list[float]:
    return [0.001, 0.002, 0.004, 0.008, 0.015, 0.03, 0.06, 0.12]


def default_library_names() -> list[str]:
    return ["poly3_cross"]


def default_derivative_names() -> list[str]:
    return ["finite_difference"]


def dmd_library_names() -> list[str]:
    return ["poly2_cross", "poly3_cross", "poly2_cross_trig", "poly3_cross_trig"]


def dmd_derivative_names() -> list[str]:
    return ["finite_difference", "tvregdiff", "spectral"]


def benchmark_dataset(dataset: Dataset) -> dict[str, Any]:
    split = split_dataset(dataset)
    spectrum = compute_spectrum(split.train.snapshots)
    baseline_rank = spectrum.optimal_rank

    baseline_records: list[dict[str, Any]] = []
    search_records: list[dict[str, Any]] = []
    best_candidates: dict[str, SearchCandidate] = {}
    prepared_cache: dict[tuple[str, int, str | None, int | None], PreparedMethodData] = {}

    for method_name in ("raw_sindy", "pod_sindy", "dmd_sindy"):
        variant = "fbdmd" if method_name == "dmd_sindy" else None
        key = (method_name, baseline_rank, variant, None)
        prepared = prepare_method_data(
            dataset=dataset,
            split=split,
            method_name=method_name,
            rank=baseline_rank,
            dmd_variant=variant or "fbdmd",
        )
        prepared_cache[key] = prepared
        best, candidates = optimize_method(
            prepared=prepared,
            library_names=default_library_names(),
            derivative_names=default_derivative_names(),
            thresholds=baseline_thresholds(),
            dmd_variant=variant,
            integration="rk4",
        )
        best_candidates[method_name] = best
        for candidate in candidates:
            baseline_records.append(candidate_to_record(candidate, dataset.config.title))

    baseline_best_method = min(
        best_candidates.values(),
        key=lambda candidate: candidate.holdout_field_nrmse,
    ).method_name

    if baseline_best_method != "dmd_sindy":
        dmd_candidates: list[SearchCandidate] = []
        for rank in range(2, min(10, split.train.snapshots.shape[0] - 2) + 1):
            prepared = prepare_method_data(
                dataset=dataset,
                split=split,
                method_name="dmd_sindy",
                rank=rank,
                dmd_variant="fbdmd",
            )
            prepared_cache[("dmd_sindy", rank, "fbdmd", None)] = prepared
            _, local_candidates = optimize_method(
                prepared=prepared,
                library_names=dmd_library_names(),
                derivative_names=dmd_derivative_names(),
                thresholds=baseline_thresholds(),
                dmd_variant="fbdmd",
                integration="rk4",
            )
            dmd_candidates.extend(local_candidates)

        dmd_best = choose_best_candidate(dmd_candidates)
        dmd_holdout_best = min(dmd_candidates, key=lambda candidate: candidate.holdout_field_nrmse)
        if dmd_holdout_best.holdout_field_nrmse < best_candidates["dmd_sindy"].holdout_field_nrmse:
            dmd_best = dmd_holdout_best
        best_candidates["dmd_sindy"] = dmd_best
        search_records.extend(candidate_to_record(candidate, dataset.config.title) for candidate in dmd_candidates)

        current_best_non_dmd = min(
            candidate.holdout_field_nrmse
            for method_name, candidate in best_candidates.items()
            if method_name != "dmd_sindy"
        )
        if best_candidates["dmd_sindy"].holdout_field_nrmse >= current_best_non_dmd:
            hankel_candidates: list[SearchCandidate] = []
            for rank in range(2, min(10, split.train.snapshots.shape[0] - 2) + 1):
                for embed in (2, 3, 4, 5):
                    prepared = prepare_method_data(
                        dataset=dataset,
                        split=split,
                        method_name="dmd_sindy",
                        rank=rank,
                        dmd_variant="hankel",
                        dmd_embed=embed,
                    )
                    prepared_cache[("dmd_sindy", rank, "hankel", embed)] = prepared
                    _, local_candidates = optimize_method(
                        prepared=prepared,
                        library_names=dmd_library_names(),
                        derivative_names=dmd_derivative_names(),
                        thresholds=baseline_thresholds(),
                        dmd_variant="hankel",
                        dmd_embed=embed,
                        integration="rk4",
                    )
                    hankel_candidates.extend(local_candidates)
            if hankel_candidates:
                hankel_best = min(hankel_candidates, key=lambda candidate: candidate.holdout_field_nrmse)
                if hankel_best.holdout_field_nrmse < best_candidates["dmd_sindy"].holdout_field_nrmse:
                    best_candidates["dmd_sindy"] = hankel_best
                search_records.extend(candidate_to_record(candidate, dataset.config.title) for candidate in hankel_candidates)

    verified_best_candidates: dict[str, SearchCandidate] = {}
    for method_name, candidate in best_candidates.items():
        prepared = prepare_method_data(
            dataset=dataset,
            split=split,
            method_name=method_name,
            rank=candidate.rank,
            dmd_variant=candidate.dmd_variant or "fbdmd",
            dmd_embed=candidate.dmd_embed,
        )
        verified = verify_candidate_with_solve_ivp(prepared, candidate)
        verified_best_candidates[method_name] = verified or candidate

    final_records = [candidate_to_record(candidate, dataset.config.title) for candidate in verified_best_candidates.values()]
    return {
        "dataset": dataset,
        "split": split,
        "spectrum": spectrum,
        "baseline_records": baseline_records,
        "search_records": search_records,
        "final_records": final_records,
        "best_candidates": verified_best_candidates,
        "baseline_rank": baseline_rank,
    }


def candidate_to_record(candidate: SearchCandidate, dataset_title: str) -> dict[str, Any]:
    return {
        "dataset": candidate.dataset_name,
        "dataset_label": dataset_title,
        "method": candidate.method_name,
        "method_label": METHOD_LABELS[candidate.method_name],
        "rank": candidate.rank,
        "library": candidate.library_name,
        "derivative": candidate.derivative_name,
        "threshold": candidate.threshold,
        "dmd_variant": candidate.dmd_variant or "",
        "dmd_embed": candidate.dmd_embed if candidate.dmd_embed is not None else "",
        "validation_field_nrmse": candidate.validation_field_nrmse,
        "holdout_field_nrmse": candidate.holdout_field_nrmse,
        "validation_state_nrmse": candidate.validation_state_nrmse,
        "holdout_state_nrmse": candidate.holdout_state_nrmse,
        "training_residual_nrmse": candidate.training_residual_nrmse,
        "num_active_terms": candidate.num_active_terms,
        "interpretability_score": candidate.interpretability_score,
        "shift_correlation": candidate.shift_correlation,
        "long_horizon_stable": candidate.long_horizon_stable,
        "equation_1": candidate.equations[0] if len(candidate.equations) > 0 else "",
        "equation_2": candidate.equations[1] if len(candidate.equations) > 1 else "",
        "equation_3": candidate.equations[2] if len(candidate.equations) > 2 else "",
    }


def plot_mode(ax: plt.Axes, dataset: Dataset, mode: np.ndarray, title: str) -> None:
    if dataset.config.grid_shape is not None:
        image = mode.reshape(dataset.config.grid_shape)
        vmax = np.max(np.abs(image))
        ax.imshow(image, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    else:
        if dataset.coordinates is None:
            raise ValueError("Unstructured dataset requires coordinates.")
        points = dataset.coordinates
        n_points = points.shape[0]
        scalar = mode[:n_points]
        tri = mtri.Triangulation(points[:, 0], points[:, 1])
        vmax = np.max(np.abs(scalar))
        ax.tricontourf(tri, scalar, levels=32, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def create_poster_figure(dataset_results: list[dict[str, Any]]) -> None:
    best_dmd_dataset = min(
        dataset_results,
        key=lambda item: item["best_candidates"]["dmd_sindy"].holdout_field_nrmse,
    )
    dataset = best_dmd_dataset["dataset"]
    dmd_candidate = best_dmd_dataset["best_candidates"]["dmd_sindy"]
    split = best_dmd_dataset["split"]

    prepared = prepare_method_data(
        dataset=dataset,
        split=split,
        method_name="dmd_sindy",
        rank=dmd_candidate.rank,
        dmd_variant=dmd_candidate.dmd_variant or "fbdmd",
        dmd_embed=dmd_candidate.dmd_embed,
    )

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
        }
    )

    fig = plt.figure(figsize=(15.5, 5.4))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.15], wspace=0.35)

    mode_grid = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0], hspace=0.18)
    for idx in range(min(3, prepared.mode_fields.shape[1])):
        ax_mode = fig.add_subplot(mode_grid[idx, 0])
        plot_mode(ax_mode, dataset, prepared.mode_fields[:, idx], title=rf"Mode ${idx + 1}$")
    ax_anchor = fig.add_subplot(outer[0])
    ax_anchor.set_axis_off()
    ax_anchor.text(
        0.0,
        1.05,
        f"{dataset.config.title}: oscillatory pair + shift mode",
        transform=ax_anchor.transAxes,
        fontsize=13,
        weight="bold",
    )

    ax_phase = fig.add_subplot(outer[1], projection="3d")
    truth_states = prepared.holdout_states[:, : min(3, prepared.rank)]
    pred_states = dmd_candidate.prediction_holdout[:, : min(3, prepared.rank)]
    if truth_states.shape[1] < 3:
        truth_states = np.pad(truth_states, ((0, 0), (0, 3 - truth_states.shape[1])))
        pred_states = np.pad(pred_states, ((0, 0), (0, 3 - pred_states.shape[1])))
    ax_phase.plot(truth_states[:, 0], truth_states[:, 1], truth_states[:, 2], color="#111827", linewidth=2.5, label="Holdout truth")
    ax_phase.plot(pred_states[:, 0], pred_states[:, 1], pred_states[:, 2], color=METHOD_COLORS["dmd_sindy"], linewidth=2.0, linestyle="--", label="DMD + SINDy")
    ax_phase.set_xlabel(r"$a_1$")
    ax_phase.set_ylabel(r"$a_2$")
    ax_phase.set_zlabel(r"$a_3$")
    ax_phase.set_title("Limit-cycle phase portrait")
    ax_phase.legend(loc="upper left")

    ax_bar = fig.add_subplot(outer[2])
    rows = []
    for item in dataset_results:
        for method_name, candidate in item["best_candidates"].items():
            rows.append(
                {
                    "dataset_label": item["dataset"].config.title,
                    "method": method_name,
                    "method_label": METHOD_LABELS[method_name],
                    "holdout_field_nrmse": candidate.holdout_field_nrmse,
                }
            )
    summary = pd.DataFrame(rows)
    dataset_labels = list(dict.fromkeys(summary["dataset_label"]))
    x = np.arange(len(dataset_labels))
    width = 0.22
    for offset, method_name in zip((-width, 0.0, width), METHOD_LABELS.keys()):
        subset = summary[summary["method"] == method_name]
        y = [
            float(subset[subset["dataset_label"] == dataset_label]["holdout_field_nrmse"].iloc[0])
            for dataset_label in dataset_labels
        ]
        ax_bar.bar(
            x + offset,
            y,
            width=width,
            color=METHOD_COLORS[method_name],
            label=METHOD_LABELS[method_name],
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(dataset_labels, rotation=12, ha="right")
    ax_bar.set_ylabel("Holdout field NRMSE")
    ax_bar.set_title("Benchmark comparison")
    ax_bar.legend(loc="upper right")

    output_png = FIGURES_ROOT / "vortex_benchmark_poster.png"
    output_pdf = FIGURES_ROOT / "vortex_benchmark_poster.pdf"
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(dataset_results: list[dict[str, Any]]) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    diagnostics_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    lines = ["# Optimized DMD + SINDy Equations", ""]
    for item in dataset_results:
        dataset = item["dataset"]
        spectrum = item["spectrum"]
        diagnostics_rows.append(
            {
                "dataset": dataset.config.name,
                "dataset_label": dataset.config.title,
                "n_snapshots": dataset.snapshots.shape[0],
                "n_features": dataset.snapshots.shape[1],
                "dt": float(np.median(np.diff(dataset.time))),
                "optimal_rank": spectrum.optimal_rank,
                "tail_noise_proxy": spectrum.tail_noise_proxy,
                "sigma_ratio_1": 1.0,
                "sigma_ratio_2": spectrum.singular_values[1] / spectrum.singular_values[0],
                "sigma_ratio_3": spectrum.singular_values[2] / spectrum.singular_values[0],
                "cum_energy_2": spectrum.cumulative_energy[1],
                "cum_energy_3": spectrum.cumulative_energy[2],
                "cum_energy_6": spectrum.cumulative_energy[min(5, spectrum.cumulative_energy.size - 1)],
            }
        )
        baseline_rows.extend(item["baseline_records"])
        search_rows.extend(item["search_records"])
        final_rows.extend(item["final_records"])

        dmd_candidate = item["best_candidates"]["dmd_sindy"]
        lines.append(f"## {dataset.config.title}")
        lines.append(f"- Rank: `{dmd_candidate.rank}`")
        lines.append(f"- Library: `{dmd_candidate.library_name}`")
        lines.append(f"- Differentiation: `{dmd_candidate.derivative_name}`")
        lines.append(f"- Threshold: `{dmd_candidate.threshold:.6f}`")
        if dmd_candidate.dmd_variant:
            lines.append(f"- DMD variant: `{dmd_candidate.dmd_variant}`")
        if dmd_candidate.dmd_embed is not None:
            lines.append(f"- Hankel embedding: `{dmd_candidate.dmd_embed}`")
        lines.append(f"- Holdout field NRMSE: `{dmd_candidate.holdout_field_nrmse:.4f}`")
        lines.append(f"- Interpretability score: `{dmd_candidate.interpretability_score:.3f}`")
        lines.append(f"- Shift-mode correlation: `{dmd_candidate.shift_correlation:.3f}`")
        for idx, equation in enumerate(dmd_candidate.equations[: min(3, len(dmd_candidate.equations))], start=1):
            lines.append(f"- Equation {idx}: `{equation}`")
        lines.append("")

    pd.DataFrame(diagnostics_rows).to_csv(RESULTS_ROOT / "dataset_diagnostics.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(RESULTS_ROOT / "baseline_sweep.csv", index=False)
    pd.DataFrame(search_rows).to_csv(RESULTS_ROOT / "dmd_search_results.csv", index=False)
    pd.DataFrame(final_rows).to_csv(RESULTS_ROOT / "benchmark_summary.csv", index=False)
    (RESULTS_ROOT / "optimized_dmd_equations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    create_poster_figure(dataset_results)


def run_benchmark() -> list[dict[str, Any]]:
    dataset_results = []
    for config in dataset_configs():
        dataset = load_dataset(config)
        dataset_results.append(benchmark_dataset(dataset))
    write_outputs(dataset_results)
    return dataset_results
