# KSME 2026 Vortex Benchmark

This repository contains the scripts, selected figures, result tables, and KSME submission files for a DMD + SINDy workflow for noise-robust governing-equation discovery in vortex-shedding data. The public benchmark scope is limited to the Kutz cylinder wake dataset and the DeepXDE/Nektar cylinder wake dataset.

## Repository Layout

- `docs/`: final KSME abstract and poster files.
- `src/`: reusable benchmark module.
- `scripts/`: executable analysis and figure-generation scripts.
- `results/`: generated CSV/Markdown reports for the Kutz and DeepXDE benchmarks.
- `assets/`: selected README images for upload.
- `figures/`: local generated PNG/PDF figure library, not intended for upload.
- `data/`: raw-data landing area. Raw datasets are local-only.

## Final Submission Files

- `docs/HajunJang_KSME_2026_abstract.pdf`
- `docs/HajunJang_KSME_2026_abstract.docx`
- `docs/HajunJang_KSME2026Poster.pdf`

## Selected Figures

### Latent Modes

<img src="./assets/vortex_latent_modes.png" alt="DMD + SINDy latent modes for Kutz and DeepXDE cylinder wakes" width="900">

### Phase Portraits

<img src="./assets/vortex_phase_compare_deepxde.png" alt="DeepXDE phase portrait comparison" width="620">

<img src="./assets/vortex_phase_compare_kutz.png" alt="Kutz phase portrait comparison" width="620">

## Data Policy

Raw benchmark data files are local-only by default. The upload scope includes documentation for the Kutz and DeepXDE datasets, selected figures, and generated summary results, but not the full raw data archives. The Zenodo single-cylinder and PIV Challenge Case B files are local-only exploratory data and are excluded from the public upload scope.

Expected raw data paths for the public benchmark datasets:

- `data/benchmarks/kutz_cylinder/CYLINDER_ALL.mat`
- `data/benchmarks/kutz_cylinder/CYLINDER_basis.mat`
- `data/benchmarks/deepxde_cylinder/cylinder_nektar_wake.mat`

Download sources:

- Kutz cylinder wake: download `DATA.zip` from `http://databookuw.com/DATA.zip`, then copy `DATA/FLUIDS/CYLINDER_ALL.mat` and `DATA/FLUIDS/CYLINDER_basis.mat` into `data/benchmarks/kutz_cylinder/`.
- DeepXDE/Nektar cylinder wake: download `cylinder_nektar_wake.mat` from `https://github.com/maziarraissi/PINNs/blob/master/main/Data/cylinder_nektar_wake.mat` or the raw URL `https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/cylinder_nektar_wake.mat`, then save it into `data/benchmarks/deepxde_cylinder/`.

For the Kutz/DeepXDE benchmark scripts, you can also point to an external data directory:

```bash
export VORTEX_BENCHMARK_DATA_ROOT=/path/to/benchmarks
```

## Environment

```bash
python -m pip install -r requirements.txt
```

## Reproducing Outputs

Run the scripts from the repository root:

```bash
python scripts/run_vortex_benchmark.py
python scripts/benchmark_vortex_shedding.py
python scripts/finalize_vortex_results.py
python scripts/generate_poster_overview_assets.py
```

Some scripts are compute-heavy and require the raw datasets listed above.
