# big-data

## Getting the Benchmark

* Download the data harmonization benchmark from git@github.com:VIDA-NYU/data-harmonization-benchmark.git.

## Layout

| Path | Contents |
|---|---|
| `datasets/` | Benchmark data. It is not in git; see `datasets/README.md`. `datasets/MANIFEST.json` fingerprints the files every score depends on |
| `env/` | The shared, locked Python environment (`bash env/setup_env.sh`) |
| `hbench/` | Shared scoring, prediction format and run logging used by every experiment (see `hbench/README.md`); tests are in `tests/` (`python -m pytest`) |
| `experiments/` | One folder per experiment. Raw run logs under `results/raw/` stay out of git; commit only small summaries |