# SpikeDistill-RUL

Distilling an attention-BiLSTM teacher into a recurrent leaky-integrate-and-fire (spiking) student for remaining-useful-life (RUL) prediction on NASA C-MAPSS (FD001–FD004). Machine Learning course project, IIT Patna.

What is here: the data pipeline and leakage tests (`src/rul/data.py`, `tests/`), 18 dense architectures plus the spiking student (`src/rul/models.py`), the distillation loss (`src/rul/distill.py`), the energy model, the resumable experiment runner (`scripts/run_suite.py`, also runnable on Modal with `scripts/modal_suite.py`), the statistics and diagnostics scripts, and the end-to-end notebook `2401CS39_MLProject.ipynb`. Every stored run is a JSON file in `results/suite/`, and all tables in the report are computed from those files by `scripts/analysis_v2.py`.

## Data

Download the C-MAPSS archive (`CMAPSSData.zip`) from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and put `train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt` (x = 1–4) in `data/raw/`. The data are not redistributed here.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q tests                                   # leakage, scalers, test-like validation, models, energy
python scripts/run_suite.py <phase>               # sweep | v2base | v2distill | pooled | v2abl (resumable)
python scripts/choose_window_v2.py                # window length from the validation sweep
python scripts/analysis_v2.py                     # tables and paired tests -> results/analysis_v2.json
python scripts/diagnose.py v2 60                  # why strong models tie
python scripts/naive_baselines.py                 # hand-feature baselines
python scripts/ann2snn.py                         # ANN-to-SNN conversion baseline
python scripts/xai.py                             # explainability
python scripts/make_notebook.py                   # build and execute the notebook (quick mode)
```

## Honest limits

Energy is counted from operations (4.6 pJ per multiply-accumulate, 0.9 pJ per accumulate, 45 nm figures from Horowitz, ISSCC 2014), not measured on hardware. The spiking student does not beat the dense models and, by this count, is not cheaper than a small GRU. See the notebook and `results/analysis_v2.json` for the numbers.
