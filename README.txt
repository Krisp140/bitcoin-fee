Bitcoin Transaction Fee Prediction (Two-Stage Structural Model)
==============================================================

Overview
--------
This repository contains the code and notebooks for a structural model of Bitcoin transaction fees. The project uses a two-stage pipeline:
1) Phase 2 (feature engineering): predict wait times and congestion metrics from raw mempool data, then export derived features.
2) Phase 3 (fee modeling): fit structural fee models using the engineered features plus transaction characteristics and exchange flags.

Key files and artifacts
-----------------------
- `phase2_export.py` – end-to-end Phase 2 export script that loads the SQLite dataset, builds epochs, trains a Random Forest wait-time model, computes congestion (rho_t), impatience proxies, quantiles/tail probabilities, and writes `phase2_derived_features.pkl`.
- `phase3_fee_model_new_data.ipynb` – Phase 3 notebook with structural fee models (OLS, quantile regression, Gamma GLM, splines) using the Phase 2 features.
- `simple-correlation.ipynb` – exploratory correlation analysis between fee_rate and mempool congestion.
- `generate_pipeline_diagram.py` – draws the two-stage flowchart saved to `plots/pipeline_diagram.{png,pdf}`.
- `scripts/fee_mempool_relationship.py` – CLI analysis of fee_rate vs mempool congestion with plots and CSV summaries.
- `scripts/lasso_congestion.py` – Lasso-based feature selection to assess colinearity of congestion with transaction-level predictors.
- `scripts/plot_block_timeseries.py` – block-level mempool time-series plotter.
- `plots/` – generated figures (pipeline diagram, mempool/fee plots, NFT plot, spline diagnostics, etc.).
- `phase2_derived_features.pkl` – Phase 2 export artifact (created by `phase2_export.py`).
- `SLIDESHOW_ARCHITECTURE.md` – slide-outline describing research questions, dataset, and model architecture.
- `exchange_addresses.py` – helper data for exchange address tagging used in notebooks.
- `requirements.txt` – Python dependencies.

Data prerequisites
------------------
- SQLite database with mempool transactions (default path in scripts: `/home/armin/datalake/data-samples/11-24-2025-15m-data-lake.db`).
- The dataset should include columns such as `tx_id`, `found_at`, `mined_at`, `waittime`, `min_respend_blocks`, `child_txid`, `rbf_fee_total`, `mempool_size`, `mempool_tx_count`, `fee_rate`, and block identifiers (`conf_block_hash`).
- Update `DB_PATH` in scripts or pass CLI flags so paths match your local environment.

Environment setup
-----------------
1) Create a virtual environment (recommended):
   - `python -m venv .venv`
   - `source .venv/bin/activate`
2) Install dependencies:
   - `pip install -r requirements.txt`

Running Phase 2 export
----------------------
- Edit `DB_PATH` and `OUTPUT_PATH` in `phase2_export.py` to point to your SQLite file and desired pickle output.
- Run:
  - `python phase2_export.py`
- Output: `phase2_derived_features.pkl` containing `tx_id`, `W_hat` (predicted wait time), `rho_t` (congestion), `time_cost`, `time_cost_quantile`, `F_tq`, and `epoch`.

Phase 3 modeling (notebook)
---------------------------
- Open `phase3_fee_model_new_data.ipynb` in Jupyter/VS Code.
- Ensure the notebook can load both the SQLite DB and `phase2_derived_features.pkl`.
- The notebook fits structural fee models, reports coefficients, and produces diagnostics/plots saved under `plots/`.

Utility scripts
---------------
- Fee vs mempool relationship (plots + CSV):
  - `python scripts/fee_mempool_relationship.py --db-path <db> --blocks 2000 --row-limit 1000000 --output-dir plots/fee_mempool`
- Lasso congestion feature check:
  - `python scripts/lasso_congestion.py --db-path <db> --phase2-path phase2_derived_features.pkl --blocks 2000 --max-rows 500000`
- Block-level mempool time series:
  - `python scripts/plot_block_timeseries.py --db-path <db> --output plots/block_timeseries.png --limit-blocks 0`
- Pipeline diagram generation:
  - `python generate_pipeline_diagram.py`

Recommended workflow
--------------------
1) Install dependencies.
2) Generate Phase 2 features with `phase2_export.py`.
3) Run exploratory scripts for sanity checks/plots (optional).
4) Open `phase3_fee_model_new_data.ipynb` to train/evaluate the structural fee models and review results.

Notes
-----
- Scripts assume the DB exists and may load millions of rows; ensure adequate RAM and consider the `--row-limit`/`--max-rows` options when running on limited machines.
- Plots and exports default to the `plots/` directory; ensure it exists or let the scripts create it.

