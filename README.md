# REFIT washing-machine disaggregation

A reproducible, CPU-only take-home study: audit and clean raw House 1, analyze every explicitly labelled washing-machine channel in official cleaned REFIT, and evaluate aggregate-only appliance-power inference with chronological and unseen-house tests.

The emphasis is traceable data decisions and honest validation. This is a research prototype with measurable failure modes, not a production appliance-metering service.

## Run end to end

Python 3.12 was used. Approximately 1.2 GB of downloads and 10 GB of local space are needed. Allow several minutes to download/extract and several more to run on a modern CPU; network and hardware vary. Use the pinned dependency versions for the most reproducible numerical output. No GPU, API token or AI service is required at runtime.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python download_data.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python pipeline.py --stage all
python write_reports.py
```

On Windows use the equivalent environment variable syntax or omit the thread limits. `make data`, `make run`, `make test` wrap the same commands. Downloading reads only public university files. If their endpoint changes, manually place the official archives/readmes in `data/downloads/`; do not substitute another dataset unnoticed. Checksums reject changed source bytes.

Optional staged reruns:

```bash
python pipeline.py --stage raw
python pipeline.py --stage eda
OMP_NUM_THREADS=2 python pipeline.py --stage model
python write_reports.py
```

`model` requires minute caches from `eda`. `write_reports.py` requires all report outputs. Reruns overwrite this project's own generated outputs. No email actions, unrelated repositories, credentials, paid services or background automation are needed.

## What is delivered

| File | Purpose |
| --- | --- |
| `pipeline.py` | Cleaning, complete washing-machine EDA, sequence features, training, validation, predictions, metrics and figures |
| `download_data.py` | Official download, atomic temporary-file handling, checksum verification and archive extraction |
| `write_reports.py` | Creates DATA, RESULTS and Recommendation narratives from measured output |
| `DATA.md` | Raw quality findings, exact files, full EDA coverage, preprocessing and source caveats |
| `RESULTS.md` | EDA plots, actual model/baseline metrics, split design, failure analysis and resolution caveats |
| `Recommendation.md` | One-page utility-facing pilot recommendation and measurement plan |
| `WALKTHROUGH.md` | Requirement mapping and detailed interview explanation of decisions and implementation |
| `reports/` | Measured audit CSV/JSON files, source manifest and plots |
| `tests/` | Behavioral regression tests for gap preservation, duplicate policy, feature leakage and energy units |
| `.github/workflows/tests.yml` | Lightweight unit-test workflow, without downloading the full dataset on every push |

The raw/cleaned archives, per-house records and intermediate caches are excluded from version control. The trained model and complete predictions are regenerated locally at `reports/model.joblib` and `reports/predictions.csv.gz`; never load joblib/pickle artifacts from untrusted sources. Summary metrics/figures are included so reviewers can inspect results without first downloading a gigabyte. Detailed per-event, outage and day-level tables remain local outputs (`cycles.csv`, `raw_outages.csv`, `daily_energy.csv`, `worst_days.csv`, `time_buckets.csv`) rather than public household-activity records; running the pipeline recreates them.

## Design choices

- All channels in raw House 1 are audited and cleaned. Stable last-record duplicate rule, explicit value caps, a per-minute minimum sample count and bounded two-minute interpolation preserve longer outages.
- Washing-machine EDA covers all 19 explicitly labelled households, treating the two House 4 machines separately. Washer-dryers and unknown labels are not silently relabelled.
- Model training uses Houses 1/2/5/6/9, March–June 2014; validation July; chronological test August. House 7 August is a separate untouched-house benchmark. Short context purges prevent overlapping windows across partitions.
- A 61-minute **offline** aggregate-only representation feeds histogram gradient boosting. The model consumes 30 minutes of future aggregate context and is not causal. Sequence-to-point here describes the input/output pattern; it is not the published Seq2Point CNN implementation.
- Compare squared/absolute loss on validation, plus zero/training-mean baselines. Keep natural class imbalance. Report active/off errors, activity F1/AP, energy error and per-home/day differences so a good off-state MAE cannot masquerade as NILM accuracy.
- Raw cleaning and model preprocessing are separate: the model uses official cleaned data without additional label interpolation. Official source forward filling and zero suppression remain limitations.

## Trade-offs and next work

CPU boosting is compact and quick to reproduce; a longer sequence CNN/TCN may improve multi-state recognition but must be evaluated on the same untouched splits. Coverage filters are deliberately conservative and create a conditional benchmark. Mean-per-minute sample averaging is a sample-count approximation; sensor-aware time integration and source freshness masks would improve energy confidence. Broader household/seasonal validation is more important than expanding a model leaderboard within this short exercise.

Next priorities: align raw/cleaned label freshness; multiple household-held-out folds; longer/causal sequence comparisons; input-level coarse-resolution retraining; calibration and abstention; an Indian submetered pilot. Do not promise energy savings from observational cross-house differences.

## AI assistance and ownership

OpenAI ChatGPT/Codex generated the implementation, helped inspect documentation, designed checks and drafted the report text. Code and reports were executed against real REFIT archives; synthetic test fixtures are used only for unit tests. The public report includes negative results and limits. There is no LLM in the inference pipeline and no external model API cost.

This disclosure reflects actual AI assistance. The candidate should review/run the code, verify the engineering choices and be prepared to explain and modify them before submission. This work does not claim eight years of personal experience or independent authorship by the candidate.

## Sources / adopted ideas

1. Murray, D., Stankovic, L. & Stankovic, V. (2017). *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study.* Scientific Data 4, 160122. https://doi.org/10.1038/sdata.2016.122 . Adopted dataset scope, sampling/measurement cautions and attribution.
2. University of Strathclyde. [REFIT dataset](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/) and [cleaned README](https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt), [raw README](https://pure.strath.ac.uk/ws/portalfiles/portal/62090176/RAW_READ_ME_081116.txt). Adopted verified household/channel labels, issue semantics and explicit caveats about gaps, clocks and appliances.
3. Zhang, C., Zhong, M., Wang, Z., Goddard, N. & Sutton, C. (2018). *Sequence-to-Point Learning With Neural Networks for Non-Intrusive Load Monitoring.* AAAI 32(1). https://doi.org/10.1609/aaai.v32i1.11873 . Adopted surrounding aggregate context → appliance midpoint prediction; **did not copy its neural architecture or claim its published results**.

Source data is CC BY 4.0; attribution must be retained. This repository's new code is MIT licensed; that license does not relicense REFIT data or source papers.
