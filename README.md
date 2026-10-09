# REFIT washing-machine disaggregation

Reproducible code for raw electricity-data cleaning, washing-machine exploratory analysis and aggregate-only appliance disaggregation using the University of Strathclyde's public REFIT research dataset.

## Run

Use Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python download_data.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python pipeline.py --stage all
python write_reports.py
```

The pipeline generates DATA.md, RESULTS.md and Recommendation.md, plus figures, measured metrics and detailed audit tables locally. Source downloads, intermediate caches and fitted models are not committed. Publication of generated household-level statistics and plots is pending explicit approval; this repository currently contains runnable source and tests.

## Methods

Stable timestamp sorting, last-record duplicate handling, explicit outlier checks, one-minute resampling and bounded short-gap interpolation preserve longer outages. Washing-machine events are detected by a documented threshold/bridging rule with sensitivity checks. An offline sequence-to-point gradient-boosted model uses only aggregate-power features. Chronological validation and a held-out household assess temporal and household generalization, with purged feature windows to prevent overlap across partitions. Baselines, active/off errors, activity metrics, energy error and daily-block uncertainty expose weaknesses that overall MAE alone may hide.

The centred feature representation uses future aggregate context and is suitable for offline analysis, not causal real-time inference. Official source forward filling, meter inconsistencies, appliance changes and domain transfer remain limitations.

## AI assistance

OpenAI ChatGPT/Codex assisted implementation, documentation and verification. The pipeline has been executed locally against the official source archives. The candidate should review and be prepared to explain its decisions. No LLM API is used at inference time.

## Attribution

REFIT data: University of Strathclyde, CC BY 4.0. See https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/ and the official source README downloaded by the script.

Murray, D., Stankovic, L. & Stankovic, V. (2017). An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study. Scientific Data 4, 160122. https://doi.org/10.1038/sdata.2016.122

Zhang et al. (2018), Sequence-to-Point Learning With Neural Networks for Non-Intrusive Load Monitoring, AAAI. https://doi.org/10.1609/aaai.v32i1.11873 . Adopted the surrounding-context-to-midpoint formulation; this implementation uses gradient boosting and does not reproduce the published CNN.

New source code is MIT licensed; that license does not relicense REFIT or the cited papers.
