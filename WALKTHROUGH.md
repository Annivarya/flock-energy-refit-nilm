# Assignment and engineering walkthrough

## What Flock asked for

Estimate washing-machine power from the household's whole-home power signal: non-intrusive load monitoring (NILM). First demonstrate basic raw-data handling on House 1. Then analyze washing-machine behaviour across all explicitly labelled homes in the official cleaned REFIT product. Finally build a model, validate it without leakage and turn the evidence into one practical utility recommendation.

The scope is a manageable 4–6-hour exercise, with thoughtful trade-offs rather than exhaustive research. The requested submission is a **public GitHub repository** containing runnable scripts/notebooks, README.md, DATA.md, RESULTS.md and Recommendation.md. The email asks for the repository link in a reply and an acknowledgement of receipt. The email was received 8 October 2026 at 11:03 AM IST and sets a 48-hour deadline: 10 October at approximately 11:03 AM IST. No private-repository requirement appears in either the actual email or brief. No email is sent automatically by this project.

## Requirement-to-evidence map

| Expectation | Implementation/evidence |
| --- | --- |
| Raw House 1 inspection | `raw_cleaning`, `raw_quality.json`, all 10 power channels audited |
| Timestamp/order/duplicate/missing/outlier decisions | Stable epoch sorting, last duplicate wins, explicit caps and short-gap rules; DATA.md |
| One-minute aggregate and two appliance before/after plots | `raw_before_after.png`, `raw_gap_detail.png` |
| All known washing-machine homes and coverage | 19 homes / 20 channels; explicit WM map; `availability.csv` |
| Threshold/cycle quality and usage patterns | `cycles`, cycle-rule sensitivity, exposure-normalized local hour/day energy |
| Four requested EDA plot types | Representative day, cycle trace, usage plots and cross-home comparison |
| Aggregate-to-washer power estimation | 30 aggregate-only time-context features → histogram gradient boosting |
| Proper splitting and leakage prevention | March–June train; July validation; August test; House 7 unseen; context purges |
| Metrics and actual versus predicted | MAE/RMSE, active/off error, F1/AP, energy error, per-home/day metrics, prediction plots |
| Missing/change/unmetered-load discussion | DATA.md/RESULTS.md, issue filters, source filling caveats and House 13 swap |
| One practical saving opportunity | Eco/low-temperature wash coaching trial with submetered impact measurement |
| UK → India and 15/30-minute implications | Recommendation.md and resolution smoothing study with explicit limits |
| Reproducible and clear AI use | Pinned dependencies, seed, checksums, tests, CLI, measured reports, AI disclosure |

## Repository setup

A new, isolated directory is used for this project. Its source and report files are published in the dedicated public repository Annivarya/flock-energy-refit-nilm. `.gitignore` keeps downloaded household records, caches, virtual environments and local model artifacts out of public history. The recruiting email/PDF and personal information are not repository content. A lightweight GitHub Actions workflow runs behavioral tests, without repeatedly downloading the large dataset.

`download_data.py` downloads only the university's public archives/readmes using the alternate official `pure.strath.ac.uk/ws/portalfiles/portal` endpoint (the supplied pureportal file URL presented a challenge in this environment). SHA256 checksums and source URLs are recorded. A changed checksum is a reason to inspect the data, not silently rewrite the manifest. Partial downloads use temporary filenames so failed transfers do not appear complete. Extraction takes both raw House 1 parts and all cleaned-house files.

## The core logic, step by step

### 1. Establish the measurement contract

Power is instantaneous/interval-sampled **watts**, not watt-hours. One-minute mean power integrates to energy as `W / 60000` kWh per minute. Unix epoch is the common clock, interpreted as UTC; UK local time is used for scheduling charts. Raw clocks lack the official DST repair and are not used for the model's calendar interpretation.

There are 20 REFIT households, but House 12 has unknown target channels; 19 have explicit washing-machine labels. House 4 has two machines. Washer-dryers remain distinct and are not silently merged with washers. Official cleaned data includes historical corrections and forward filling: a complete-looking signal is not a guarantee of fresh sensor labels.

### 2. Clean raw House 1 conservatively

Audit both files separately, then combine them in a known source order. Inspect timestamp parsing, backward steps, interval distribution, duplicate epochs, per-channel missing/zero/negative/extreme values. Sorting stabilizes time order. Last duplicate wins instead of summing repeated readings.

High impossible appliance readings become missing, rather than being clipped to a plausible but invented load. One-minute means require three valid samples per channel. Only entire bounded gaps of at most two minutes are interpolated. A ten-minute outage stays completely blank. Outages and per-channel fill counts are saved, so readers can distinguish off from unknown. This raw pipeline deliberately does not reproduce all specialized official channel/DST repairs.

### 3. Describe washing-machine activity

Convert every labelled official cleaned channel to the same one-minute grid. A minute above 20 W is active. Merge active pieces separated by no more than ten observed off minutes, but split at missing observations. Reject very short/low-energy events and events touching missing-data or dataset boundaries.

The output is a **candidate operating event**, not a certified laundry cycle. The algorithm cannot see programme settings, garments or monitor freshness. Vary thresholds/bridging time and expose the event-count sensitivity rather than hiding the choice behind a precise-looking cycle count.

For each home/channel calculate coverage, missing minutes, zero/off proportions, flags, event count, duration, peak and energy. Plot a whole-home day alongside its washer, a cycle's internal phases, scheduling patterns and cross-home event distributions. Normalize energy by observed time so outages do not masquerade as inactivity.

### 4. Create aggregate-only time-window features

A single aggregate wattage value cannot identify which device produced it. Nearby transitions and phase structure help. For each prediction minute extract aggregate at offsets ±1/2/5/10/15/30 minutes and the current minute; calculate centred rolling mean, standard deviation, min and max at 5/15/31/61 minutes; add the first difference. This gives 30 features.

No target information, household identity, calendars, issue flags or appliance-derived totals are input features. Reject a context window if any aggregate minute is missing. The centred context needs 30 minutes of future readings; the model is an offline feedback estimator. Calling this real-time would be incorrect.

### 5. Partition before fitting or tuning

Use Houses 1/2/5/6/9 for March–June training, July validation and August testing. Every partition loses 30 minutes at each boundary so no window borrows readings from an adjacent partition. House 7 August is held out as a separate generalization check. Fit nothing on test houses/periods.

The training sample is a deterministic uniform subset (180,000 eligible minutes) for CPU efficiency, keeping real off/on imbalance. Validation/test use every eligible point. The model excludes inconsistent Issues labels and targets above aggregate; the benchmark is conditional on this quality filter. `split_coverage.csv` records excluded-data effects; reasons overlap and must not be summed as independent counts.

### 6. Fit a compact model with meaningful baselines

Histogram gradient boosting learns nonlinear relationships among the time-window features. The sample count, tree complexity and fixed iteration budget keep training practical on a CPU. Compare squared and absolute regression loss using an energy-aware validation objective. Absolute loss tends to the conditional median, which can be zero when the device is usually off; the experiment actually observes that failure.

Compare the selected model with zero and the training mean. Clip negative predicted appliance power to zero and power above the current aggregate to aggregate. This physical output bound uses only aggregate input, never ground truth. Save the fitted model locally; it is reproducible from code and source data.

### 7. Evaluate the right failure modes

Overall MAE alone rewards always-off predictions. RMSE emphasizes missed heating peaks; active MAE exposes error when the washer runs; off MAE captures false inferred energy; activity precision/recall/F1 and AP measure separation; energy error evaluates integrated consumption. Report every home and day rather than only a pooled score that cancels opposing errors.

Predictions and ground truth use identical coverage masks. Energy totals are over eligible observations, **not monthly appliance bills**. Bootstrap complete day blocks to preserve temporal dependence; those intervals remain conditional on this small household sample.

The chronological test's pooled energy error is about **5.3%**, with F1 about **0.338**. The unseen home's energy error is about **13.7%**, F1 about **0.479**. Zero achieves lower MAE but misses all appliance energy. The learned model improves RMSE and active error while still producing false positives. These results justify a limited pilot, not a claim of production-level recognition.

### 8. Turn findings into a measurable recommendation

Heating peaks and wide cycle-energy variation motivate a trial of eco/low-temperature wash coaching where garment care/hygiene permit. They do not prove that a high-energy household is inefficient. Use real submeters, programme/load information and a before/after or randomized control design to measure kWh per comparable laundry load and weekly energy. No observational percentage saving is promised.

For 15/30-minute meters, retrain on coarse input. Simply averaging one-minute predictions smooths errors but does not recreate a coarse-meter inference task. The supplied resolution CSV is explicitly a smoothing diagnostic. In India, validate local washer types, cold-water patterns, voltage, behaviour and seasons with held-out Indian homes before applying UK-derived rules.

## Checks completed and review expectations

The tests target real failure modes: long-gap partial filling, duplicate timestamps, bridging missing labels, window leakage and watt-to-kWh conversion. The complete data/EDA/model path is run on the downloaded official archives, and figures/metrics are inspected. GitHub CI passed on the published source commit (run 37896080786); all five behavioral tests also passed locally again before final publication.

AI assistance is disclosed in README. Before an interview, review the outputs and rerun the tests; be able to explain why zero can win MAE, why centred windows are offline, why pooled energy can hide errors, and why the cleaned source cannot prove label freshness. Those are the central engineering trade-offs in this submission.
