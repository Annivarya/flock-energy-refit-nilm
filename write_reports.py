"""Generate submission narrative from measured experiment outputs, never placeholders."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
R=ROOT/'reports'

def table(df):
    # Avoid extra optional tabulate dependency.
    rows=[list(df.columns)]+df.fillna('—').astype(str).values.tolist()
    return '| '+' | '.join(rows[0])+' |\n| '+' | '.join(['---']*len(rows[0]))+' |\n'+ '\n'.join('| '+' | '.join(r)+' |' for r in rows[1:])

def main():
    raw=json.loads((R/'raw_quality.json').read_text())
    availability=pd.read_csv(R/'availability.csv')
    metrics=pd.read_csv(R/'metrics.csv')
    events=pd.read_csv(R/'cycles.csv')
    selection=json.loads((R/'model_selection.json').read_text())
    pooled=metrics[(metrics.house=='pooled') & metrics.split.isin(['test','unseen'])].copy()
    shown=pooled[['split','model','n','mae_w','rmse_w','active_mae_w','f1','relative_energy_error']]
    shown=shown.round(3)
    perhome=metrics[(metrics.model=='sequence_gbdt') & metrics.split.isin(['test','unseen']) & (metrics.house!='pooled')]
    perhome=perhome[['split','house','n','mae_w','rmse_w','f1','observed_true_kwh','observed_pred_kwh','relative_energy_error']].round(3)
    coverage=pd.read_csv(R/'split_coverage.csv')
    coverage_table=table(coverage[['house','split','embargoed_grid_minutes','eligible_minutes','eligible_pct']].round(2))
    scope=availability[['house','channel','start_utc','end_utc','grid_minutes','valid_minutes','events','median_cycle_min','median_cycle_kwh']].copy()
    scope[['median_cycle_min','median_cycle_kwh']]=scope[['median_cycle_min','median_cycle_kwh']].round(3)
    rawquality=pd.DataFrame(raw['source_channel_quality']).T.reset_index(names='channel')
    local=pd.to_datetime(events.start,utc=True).dt.tz_convert('Europe/London')
    hour=int(local.dt.hour.value_counts().idxmax())
    weekday=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][int(local.dt.dayofweek.value_counts().idxmax())]
    sensitivity=pd.read_csv(R/'cycle_sensitivity.csv').groupby(['threshold_w','bridge_min']).events.sum().reset_index()
    data=f'''# Data audit and decisions

## Sources and provenance

REFIT, University of Strathclyde, CC BY 4.0. The archive URLs, byte counts and SHA256 checksums are in [source_manifest.json](reports/source_manifest.json). `download_data.py` verifies future downloads against these recorded checksums. Raw and cleaned datasets are **different products**, not independent train/test sets. Archives and household records are deliberately excluded from Git; download them using the documented command. Public source documentation is linked in README, not the private recruiting email/PDF.

The documented nominal interval is 8 seconds, with active power in watts. Sensors are asynchronous (roughly 6–8 seconds). We analyze mean watts on a UTC one-minute grid; energy = sum(W × 1/60 hour)/1000. UK local time is used only for behavioural EDA. House IDs skip 14. House 12 has unknown labels, so it is excluded. Washer-dryer labels are excluded; House 4's two washing-machine channels remain separate. Exact channel mapping is in `pipeline.py:WM`.

## Raw House 1

- Files: `RAW_House1_Part1.csv`, `RAW_House1_Part2.csv`.
- {sum(f['rows'] for f in raw['files']):,} source rows; {raw['combined_duplicate_epochs']:,} duplicate epochs (before dropping invalid timestamps).
- Actual epoch coverage: **{raw['start_utc']} to {raw['end_utc']}**; {raw['grid_minutes']:,} minute slots.
- Parsing/order/interval details: [raw_quality.json](reports/raw_quality.json). One invalid/header-like row is dropped; Part 1 contains a backward step. Positive median intervals differ by file (5 and 7 seconds), so hard-coding an exact 8-second cadence is inappropriate. Text times differ from epochs by zero/one second, consistent with rounding. The raw clock does not have the official DST correction: treat raw epoch time as a consistent audit clock, not corrected UK calendar time.

{table(rawquality)}

## Cleaning policy

1. Convert Unix epochs to UTC timestamps; drop invalid epochs. Inspect text timestamp parsing separately.
2. Stable sort; last source record wins for a duplicate epoch. Part 2 follows Part 1. No summing duplicate power readings.
3. Convert negative values and appliance readings above 5,000 W to missing; aggregate upper cap is 30,000 W. These are broad engineering guardrails, not household electrical ratings. Record counts/maxima for **all** channels. Real aggregate readings up to 29,159 W survive; unusually high plausible values still need manual investigation. Official cleaned IAM cap is 4,000 W and is a separate historical policy.
4. Mean to one-minute bins; require at least three valid readings per channel per minute. This is a documented sample-count proxy, **not proof of 60 seconds of sensor coverage**. A time-weighted integration and raw sensor freshness flags would be a stronger next step.
5. Interpolate bounded missing runs of one or two minutes only. Never partially fill a longer run, fill an edge, or convert an outage to zero. The complete set of channels is cleaned. Gaps/outlier replacement are channel-specific.
6. Preserve remaining missing intervals; `reports/raw_outages.csv` (generated locally) records every remaining run. Interpolated points are for this raw cleaning exercise only; model training uses official cleaned readings with no additional label interpolation.

After cleaning, missing minutes per channel are {raw['after_missing_minutes']}. Short-gap filled counts are {raw['filled_minutes']}. See measured audit JSON for details.

![Raw cleaning](reports/raw_before_after.png)

This is the day of the largest unfiltered one-minute washing-machine load, selected deterministically to expose appliance extremes; the traces include the washing machine and dishwasher. Overlap indicates that ordinary readings were preserved. The plots compare resampled, unfiltered readings against resampled, cleaned readings, rather than every original high-frequency point.

![Outage detail](reports/raw_gap_detail.png)

The beginning of the longest aggregate missing run remains blank. This demonstrates outage preservation; blank regions are not off states.

## Official cleaned data limitations

The official README says missing readings were forward-filled, IAM spikes above 4,000 W replaced by zero, and some appliance columns moved to repair monitor changes. Zero can therefore be true off, suppressed spike, or early sensor unavailability. Those distinctions cannot be recovered from the cleaned data alone. Part 1 raw zeros also conflate unavailable/off. A regular minute index does not establish label freshness.

`Issues=1` means summed monitored loads exceed aggregate. **Descriptive appliance EDA retains these flagged readings** and reports flagged minutes and target-above-aggregate counts in [availability.csv](reports/availability.csv); appliance readings can still describe usage while meter totals disagree. Modeling excludes flagged target minutes and target > aggregate, with coverage disclosed. This creates a conservative, conditional benchmark and may exclude difficult overlapping loads. Because Issues is derived from submeter totals, this evaluation quality gate is not generally available to an aggregate-only deployed system; production needs independent meter-health checks and an unfiltered sensitivity benchmark. Aggregate feature windows must have no missing minutes, but issue flags are not model features.

House 13's washing machine changed on 25 March 2015; full-period EDA mixes both eras and should not be interpreted as a single stable appliance comparison. It is excluded from modeling. Other unrecorded swaps remain possible. House 4 has two labelled machines and is excluded from the single-target modeling cohort to keep the target definition unambiguous.

## EDA scope and coverage

All 19 homes with explicit washing-machine labels, 20 appliance channels, use their complete supplied coverage. No claim is made that every home has identical coverage. Missing/unusable counts, zero/off percentages and quality flags are in the CSVs.

{table(scope)}

The cleaned records' actual dates, including some July 2015 coverage, take precedence over the dataset page's approximate temporal summary. Nominal cadence is distinct from empirically observed interval distributions.
'''
    (ROOT/'DATA.md').write_text(data)
    results=f'''# Measured results and interpretation

## Washing-machine EDA

There are **{len(events):,} detected candidate operating events** across 19 houses / 20 channels. An active minute is >20 W. Merge active portions across at most ten **observed** off minutes; never bridge missing labels. Retain events at least ten minutes long and at least 0.02 kWh; reject events touching dataset/missing-data boundaries. These are heuristic operating events, not verified wash cycles: long pauses can split cycles, close washes can merge, and cold washes/standby can confuse detection. Report counts after the rule, not assumed household wash frequency.

Event-count sensitivity (full supplied coverage):

{table(sensitivity)}

The event-count peak is local hour **{hour}:00** and **{weekday}**, but homes have unequal exposure. Use the normalized energy chart below for exposure-aware comparisons, rather than concluding which day is intrinsically preferred from counts alone.

![Representative day](reports/representative_day.png)

Whole-home consumption contains substantial activity unrelated to the washing machine. The day is the first House 1 event in the 45–180 minute / 0.3–2 kWh range (first detected event fallback). It is a descriptive example, not a random sample or performance-selected trace.

![Cycle detail](reports/cycle_trace.png)

Heating, agitation and lower-load phases yield a multi-state signature. The 20 W threshold captures operating portions; it does not identify programme temperature or load size.

![Start patterns](reports/usage_patterns.png)

Event start counts by local hour/day show scheduling patterns but reflect uneven household coverage.

![Normalized energy](reports/usage_energy_normalized.png)

Energy divided by observed minute exposure (scaled to an observed hour) avoids treating unobserved periods as zero. It includes all valid appliance minutes, not just accepted events; it is pooled/exposure weighted, not an equal-household average.

![Home comparison](reports/cycle_comparison.png)

Energy/duration vary across selected homes. Boxes omit outlier markers for readability; all events remain in the CSV. Appliance model, wash setting, laundry size and event fragmentation are confounders. Higher energy does not prove appliance inefficiency.

## Experiment design

- Train: Houses **1, 2, 5, 6, 9**, 1 March–30 June 2014.
- Validation: same homes, July 2014.
- Locked chronological test: same homes, August 2014.
- Unseen-house test: **House 7**, August 2014, never used for training/model selection.
- Thirty minutes are purged at **both ends of every split**. A 61-minute centred feature window must lie wholly within its partition, producing a 60-minute separation across adjacent partitions. Missing context windows and invalid labels are excluded before sampling.
- {selection['training_rows_full']:,} eligible training minutes; a fixed-seed uniform subset of {selection['training_rows_used']:,} is used. Validation/test are unsampled and retain the natural class imbalance. No global scaling, feature selection or oversampling uses held-out data. No household ID, calendar, Issues, target lags, appliance power or energy-derived features enter the model.

The model is a compact **gradient-boosted sequence-to-point regressor**, not a reproduction of the Seq2Point CNN. Aggregate values at 13 offsets and centred rolling mean/std/min/max at four spans, plus a first difference, represent the surrounding signature. The longest span is 61 minutes. It is **offline**, requiring 30 minutes of future aggregate readings; it is not a real-time causal estimator. For causal operation retrain with trailing windows and reevaluate.

Both squared-error and absolute-error objectives use 140 boosting iterations, 31 leaves, minimum 50 rows/leaf, learning rate 0.08, L2=1 and seed 42. Internal random early stopping is disabled. Validation chooses the loss minimizing `MAE + 50 × relative energy error`; this illustrative energy-aware weight is declared in code, not a utility-calibrated cost. The selected loss is **{selection['selected_loss']}**. The absolute-error candidate collapses to off predictions under extreme class imbalance, illustrating why MAE alone is insufficient. No hyperparameters are selected on either test. Output is clipped to [0, aggregate] without consulting appliance labels.

## Baseline comparison

**All figures below are produced by running the supplied pipeline on actual REFIT files.**

{table(shown)}

Energy error is |sum(pred)−sum(true)|/sum(true), dimensionless (0.05=5%). It can hide opposing daily/home errors. MAE/RMSE are watts. Active MAE uses truth >20 W. F1 uses predicted >20 W. Average precision is also supplied in [metrics.csv](reports/metrics.csv) because prevalence is low. The training-mean baseline is learned from training only, capped by aggregate; zero predicts no use.

The sequence model improves RMSE, active-period error and energy capture versus zero. Overall MAE remains worse than zero, and activity F1 is modest. This is a defensible prototype for coarse energy feedback, **not precise appliance identification**. Pooled energy accuracy is insufficient evidence for household billing or control.

Per-home breakdown:

{table(perhome)}

Eligible model coverage (after the context embargo):

{coverage_table}

Exclusion reasons in [split_coverage.csv](reports/split_coverage.csv) overlap and cannot be summed. All energy totals are for **eligible observed test minutes only**, not complete monthly bills. Compare only matched masks; excluded minutes have no estimate. `reports/daily_energy.csv` (generated locally) reports evaluated minutes, so low-coverage days can be identified. [uncertainty.json](reports/uncertainty.json) provides 500 day-block bootstrap intervals on energy error, conditional on the selected homes. These do not estimate general uncertainty across all UK or Indian households.

![Seen-home predictions](reports/prediction_seen.png)

Highest actual-energy August test day in House 1, chosen to make appliance activity visible. It is intentionally labelled an illustrative, high-activity example; it is not cherry-picked for low error. Actual/prediction differences show signature ambiguity and amplitude error.

![Unseen-home predictions](reports/prediction_unseen.png)

Same deterministic selection for House 7. A new household can introduce unseen appliance/load combinations; no adaptation is performed.

![Daily energy](reports/daily_energy_scatter.png)

Distance from the diagonal exposes daily errors which may cancel in pooled energy totals. `reports/worst_days.csv` (generated locally) ranks the worst daily absolute energy errors without hiding failures.

## Failure analysis and limitations

1. Dishwashers, kettles, heaters and other simultaneous loads can resemble washing-machine heating. The low precision and nonzero off MAE show false positives; missed motor/low-load phases contribute to active MAE.
2. The 61-minute engineered representation is compact but cannot represent every multi-hour programme. Longer sequences / CNNs / TCNs need budgeted comparison against these same splits, not a claim of superiority without a test.
3. Source forward filling and zero replacement may produce stale/misleading labels. Filtering Issues and target > aggregate removes some difficult minutes; results are conditional on this quality gate.
4. Five training homes, one unseen home and one summer month do not establish seasonal or population generalization. A household-grouped cross-validation with multiple untouched homes is the next validation priority.
5. Appliance changes (including House 13), occupancy, behaviour and unmetered loads matter. Nine IAMs never measure every load, and asynchronous sampling can cause apparent physical inconsistency.

## 15-/30-minute resolution

Averaging reduces heating peaks and merges brief phases and overlapping loads. Long-window energy advice may remain possible, but exact event boundaries/signature separation worsen and duration uncertainty increases. [resolution_sensitivity.csv](reports/resolution_sensitivity.csv) averages matched 1-minute truth and predictions into complete 15/30-minute bins. It measures **smoothing of existing predictions only**, not performance of a model trained on coarse aggregate meters. Apparently improved errors/F1 after smoothing are not evidence that coarse input is equally informative. A real coarse-meter experiment must resample inputs first, retrain, retain household/time splits and evaluate energy and event detection separately.
'''
    (ROOT/'RESULTS.md').write_text(results)
    recommendation=f'''# Utility recommendation: pilot feedback, not appliance billing

**Decision:** use this prototype only in a consent-based, monitored household energy-feedback pilot. It estimates washing-machine consumption from aggregate power at one-minute resolution. The chronological/unseen-house tests show useful energy capture, but modest minute-level detection and uneven per-home errors. Do not use it to bill appliance consumption, automatically operate devices, or assert fault/inefficiency.

**Suitable uses:** indicative washing-energy trends on adequately observed days; identifying households that may benefit from a conversation about wash settings; aggregate programme evaluation with uncertainty and coverage shown. Only disclose estimates at suitable aggregation levels and clearly label them as inferred.

**One energy-saving opportunity:** offer a low-temperature/eco-wash coaching trial to participating homes with frequent high-energy detected events. The measured cycle-energy dispersion and observed heating peaks motivate a **testable hypothesis**, not proof that a specific household wastes energy. Ask about garment care and hygiene requirements, programme choice and full loads before suggesting a change. Do not infer wash temperature from power alone.

**Measure impact:** collect at least 2–4 weeks of reliable washer submeters plus aggregate readings before and after (or randomized encouragement/control). Record programme, temperature, laundry load, cycle counts and household occupancy. Compare kWh per comparable load/cycle and weekly laundry energy with confidence intervals, matched monitoring coverage and a control group; track participation, false alerts and satisfaction. Change in inferred NILM energy alone is not enough to establish savings. No numerical savings promise is supported by this observational dataset.

**Limits:** REFIT is a small 2013–2015 UK sample, with forward-filled labels, monitor changes, outages, inconsistent totals and overlapping appliance signatures. Home-to-home cycle energy is confounded by appliance and behaviour. One-minute means lose short transitions; 15/30-minute readings further merge phases and reduce identifiability. Re-averaging these model predictions is only a smoothing study, not validation on coarse meters.

**Before India rollout:** collect consented Indian aggregate and washer submeter data covering voltage variation, local top/front-loading models, cold-water usage, programme diversity, occupancy and seasonal patterns. Validate across entirely held-out homes and 1/15/30-minute input resolutions. Obtain fresh-label/monitor health flags, device-change logs, complete meter coverage and tariff context. Demonstrate actionable household-level accuracy and calibrated uncertainty before scaling. Do not transfer UK thresholds or economic savings directly.

**Next investment:** stronger label freshness checks; multiple held-out households and seasons; a causal/trailing-window comparison if timely alerts are needed; targeted sequence-model benchmark. Establish an abstention policy for missing context and low confidence, and monitor drift/false positives in the pilot.
'''
    (ROOT/'Recommendation.md').write_text(recommendation)

if __name__=='__main__':main()
