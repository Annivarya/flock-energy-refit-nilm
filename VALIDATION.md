# Validation record

Executed locally against the official REFIT raw and cleaned archives on 9 October 2026.

- Official archives and both README files downloaded; byte counts/SHA256 checksums recorded in source_manifest.json.
- Full pipeline (raw cleaning → all-home EDA → model fit/evaluation) completed successfully.
- Raw/model stages rerun after final chart refinements; EDA logic was unchanged.
- All five behavioral tests passed; output in test_results.txt.
- Source scripts compiled successfully.
- Integration checks verified 19 EDA households / 20 channels, all ten raw power channels, and 418,203 validation/test/unseen predictions.
- All predictions are nonnegative and at most their corresponding aggregate power.
- All report Markdown image references resolve. Raw cleaning/outage, representative-day, comparison, normalized-use and prediction plots were visually reviewed.
- Measured metrics, coverage and uncertainty files exist. Reports were generated from those outputs.

GitHub Actions passed on the published source commit d27afb9dce283545c871652b07e15f07e429d6ed (run 37896080786). Dataset files and fitted model can be regenerated using README instructions.
