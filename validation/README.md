# Two-branch real data validation

Status: adapters implemented, public-source URLs checked, **original RflyMAD and NASA data not available in this runtime**. No real-data metrics are claimed.

## A. RflyMAD diagnosis
Official: https://github.com/rfly-openha/documents/blob/main/4_resources/dataset.md
1. Download RflyMAD SampleData or processed real/SIL/HIL data from official links.
2. Inspect CSV headers and native flight/fault directory structure. Adapt `rfly_mapping_TEMPLATE.json`: keys must be REAL column names; value `timestamp_s` must be seconds; parent folder labels must actually designate faults. If not, add `fault_label` and `device_id` via a preprocessing step.
3. Execute: `python real_data_validation.py rfly --root /data/RflyMAD_processed --mapping mapping.json --out results/rfly`
4. Audit `ingestion_report.json` and `metrics.json`. Split by *flight*, not random time window. For strong real-world conclusions, add an out-of-domain real-flight-only test, leave-one-airframe-out testing, per-flight false alarm rate, event detection latency and time-to-detection.

## B. NASA battery RUL (real degradation data)
Official: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
1. Download **Battery Data Set** from NASA and extract B0005.mat/B0006.mat/B0007.mat/B0018.mat.
2. Run: `python real_data_validation.py nasa-battery --root /data/NASA_battery --out results/nasa_battery`
3. This baseline uses only discharge capacities; defines observed EOL as SOH <= 70% of first observed discharge; excludes units without observed EOL; leave-one-battery-out validation; outputs RUL in discharge cycles. A specimen may not reach the criterion; then 3+ observed EOL specimens are required or this script deliberately refuses to report an RUL score.
4. For bearing RUL, separately adapt NASA IMS/FEMTO high frequency vibration streams; do not treat instantaneous fault detection as RUL. Verify physical failure criteria, censoring and experimental time base.

## Quality gate
No synthetic fallback; metrics generated only if genuine input files exist and minimum validation conditions are met. Metrics are algorithmic research measures, not flightworthiness or operational safety certification.
