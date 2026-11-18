# Reproducing the published results

On 18 November 2026, the project was checked on Linux with Python 3.12.14 and the analysis dependency versions in [requirements.txt](../requirements.txt).

- All 23 automated tests passed.
- The command `python run_analysis.py` rebuilt the outputs from a clean output directory.
- All 4 published CSV tables matched the rerun, including 13,153 numeric entries checked with relative tolerance `1e-8` and absolute tolerance `1e-7`. Text columns and table dimensions matched.
- The run rebuilt 1 published chart. The saved notebook passed schema validation. All five code cells executed in order under IPython from the notebook folder, and its embedded chart matched the published PNG byte for byte.

The first run downloaded all four source archives. Their archive and extracted-file SHA-256 checksums matched the published provenance. The 25% allocation reproduced 13.19% annualised net growth and 1.73% tracking error.

Follow the [run instructions](RUNNING.md) to repeat the checks. The [source manifest](ORIGINAL_FILES.json) records checksums for the release files used to prepare this repository.
