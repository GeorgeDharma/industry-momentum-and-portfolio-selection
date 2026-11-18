# Sizing an industry momentum portfolio

George Dharma

How much of an equal-industry portfolio should be allocated to the ten industries with the strongest lagged returns?

The 25% allocation returns 13.19% a year after costs, has 1.73% annual tracking error and limits an industry to 4.03% at rebalance. Its 95% interval for annual active return covers both −0.28 and +1.15 percentage points.

[Open the notebook](notebooks/analysis.ipynb) · [Read the report (PDF)](docs/report.pdf) · [View the result table](outputs/allocation_summary.csv) · [Download ZIP](https://github.com/GeorgeDharma/industry-momentum-and-portfolio-selection/archive/refs/heads/main.zip)

## Run the analysis

The notebook opens with its results already displayed. A full run downloads four source files from the Kenneth R. French Data Library and caches them locally. Use Python 3.12. Download and extract the ZIP, then follow the [Windows, macOS or Linux setup instructions](docs/RUNNING.md). In an activated environment, run these commands from the project folder.

```bash
python -m pip install -r requirements.txt
python run_analysis.py
python -m unittest discover -s tests -v
```

## Code and results

`equity/engine.py` ranks industries, funds rebalancing costs and estimates factor exposures. `equity/analysis.py` separates development, validation and the final test. `equity/allocation.py` simulates four single-account mixes and estimates the block intervals. `equity/allocation_figure.py` draws the interval plot.

`selection_lock.json` records the validation decision. `allocation_summary.csv` reports portfolio metrics, `allocation_monthly.csv` contains every monthly mix, and `active_industries.csv` attributes the full sleeve’s gross active performance. The new sizing question revisits the 2010–2025 period.

## Source

Kenneth R. French Data Library, 49 US industry portfolios, US research factors and stock momentum. The file `data/source/provenance.json` records the URLs, archive members and SHA-256 hashes of the inputs used for the published results. The publisher revises its data. When the current download differs, the script reports the affected series and records its hashes in `data/source/downloaded.json`. The sample dates stay fixed. Historical revisions can change the results.

[Original source](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_49_ind_port.html)

To verify that downloaded data match the published snapshot, run `python fetch_data.py --exact`. For offline reproduction with the original four ZIP files, run `python fetch_data.py --from-dir "/path/to/archives" --exact` before running the analysis. Local archives can use their original publisher filenames or the names industry.zip, factors.zip, momentum.zip and definitions.zip.

Run `python run_analysis.py --stage develop` to produce the validation lock and `python run_analysis.py --stage test` to run the final test using that lock.

[Methodology](docs/METHODOLOGY.md) · [Validation](docs/VALIDATION.md)
