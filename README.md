# Quantifying Societal Stress (Old Bailey × Bills of Mortality)

This repository accompanies the LM4DH @ RANLP 2025 paper  
**“Quantifying Societal Stress: Forecasting Historical London Mortality using Hardship Sentiment and Crime Data with NLP and Time-Series.”**

## Overview
We extract **hardship sentiment** from Old Bailey trial texts with **MacBERTh** embeddings and relate it to **weekly mortality** from the London Bills of Mortality using CCF, Granger tests, VAR/IRF, and **Temporal Fusion Transformer (TFT)** forecasting.

## Data
- **Old Bailey Sessions Papers (XML, 1678–1849)** – local path to the TEI files.  
- **Weekly Bills of Mortality** – pipe-delimited file with `weekID|counttype|countn` (paper uses total mortality excluding `christened`).

> Please ensure you have permission to use and redistribute these datasets as per their licenses.

## Environment
Tested with **Python 3.10+**.

Minimal requirements:
```
torch>=2.2
pytorch-lightning>=2.2
pytorch-forecasting>=1.0.0
transformers>=4.41
sentence-transformers>=2.5
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.13
statsmodels>=0.14
joblib>=1.3
tqdm>=4.66
nltk>=3.8
symspellpy>=6.7
```

## Quick start
```
python script2.py \
  --old_bailey_dir /path/to/oldbailey/sessionsPapers \
  --mortality_file /path/to/BillsMortality/counts.txt \
  --outdir artifacts \
  --agg max \
  --lag_weeks 6 \
  --device auto \
  --seed 1337
```

This will:
- parse Old Bailey XML, compute **hardship** scores with MacBERTh,
- aggregate to **weekly** series and merge with mortality,
- save `artifacts/merged_weekly.csv`,
- generate publication-quality figures in `artifacts/plots/*.pdf`.

## Reproducing paper figures
- CCF / rolling corr: `artifacts/plots/figure_ccf.pdf`, `figure_rolling_corr.pdf`  
- Granger tests: `figure_granger.pdf`  
- VAR/IRF: `figure_irf_hardship_to_deaths.pdf`, `figure_irf_deaths_to_hardship.pdf`  
- TFT: `figure_tft_loss.pdf`, `figure_tft_forecasts.pdf`, `figure_tft_importance_encoder_decoder.pdf`

> Fonts are ≥8–9 pt and saved as **PDF** for two-column print quality.

## Configuration
Key flags:
- `--agg` ∈ {`mean`, `max`, `proportion`};  
- `--lag_weeks` (default: 6 to match the paper);  
- `--device` (`auto`/`cpu`/`cuda`);  
- `--seed` for determinism.

## Notes on validity
- The hardship measure is a **proxy** derived from embedding similarity; see paper’s **Validation & Robustness** and appended example snippets.  
- We report **predictive associations** (lead–lag); causal identification is **out of scope**.

## Citation
If you use this code or data processing, please cite the workshop paper.

```
@inproceedings{OlsenBloem2025,
  title={Quantifying Societal Stress: Forecasting Historical London Mortality using Hardship Sentiment and Crime Data with NLP and Time-Series},
  author={Olsen, Sebastian and Bloem, Jelke},
  booktitle={LM4DH Workshop @ RANLP 2025},
  year={2025}
}
```

## License
Code: MIT (see `LICENSE`). Data follow their original licenses.
