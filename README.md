# RANLP 2025 – Hardship Sentiment and Mortality

This repository accompanies the LM4DH @ RANLP 2025 paper  
**“Quantifying Societal Stress: Forecasting Historical London Mortality using Hardship Sentiment and Crime Data with NLP and Time-Series.”**

## Contents
- `hardship_mortality.py`: main script (entry point for analysis)  
- `README.md`: documentation and usage instructions  
- `LICENSE`: MIT license for code reuse  

The script extracts **hardship sentiment** from Old Bailey trial texts using **MacBERTh** embeddings and relates it to weekly mortality from the Bills of Mortality.  
We evaluate associations using CCF, Granger causality, and VAR/IRF, and perform forecasting with the **Temporal Fusion Transformer (TFT)**.

## Requirements
Tested with Python 3.10+.  

Install dependencies:
```
pip install torch pytorch-lightning pytorch-forecasting transformers sentence-transformers \
  pandas numpy matplotlib seaborn statsmodels joblib tqdm nltk symspellpy
```

## Usage
1. Edit the configuration block at the top of `hardship_mortality.py`:
   - `OLD_BAILEY_DIR`: path to Old Bailey XML files  
   - `COUNTS_FILE`: path to Bills of Mortality weekly counts  
   - optional: `HISTORICAL_DICT_PATH` for spelling normalization  

2. Run the script:
```
python hardship_mortality.py
```

3. Outputs:
   - merged weekly hardship–mortality dataset (`merged_weekly.csv`)  
   - plots (CCF, rolling correlations, Granger, VAR/IRF, TFT forecasts) in `artifacts/plots/`  

## Notes
- Results may vary slightly across library versions.  
- The hardship measure is a **proxy** based on embedding similarity; interpret findings as predictive associations, not causal claims.  

## Citation
If you use this code, please cite the workshop paper:

```
@inproceedings{OlsenBloem2025,
  title={Quantifying Societal Stress: Forecasting Historical London Mortality using Hardship Sentiment and Crime Data with NLP and Time-Series},
  author={Olsen, Sebastian and Bloem, Jelke},
  booktitle={LM4DH Workshop @ RANLP 2025},
  year={2025}
}
```

## License
Code: MIT License.  
Data follow their original licenses (Old Bailey Online, Bills of Mortality).
