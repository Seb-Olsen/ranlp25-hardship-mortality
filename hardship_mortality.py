#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Historical Analysis Script - Old Bailey Sessions Papers & Weekly Bills of Mortality
Analyzes correlation between WEEKLY fear sentiment (derived using MacBERTh embeddings
from Old Bailey) and WEEKLY mortality trends using TFT forecasting.

*** TARGET ENVIRONMENT: Python 3.10+, pytorch-forecasting (latest compatible),
                       pytorch-lightning (latest compatible), transformers, sentence-transformers,
                       torch, nltk, pandas, numpy, matplotlib, seaborn, joblib, symspellpy, statsmodels ***
"""

import os
import logging
import re
import warnings
from datetime import datetime, timedelta # Added timedelta
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
import torch
from typing import Dict, Tuple, Union, List, Optional
from tqdm.auto import tqdm
from statsmodels.tsa.stattools import ccf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from statsmodels.tools.sm_exceptions import ValueWarning, EstimationWarning

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*Checkpoint directory*")
warnings.filterwarnings("ignore", ".*MPS available but not used.*")
warnings.filterwarnings("ignore", ".*does not have valid feature names*")

# ---------------------
# Basic Setup
# ---------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import argparse, random
def set_seed(seed:int=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------
# NLTK Setup & Download
# ---------------------
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer

def download_nltk_data():
    """Downloads required NLTK data if not already present."""
    logger.info("Checking/Downloading required NLTK data...")
    required_packages = [
        ('punkt', 'tokenizers/punkt'),
        ('wordnet', 'corpora/wordnet'),
        ('stopwords', 'corpora/stopwords'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')
    ]
    downloader = nltk.downloader.Downloader()
    for package_id, path in required_packages:
        package_found = False
        try: nltk.data.find(path); logger.debug(f"NLTK data '{package_id}' found."); package_found = True
        except LookupError: package_found = False
        except Exception as e: logger.warning(f"NLTK check failed: {e}"); package_found = False
        if not package_found:
            logger.info(f"Downloading NLTK package: {package_id}")
            try:
                force_dl = (package_id == 'punkt')
                if not downloader.download(package_id, quiet=True, force=force_dl):
                    try: nltk.data.find(path); logger.info(f"NLTK '{package_id}' found after check.")
                    except LookupError: raise RuntimeError(f"Failed download/locate: {package_id}")
                else: logger.info(f"NLTK '{package_id}' downloaded.")
            except Exception as e: logger.error(f"NLTK download error: {e}", exc_info=True); raise
download_nltk_data()

# ---------------------
# Other Imports
# ---------------------
from symspellpy.symspellpy import SymSpell, Verbosity
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import grangercausalitytests
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import GroupNormalizer, TorchNormalizer, EncoderNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from joblib import Memory

# ---------------------
# Configuration (UPDATED FOR OLD BAILEY & WEEKLY ANALYSIS)
# ---------------------
OLD_BAILEY_DIR = ""   # <-- fill with local path to Old Bailey XMLs
COUNTS_FILE = ""      # <-- fill with Bills of Mortality counts file
HISTORICAL_DICT_PATH = "historical_dict.txt" # Optional
START_YEAR = 1678 # Old Bailey data starts around here
END_YEAR = 1849   # Align end year with previous analysis for now
INFECTIOUS_DISEASE_TYPES = None # Use None for total mortality (excluding christened)

# Word lists for targeted sentiment
DISEASE_FEAR_WORDS = ["pestilence", "plague", "contagion", "infection", "epidemic", "pox", "sickness", "disease", "illness", "malady", "distemper", "smallpox", "sweating"]
HARDSHIP_WORDS = ["poor", "poverty", "necessity", "distress", "hardship", "starve", "desperate", "ruin", "beggar", "vagrant", "hunger", "want"]
VIOLENCE_SEVERITY_WORDS = ["kill", "murder", "murther", "wound", "stab", "pistol", "weapon", "sword", "hanger", "deadly", "death", "slay", "violence"]

# Define the LAG_WEEKS constant
LAG_WEEKS = 4

# --- Sentiment Analysis Config ---
MACBERTH_MODEL_NAME = 'emanjavacas/MacBERTh'
FEAR_WORDS = [ # Refined list including crime context
    "fear", "afraid", "scared", "terror", "dread", "panic", "anxiety", "worry",
    "horror", "phobia", "fright", "alarm", "apprehension", "nervousness",
    "trembling", "timidity", "consternation", "distress", "unease", "danger",
    "pestilence", "plague", "contagion", "infection", "epidemic", "pox",
    "sickness", "disease", "illness", "malady", "distemper",
    "dying", "death", "mortality", "fatal", "corpse", "grave", "burial",
    "suffering", "agony", "misery", "calamity", "crisis",
    "murder", "robbery", "theft", "violence", "crime", "villain", "malefactor",
    "guilty", "hanged", "executed", "sentence", "prison", "newgate", "punishment"
]
FEAR_WORDS = list(set(FEAR_WORDS)) # Unique words

# --- Aggregation Method (Now WEEKLY) ---
AGGREGATION_METHOD = 'max'

# --- TFT Parameters (Weekly) ---
MAX_EPOCHS = 100
BATCH_SIZE = 32 # Can often increase slightly for weekly data vs monthly
LEARNING_RATE = 4.5e-5
HIDDEN_SIZE = 32
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.36
HIDDEN_CONTINUOUS_SIZE = 8
WEEKLY_MAX_ENCODER_LENGTH = 40 # Use 40 weeks of history
WEEKLY_MAX_PREDICTION_LENGTH = 8  # Predict 8 weeks ahead
GRADIENT_CLIP_VAL = 0.1

APPLY_TARGET_SMOOTHING_EXPERIMENT = True # Set to True to run this experiment
TARGET_SMOOTHING_WINDOW = 3             # e.g., 3-week rolling median
TARGET_SMOOTHING_TYPE = 'median' 

HARDSHIP_ONLY = True

# --- Plotting Directory ---
PLOT_DIR = "plots_weekly_oldbailey" # New plot dir
if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)

# --- Sentiment‑sampling output ---
SENTIMENT_SAMPLE_DIR = os.path.join(PLOT_DIR, "sentiment_samples")
os.makedirs(SENTIMENT_SAMPLE_DIR, exist_ok=True)
TOP_SENTIMENT_SAMPLE_N = 30       # rows to save from top of the distribution
BOTTOM_SENTIMENT_SAMPLE_N = 30    # rows to save from bottom
RANDOM_SENTIMENT_SAMPLE_N = 40    # random rows

# --- Device Setup ---
logger.info("Forcing CPU due to potential MPS compatibility issues or preference.")
DEVICE = torch.device("cpu")
logger.info(f"Using device: {DEVICE}")

# --- Caching Setup ---
cache_dir = "cache_dir_weekly_oldbailey2" # New cache dir
if not os.path.exists(cache_dir): os.makedirs(cache_dir)
memory = Memory(cache_dir, verbose=0)


# -----------------------------------------------------------------------------
# 1. Text Normalization & Preprocessing Helpers (Unchanged)
# -----------------------------------------------------------------------------
def normalize_historical_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.replace("ſ", "s")
    return text

def get_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith('J'): return wn.ADJ
    elif treebank_tag.startswith('V'): return wn.VERB
    elif treebank_tag.startswith('N'): return wn.NOUN
    elif treebank_tag.startswith('R'): return wn.ADV
    else: return wn.NOUN

# -----------------------------------------------------------------------------
# 2. SymSpell Setup & Correction (Optional, Unchanged)
# -----------------------------------------------------------------------------
@memory.cache
def setup_symspell(dictionary_path=HISTORICAL_DICT_PATH, max_edit_distance=1):
    if not os.path.exists(dictionary_path): logger.warning(f"SymSpell dict NF: {dictionary_path}. Skip."); return None
    try:
        sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
        try: loaded = sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')
        except UnicodeDecodeError: logger.warning("UTF-8 failed, try latin-1."); loaded = sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='latin-1')
        if not loaded: logger.error(f"Failed load SymSpell dict: {dictionary_path}"); return None
        logger.info(f"SymSpell loaded (max_edit={max_edit_distance})."); return sym_spell
    except Exception as e: logger.error(f"SymSpell setup error: {e}", exc_info=True); return None
sym_spell_global = setup_symspell()

def correct_ocr_spelling(text: str, sym_spell: Optional[SymSpell]) -> str:
    if not sym_spell or not isinstance(text, str) or not text.strip(): return text
    words = text.split(); corrected_words = []
    for word in words:
        clean_word = word.strip('.,!?;:"()[]')
        if not clean_word or not clean_word.isalpha(): corrected_words.append(word); continue
        suggestions = sym_spell.lookup(clean_word, Verbosity.CLOSEST, max_edit_distance=sym_spell.max_dictionary_edit_distance, include_unknown=True)
        if suggestions:
            best_suggestion = suggestions[0].term; apply_correction = best_suggestion.lower() != clean_word.lower()
            if apply_correction:
                if word.istitle() and len(word) > 1: corrected_words.append(best_suggestion.capitalize())
                elif word.isupper() and len(word) > 1: corrected_words.append(best_suggestion.upper())
                else: corrected_words.append(best_suggestion)
            else: corrected_words.append(word)
        else: corrected_words.append(word)
    return " ".join(corrected_words)

# -----------------------------------------------------------------------------
# 3. Core Text Preprocessing Function (Unchanged)
# -----------------------------------------------------------------------------
def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set, sym_spell: Optional[SymSpell] = None, use_symspell: bool = False) -> str:
    if not isinstance(text, str) or not text.strip(): return ""
    text = normalize_historical_text(text)
    if use_symspell and sym_spell: text = correct_ocr_spelling(text, sym_spell)
    text_cleaned = re.sub(r"[^\w\s]", " ", text); text_cleaned = re.sub(r"\d+", "", text_cleaned)
    text_cleaned = text_cleaned.lower(); text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
    if not text_cleaned: return ""
    try: tokens = nltk.word_tokenize(text_cleaned)
    except Exception as e: logger.warning(f"Tokenization failed: {e}"); return ""
    if not tokens: return ""
    try: tagged_tokens = nltk.pos_tag(tokens)
    except Exception as e: logger.warning(f"POS Tagging failed: {e}. Default noun."); tagged_tokens = [(t, 'NN') for t in tokens]
    processed_tokens = []
    for word, tag in tagged_tokens:
        if word not in stop_words and len(word) > 1:
            try: lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)); processed_tokens.append(lemma)
            except Exception as e: logger.warning(f"Lemmatization failed: {word}/{tag}: {e}"); processed_tokens.append(word)
    return " ".join(processed_tokens)

# -----------------------------------------------------------------------------
# 4. Cached Data Loading and Processing Functions (UPDATED FOR WEEKLY)
# -----------------------------------------------------------------------------

@memory.cache
def parse_old_bailey_papers(ob_dir: str = OLD_BAILEY_DIR, start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    """
    (Cached) Parses Old Bailey Sessions Papers XML files (TEI.2 format).
    Extracts session start date, text, AND structured trial data
    (primary offence category, verdict category, punishment category) for each trial account.
    Maps date to the start of the ISO week (Monday).
    Returns DataFrame with ['week_date', 'doc_id', 'trial_id', 'text', 'offence_cat', 'verdict_cat', 'punishment_cat'].
    """
    records = []
    logger.info(f"Starting Old Bailey Sessions Papers parsing (Years: {start_year}-{end_year}) - Extracting Structured Data...")
    file_count = 0; processed_trials = 0; skipped_date = 0; parse_errors = 0; date_parse_attempts = 0

    for rootdir, _, files in os.walk(ob_dir):
        for fname in files:
            if not (fname.endswith('.xml') or re.match(r'^\d{8}$', fname) or '.' not in fname): continue
            file_count += 1
            if file_count % 500 == 0: logger.info(f" Scanning file {file_count}...")

            fpath = os.path.join(rootdir, fname)
            doc_id_base = os.path.splitext(fname)[0] # Use filename as base doc id

            try:
                tree = ET.parse(fpath)
                root_tei = tree.getroot() # <TEI.2>

                # --- Extract Session Date (from <div0 type="sessionsPaper">) ---
                session_div = root_tei.find('.//div0[@type="sessionsPaper"]')
                session_date_str = None
                session_date = None
                doc_id = doc_id_base # Default doc_id

                if session_div is not None:
                    # Use ID from session_div if available and looks like a date
                    if 'id' in session_div.attrib and re.match(r'^\d{8}$', session_div.attrib['id']):
                        session_date_str = session_div.attrib['id']
                        doc_id = session_date_str # Prefer date string as doc_id
                        date_parse_attempts += 1
                    # Fallback to interp date if div0 id is missing/invalid
                    interp_date_node = session_div.find('.//interp[@type="date"]')
                    if session_date_str is None and interp_date_node is not None and 'value' in interp_date_node.attrib:
                         session_date_str = interp_date_node.attrib['value']
                         if re.match(r'^\d{8}$', session_date_str):
                            doc_id = session_date_str # Use date string as doc_id
                            date_parse_attempts += 1
                         else: session_date_str = None # Ignore invalid date format

                if session_date_str:
                    try:
                        session_date = datetime.strptime(session_date_str, '%Y%m%d')
                        if not (start_year <= session_date.year <= end_year):
                            skipped_date += 1; continue
                    except ValueError:
                        logger.warning(f"Date parse failed '{session_date_str}' in {fname}. Skip file."); skipped_date += 1; continue
                else:
                    logger.debug(f"No valid session date found in {fname}. Skip file."); skipped_date += 1; continue

                # Map Session Date to Start of ISO Week (Monday)
                iso_year, iso_week, _ = session_date.isocalendar()
                try: week_start_date = datetime.fromisocalendar(iso_year, iso_week, 1)
                except ValueError: logger.warning(f"Week start date fail for {session_date_str} in {fname}. Skip file."); skipped_date += 1; continue

                # Check against pandas min date
                min_pandas_date = datetime(1678, 1, 1)
                if week_start_date < min_pandas_date:
                    # logger.debug(f"Skipping file {fname}, date {week_start_date} before pandas min.")
                    skipped_date += 1; continue

                # --- Iterate through Trial Accounts (<div1 type="trialAccount">) ---
                trial_accounts = root_tei.findall('.//div1[@type="trialAccount"]')
                if not trial_accounts:
                    # logger.debug(f"No trial accounts found in {fname}.")
                    continue

                for trial_div in trial_accounts:
                    trial_id = trial_div.get('id', f"{doc_id}_trial_{processed_trials+1}") # Get trial ID or create one

                    # Extract Text for this trial
                    trial_text_parts = []
                    for p_node in trial_div.findall('.//p'):
                        node_text = ' '.join(t.strip() for t in p_node.itertext() if t and t.strip())
                        node_text_clean = re.sub(r'\s+', ' ', node_text).strip()
                        if node_text_clean: trial_text_parts.append(node_text_clean)
                    trial_text = ' '.join(trial_text_parts) if trial_text_parts else ""

                    # Extract Structured Info (taking the first one found for simplicity, might need refinement)
                    offence_cat = None; verdict_cat = None; punishment_cat = None

                    offence_interp = trial_div.find('.//rs[@type="offenceDescription"]/interp[@type="offenceCategory"]')
                    if offence_interp is not None and 'value' in offence_interp.attrib:
                        offence_cat = offence_interp.get('value')

                    verdict_interp = trial_div.find('.//rs[@type="verdictDescription"]/interp[@type="verdictCategory"]')
                    if verdict_interp is not None and 'value' in verdict_interp.attrib:
                        verdict_cat = verdict_interp.get('value')

                    punishment_interp = trial_div.find('.//rs[@type="punishmentDescription"]/interp[@type="punishmentCategory"]')
                    if punishment_interp is not None and 'value' in punishment_interp.attrib:
                        punishment_cat = punishment_interp.get('value')

                    # Only add record if we have at least an offence or verdict
                    if offence_cat or verdict_cat:
                        records.append({
                            'week_date': week_start_date,
                            'doc_id': doc_id, # Session document ID
                            'trial_id': trial_id, # Individual trial ID
                            'text': trial_text, # Text specific to this trial (optional)
                            'offence_cat': offence_cat,
                            'verdict_cat': verdict_cat,
                            'punishment_cat': punishment_cat
                        })
                        processed_trials += 1
                        if processed_trials % 1000 == 0: logger.info(f" Found {processed_trials} valid trial records...")

            except ET.ParseError as e: logger.warning(f"XML Parse Error {fname}: {e}"); parse_errors += 1
            except Exception as e: logger.warning(f"General Error processing {fname}: {e}", exc_info=False); parse_errors += 1

    logger.info(f"Finished Old Bailey parsing. Files scanned: {file_count}")
    if processed_trials == 0: logger.error("CRITICAL: No trial records processed. Check XML structure or parsing logic.")
    else: logger.info(f" Processed {processed_trials} valid trial account records.")
    logger.info(f" Files skipped due to date issues: {skipped_date}.")
    logger.info(f" Errors during parsing: {parse_errors}.")
    if not records: return pd.DataFrame(columns=['week_date', 'doc_id', 'trial_id', 'text', 'offence_cat', 'verdict_cat', 'punishment_cat'])

    # Create DataFrame
    df = pd.DataFrame(records)
    df['week_date'] = pd.to_datetime(df['week_date']) # Convert date column
    # Clean categories slightly
    for col in ['offence_cat', 'verdict_cat', 'punishment_cat']:
        df[col] = df[col].str.lower().str.strip().fillna('unknown')

    logger.info(f"Old Bailey Structured DataFrame prepared: {df.shape[0]} trial records. Date Range: {df['week_date'].min():%Y-%m-%d} to {df['week_date'].max():%Y-%m-%d}")
    logger.info(f"Sample Offence Categories: {df['offence_cat'].value_counts().head().to_dict()}")
    logger.info(f"Sample Verdict Categories: {df['verdict_cat'].value_counts().head().to_dict()}")
    logger.info(f"Sample Punishment Categories: {df['punishment_cat'].value_counts().head().to_dict()}")

    return df


# === Mortality Loading (WEEKLY - UPDATED) ===
def parse_bill_weekID_to_weekly(week_str: str) -> Optional[datetime]:
    """Parses Bills of Mortality weekID (YYYY/WW) to a datetime object for the START of the ISO WEEK (Monday)."""
    try:
        year_str, week_ = week_str.split("/")
        year = int(year_str); week = int(week_)
        # Use START_YEAR and END_YEAR from config
        if not (START_YEAR <= year <= END_YEAR) or not (1 <= week <= 53): return None
        # Calculate Monday of the given ISO year and week
        return datetime.fromisocalendar(year, week, 1)
    except ValueError:
        # Handle week 53 issue for years that don't have it
        if week == 53:
            try: return datetime.fromisocalendar(year, 52, 1) # Fallback to week 52
            except ValueError: logger.debug(f"Invalid week 52/53 for year {year_str}."); return None
        else: logger.debug(f"Cannot parse week {week_} for year {year_str}."); return None
    except Exception as e: logger.debug(f"Error parsing bill weekID '{week_str}' to weekly: {e}"); return None

def validate_trial_sentiment_scores(df_sentiment_trials: pd.DataFrame,
                                    text_col_original: str = 'text',
                                    text_col_processed: str = 'processed_text',
                                    n_samples_spot_check: int = 15,
                                    min_text_length: int = 30): # Added parameter
    """
    Performs qualitative spot-checking and correlates sentiment scores with keyword counts
    at the trial level, AFTER filtering trials by minimum text length. Prints results to the log.
    """
    logger.info("\n--- Validating Trial-Level Sentiment Scores (Filtered by Text Length) ---")

    if df_sentiment_trials.empty:
        logger.warning("Input DataFrame for sentiment validation is empty. Skipping.")
        return

    # Ensure required text columns are present for filtering and analysis
    if text_col_original not in df_sentiment_trials.columns:
        logger.error(f"Column '{text_col_original}' not found. Cannot perform text length filter or spot check.")
        return
    if text_col_processed not in df_sentiment_trials.columns:
        logger.warning(f"Column '{text_col_processed}' not found. Keyword correlation will be skipped.")
        # Allow spot checking to proceed if original text is available

    # --- Filter by Text Length FIRST ---
    logger.info(f"Applying text length filter: trials with original text length > {min_text_length} words.")
    # Calculate word count on the original text for filtering
    df_sentiment_trials['word_count_for_validation_filter'] = df_sentiment_trials[text_col_original].apply(lambda x: len(str(x).split()))
    df_validated_filtered = df_sentiment_trials[
        df_sentiment_trials['word_count_for_validation_filter'] > min_text_length
    ].copy()

    if df_validated_filtered.empty:
        logger.warning(f"No trials meet the text length criteria (>{min_text_length} words) for validation. Skipping.")
        return
    logger.info(f"{len(df_validated_filtered)} trials remaining for validation after text length filter.")
    # --------------------------------

    sentiment_cols = [col for col in df_validated_filtered.columns if col.endswith('_sentiment')]
    if not sentiment_cols:
        logger.warning("No sentiment score columns found in the filtered DataFrame. Skipping validation.")
        return

    sentiment_to_keywords_map = {
        'hardship_sentiment': HARDSHIP_WORDS
    }

    for sent_col in sentiment_cols:
        logger.info(f"\n--- Validating: {sent_col.upper()} (on length-filtered data) ---")
        if sent_col not in df_validated_filtered.columns:
            logger.warning(f"Sentiment column '{sent_col}' not found after filtering. Skipping.")
            continue

        # --- 1. Qualitative Spot-Checking (on filtered data) ---
        logger.info(f"--- Top {n_samples_spot_check} Trials for HIGH {sent_col} ---")
        df_sorted_high = df_validated_filtered.sort_values(by=sent_col, ascending=False)
        for index, row in df_sorted_high.head(n_samples_spot_check).iterrows():
            text_to_display = row.get(text_col_original, "TEXT NOT AVAILABLE")
            logger.info(f"  Trial ID: {row.get('trial_id', 'N/A')}, Score: {row[sent_col]:.4f}, Word Count: {row['word_count_for_validation_filter']}")
            logger.info(f"  Text Sample: {text_to_display[:250]}...")

        logger.info(f"--- Bottom {n_samples_spot_check} (Non-Zero) Trials for LOW {sent_col} ---")
        df_non_zero_low = df_sorted_high[df_sorted_high[sent_col] > 0.01].sort_values(by=sent_col, ascending=True) # Use df_sorted_high here
        if df_non_zero_low.empty:
            logger.info(f"  No trials found with {sent_col} > 0.01 for low score spot-checking in filtered data.")
        else:
            for index, row in df_non_zero_low.head(n_samples_spot_check).iterrows():
                text_to_display = row.get(text_col_original, "TEXT NOT AVAILABLE")
                logger.info(f"  Trial ID: {row.get('trial_id', 'N/A')}, Score: {row[sent_col]:.4f}, Word Count: {row['word_count_for_validation_filter']}")
                logger.info(f"  Text Sample: {text_to_display[:250]}...")

        # --- 3. Correlation with Keyword Counts (on filtered data, using processed_text if available) ---
        keyword_list = sentiment_to_keywords_map.get(sent_col)
        if keyword_list and text_col_processed in df_validated_filtered.columns:
            keyword_count_col = f"{sent_col}_keyword_count_filt" # Use different name to avoid clash if run multiple times
            logger.info(f"Calculating keyword counts for '{sent_col}' using '{text_col_processed}' on filtered data...")
            
            # Use the already filtered df_validated_filtered
            temp_df_corr = df_validated_filtered[[text_col_processed, sent_col]].copy()
            temp_df_corr.dropna(subset=[text_col_processed, sent_col], inplace=True) # Ensure no NaNs in these specific cols

            temp_df_corr[keyword_count_col] = temp_df_corr[text_col_processed].apply(
                lambda text: sum(1 for word in str(text).split() if word in keyword_list)
            )

            if not temp_df_corr.empty and temp_df_corr[sent_col].nunique() > 1 and temp_df_corr[keyword_count_col].nunique() > 1:
                try:
                    correlation = temp_df_corr[sent_col].corr(temp_df_corr[keyword_count_col])
                    logger.info(f"  Correlation (filtered data) between '{sent_col}' and its Keyword Count: {correlation:.3f}")
                except Exception as e:
                    logger.warning(f"  Could not calculate correlation for '{sent_col}' on filtered data: {e}")
            else:
                logger.warning(f"  Not enough variance or data to calculate correlation for '{sent_col}' on filtered data.")
        elif not keyword_list:
            logger.warning(f"  No keyword list defined for '{sent_col}', skipping keyword correlation.")
        elif text_col_processed not in df_validated_filtered.columns: # Check on the right df
            logger.warning(f"  '{text_col_processed}' column not found in filtered data, skipping keyword correlation for '{sent_col}'.")

    # --- Save top, bottom, and random samples for manual review ---
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        score_min, score_max = df_validated_filtered[sent_col].min(), df_validated_filtered[sent_col].max()
        pct_col = f"{sent_col}_pct"
        # normalise 0‑100
        df_validated_filtered[pct_col] = (
            (df_validated_filtered[sent_col] - score_min) /
            (score_max - score_min + 1e-9) * 100
        ).round(2)

        cols_to_save = ['trial_id', 'week_date', sent_col, pct_col, text_col_original]

        top_df = df_validated_filtered.sort_values(sent_col, ascending=False)\
                                      .head(TOP_SENTIMENT_SAMPLE_N)[cols_to_save]
        bottom_df = df_validated_filtered.sort_values(sent_col, ascending=True)\
                                         .head(BOTTOM_SENTIMENT_SAMPLE_N)[cols_to_save]
        random_df = df_validated_filtered.sample(
            min(RANDOM_SENTIMENT_SAMPLE_N, len(df_validated_filtered))
        )[cols_to_save]

        top_path = os.path.join(SENTIMENT_SAMPLE_DIR,
                                f"top_{TOP_SENTIMENT_SAMPLE_N}_{sent_col}_{timestamp}.csv")
        bottom_path = os.path.join(SENTIMENT_SAMPLE_DIR,
                                   f"bottom_{BOTTOM_SENTIMENT_SAMPLE_N}_{sent_col}_{timestamp}.csv")
        random_path = os.path.join(SENTIMENT_SAMPLE_DIR,
                                   f"random_{RANDOM_SENTIMENT_SAMPLE_N}_{sent_col}_{timestamp}.csv")

        top_df.to_csv(top_path, index=False)
        bottom_df.to_csv(bottom_path, index=False)
        random_df.to_csv(random_path, index=False)

        logger.info(f"Saved sentiment‑sample CSVs to {SENTIMENT_SAMPLE_DIR}")
    except Exception as csv_err:
        logger.error(f"Could not write sentiment‑sample CSVs for '{sent_col}': {csv_err}")

    logger.info("--- End of Trial-Level Sentiment Score Validation (Filtered by Text Length) ---")
    
# === Text Preprocessing (Unchanged Function, applied to Old Bailey Text) ===
@memory.cache
def preprocess_text_dataframe(df: pd.DataFrame, text_col: str = "text", use_symspell: bool = False) -> pd.DataFrame:
    logger.info(f"Preprocessing text column '{text_col}' (use_symspell={use_symspell})...")
    if text_col not in df.columns: raise ValueError(f"Column '{text_col}' not found.")
    df_copy = df.copy(); df_copy[text_col] = df_copy[text_col].astype(str).fillna('')
    lemmatizer = WordNetLemmatizer(); stop_words = set(stopwords.words("english"))
    global sym_spell_global
    total_rows = len(df_copy); processed_texts = []
    # Preprocessing can be slow, log progress less frequently
    log_interval = max(1, total_rows // 10)
    for i, text in enumerate(df_copy[text_col]):
         if (i + 1) % log_interval == 0: logger.info(f" Preprocessing text {i+1}/{total_rows}...")
         processed = preprocess_text(text, lemmatizer, stop_words, sym_spell=sym_spell_global, use_symspell=use_symspell)
         processed_texts.append(processed)
    df_copy['processed_text'] = processed_texts
    original_len = len(df_copy); df_copy = df_copy[df_copy['processed_text'].str.strip().astype(bool)]
    if len(df_copy) < original_len: logger.warning(f"Dropped {original_len - len(df_copy)} rows due to empty processed text.")
    logger.info(f"Text preprocessing complete. Shape: {df_copy.shape}")
    return df_copy

# === Fear Scoring using MacBERTh (MODIFIED FOR MULTIPLE SENTIMENTS) ===
class MacBERThSentimentScorer:
    _instance = None
    _model_name = MACBERTH_MODEL_NAME
    # Define the word lists and corresponding score names HERE
    _word_lists = {
        'hardship_sentiment': HARDSHIP_WORDS
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logger.info(f"Creating MacBERThSentimentScorer instance...")
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        if self._initialized:
            return
        logger.info(f"Initializing MacBERTh model for embedding: {self._model_name}...")
        self.device = device if device else DEVICE
        logger.info(f"MacBERTh on device: {self.device}")
        self.reference_vectors = {} # Dictionary to hold reference vectors

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self.model = AutoModel.from_pretrained(self._model_name).to(self.device)
            self.model.eval()

            logger.info("Calculating average reference vectors for each sentiment type...")
            with torch.no_grad():
                # Iterate through the specified word lists
                for score_name, word_list in self._word_lists.items():
                    if not word_list:
                        logger.warning(f"Word list for '{score_name}' is empty. Skipping.")
                        self.reference_vectors[score_name] = None
                        continue
                    valid_word_list = [str(w) for w in word_list if isinstance(w, str) and w]
                    if not valid_word_list:
                         logger.warning(f"Valid word list for '{score_name}' is empty after filtering. Skipping.")
                         self.reference_vectors[score_name] = None
                         continue

                    inputs = self.tokenizer(valid_word_list, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
                    outputs = self.model(**inputs)
                    embeddings = self._mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()
                    avg_vector = np.mean(embeddings, axis=0).reshape(1, -1)
                    self.reference_vectors[score_name] = avg_vector
                    logger.info(f" - Calculated reference vector for '{score_name}' (shape: {avg_vector.shape}).")

            self._initialized = True
            logger.info("MacBERThSentimentScorer initialization complete.")
        except Exception as e:
            logger.error(f"Failed to initialize MacBERTh scorer: {e}", exc_info=True)
            raise

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def calculate_sentiment_scores(self, texts: List[str], batch_size: int = 32) -> Dict[str, List[float]]:
        """Calculates multiple sentiment scores for a list of texts."""
        if not self._initialized: raise RuntimeError("Scorer not initialized.")
        if not texts: return {score_name: [] for score_name in self.reference_vectors}

        num_texts = len(texts)
        all_scores = {score_name: [] for score_name in self.reference_vectors if self.reference_vectors.get(score_name) is not None} # Use .get() for safety
        valid_score_names = list(all_scores.keys())
        if not valid_score_names:
            logger.warning("No valid reference vectors calculated. Returning empty scores dict.")
            return {}

        logger.info(f"Calculating {len(valid_score_names)} types of sentiment scores for {num_texts} texts...")
        log_interval = max(1, (num_texts // batch_size) // 10) if batch_size > 0 else num_texts # Avoid division by zero

        # Wrap the range with tqdm for a progress bar
        for i in tqdm(range(0, num_texts, batch_size), desc="Calculating Sentiment Scores"): # <<<--- WRAP HERE
            batch_texts = texts[i : i + batch_size]
            valid_batch_texts = [str(t) if t else "" for t in batch_texts]
            try:
                inputs = self.tokenizer(valid_batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()
                for score_name in valid_score_names:
                    ref_vector = self.reference_vectors[score_name]
                    similarities = cosine_similarity(embeddings, ref_vector)
                    all_scores[score_name].extend(similarities.flatten().tolist())
            except Exception as e:
                 logger.error(f"Error processing batch starting at index {i}: {e}. Appending zeros.")
                 batch_len = len(valid_batch_texts)
                 for score_name in valid_score_names:
                     all_scores[score_name].extend([0.0] * batch_len)

        for score_name in valid_score_names:
            if len(all_scores[score_name]) != num_texts:
                logger.error(f"Score length mismatch '{score_name}'. Padding."); all_scores[score_name].extend([0.0] * (num_texts - len(all_scores[score_name])))
        logger.info("Sentiment score calculation complete.")
        return all_scores

@memory.cache
def calculate_sentiment_scores_dataframe(df: pd.DataFrame, text_col: str = "text", batch_size: int = 32) -> pd.DataFrame:
    """Calculates multiple MacBERTh sentiment scores for the text column."""
    logger.info(f"Calculating multiple MacBERTh sentiment scores for '{text_col}'...")
    if text_col not in df.columns: raise ValueError(f"Column '{text_col}' not found.")

    df_copy = df.copy()
    # Ensure text column is string and handle NaNs
    df_copy[text_col] = df_copy[text_col].astype(str).fillna('')

    scorer = MacBERThSentimentScorer() # Initialize the multi-score scorer
    texts_to_score = df_copy[text_col].tolist()

    # Get the dictionary of scores {score_name: [list_of_scores]}
    sentiment_scores_dict = scorer.calculate_sentiment_scores(texts_to_score, batch_size=batch_size)

    # Add each score list as a new column to the DataFrame
    score_cols_added = []
    for score_name, scores_list in sentiment_scores_dict.items():
        if scores_list: # Only add if scores were calculated
             # Ensure score name doesn't clash with existing columns if df_copy is reused
             if score_name in df_copy.columns:
                 logger.warning(f"Column '{score_name}' already exists. Overwriting.")
             df_copy[score_name] = scores_list
             score_cols_added.append(score_name) # Keep track of added columns
             logger.info(f"Added sentiment scores for '{score_name}'. Stats: Min={np.min(scores_list):.3f}, Max={np.max(scores_list):.3f}, Mean={np.mean(scores_list):.3f}, Std={np.std(scores_list):.3f}")
        else:
             logger.warning(f"No scores calculated for '{score_name}', column not added.")

    logger.info("Multiple sentiment scoring complete.")
    # Return relevant identifier columns along with the new scores
    # Ensure identifiers exist in the input df
    id_cols = ['week_date', 'doc_id', 'trial_id']
    valid_id_cols = [col for col in id_cols if col in df_copy.columns]
    final_cols = valid_id_cols + score_cols_added
    return df_copy[final_cols]

@memory.cache
def load_and_aggregate_weekly_mortality(file_path: str = COUNTS_FILE, disease_types: Optional[List[str]] = INFECTIOUS_DISEASE_TYPES, start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    """
    (Cached) Loads mortality counts, aggregates to WEEKLY totals (start of week - Monday),
    filtering by year range and optionally by disease type. Excludes 'christened'.
    Returns DataFrame with ['week_date', 'year', 'week_of_year', 'deaths'].
    """
    if not os.path.exists(file_path): raise FileNotFoundError(f"Mortality file NF: {file_path}")
    logger.info(f"Loading mortality data from {file_path}...")
    df = pd.read_csv(file_path, delimiter="|", low_memory=False, dtype={'weekID': str})
    required_cols = ["weekID", "counttype", "countn"]
    if not all(c in df.columns for c in required_cols): raise ValueError("Mortality file missing required columns")
    df["countn"] = pd.to_numeric(df["countn"], errors="coerce")
    original_len = len(df); df.dropna(subset=["countn"], inplace=True); df["countn"] = df["countn"].astype(int)
    if len(df) < original_len: logger.warning(f"Dropped {original_len - len(df)} rows non-numeric 'countn'.")

    logger.info("Parsing week IDs to week start dates (Monday)...")
    df["week_date_obj"] = df["weekID"].astype(str).apply(parse_bill_weekID_to_weekly) # Store results temporarily

    # --- Filter rows with valid dates AND within rough year range FIRST ---
    original_len = len(df)
    # Also filter out rows where week_date_obj is None (parsing failed)
    df = df.dropna(subset=["week_date_obj"]).copy()

    # *** ADD PRE-FILTER BASED ON YEAR ***
    # This requires extracting year from week_date_obj BEFORE full conversion
    # This is safe because parse_bill_weekID_to_weekly returns datetime objects
    try:
        df['temp_year'] = df['week_date_obj'].apply(lambda x: x.isocalendar().year if pd.notnull(x) else None)
        df.dropna(subset=['temp_year'], inplace=True) # Drop if year extraction failed
        df['temp_year'] = df['temp_year'].astype(int)
        df = df[(df['temp_year'] >= start_year) & (df['temp_year'] <= end_year)].copy()
        df = df.drop(columns=['temp_year']) # Remove temporary column
    except Exception as e:
         logger.error(f"Error during pre-filtering by year: {e}. Proceeding without pre-filter, might still fail.")

    dropped_rows = original_len - len(df)
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with invalid weekID/date or outside year range {start_year}-{end_year}.")

    if df.empty:
         logger.warning("No rows remaining after initial date parsing and year filtering.")
         return pd.DataFrame(columns=["week_date", "year", "week_of_year", "deaths"])

    # --- NOW convert the (already filtered) column to datetime ---
    try:
        # This should now succeed as all rows have valid datetime objects within pandas range
        df['week_date'] = pd.to_datetime(df['week_date_obj'])
    except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime as e:
        logger.error(f"OutOfBoundsDatetime error persist AFTER filtering: {e}. Check START_YEAR ({start_year}) config.")
        # Optionally print problematic dates:
        problematic_dates = df.loc[pd.to_datetime(df['week_date_obj'], errors='coerce').isna(), 'week_date_obj']
        logger.error(f"Problematic date objects (first 5): {problematic_dates.head().tolist()}")
        raise e # Re-raise the error as it's unexpected now

    df = df.drop(columns=['week_date_obj']) # We no longer need the temporary column

    # --- Filter by Year Range (Redundant check, but safe) ---
    df["year"] = df["week_date"].dt.isocalendar().year
    # This filter should ideally not remove anything now, but kept as safeguard
    df = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()

    # --- Continue with type filtering and aggregation ---
    df['counttype'] = df['counttype'].str.lower().str.strip()
    df = df[df["counttype"] != "christened"]
    if disease_types: df = df[df["counttype"].isin([d.lower() for d in disease_types])]
    else: logger.info("Using total mortality (excluding 'christened').")
    if df.empty: logger.warning("No records after type filter."); return pd.DataFrame(columns=["week_date", "year", "week_of_year", "deaths"])
    logger.info(f"{len(df)} weekly records after type filter.")

    logger.info("Aggregating counts per WEEK...")
    weekly_sum = df.groupby("week_date")["countn"].sum().reset_index()
    weekly_sum.rename(columns={"countn": "deaths"}, inplace=True)
    weekly_sum["deaths"] = weekly_sum["deaths"].astype(float)
    # Re-calculate year and week from the final aggregated week_date
    weekly_sum["year"] = weekly_sum["week_date"].dt.isocalendar().year
    weekly_sum["week_of_year"] = weekly_sum["week_date"].dt.isocalendar().week
    weekly_sum = weekly_sum.sort_values("week_date").reset_index(drop=True)
    logger.info(f"Mortality aggregated: {weekly_sum.shape[0]} weeks. Date Range: {weekly_sum['week_date'].min():%Y-%m-%d} to {weekly_sum['week_date'].max():%Y-%m-%d}")
    return weekly_sum[["week_date", "year", "week_of_year", "deaths"]]

@memory.cache
def aggregate_weekly_combined_metrics(
    structured_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    weekly_mortality_df: pd.DataFrame,
    lags_to_create: Dict[str, List[int]]
    ) -> pd.DataFrame:
    """
    (REVISED V3 - ARTEFACT HANDLING) Aggregates weekly structured trial metrics AND the
    single trial-level hardship sentiment score, merges with mortality data,
    log-transforms deaths, creates specified lagged features based on UNSMOOTHED data,
    clips ends, standardizes relevant columns, AND ADDS YEAR-END SPIKE FEATURE.
    Returns a weekly DataFrame ready for analysis/TFT.
    """
    logger.info(f"REVISED V3 (ARTEFACT HANDLING): Aggregating weekly combined structure & hardship...")
    # --- Input Checks (keep as is) ---
    required_structured_cols = ['week_date', 'trial_id', 'offence_cat', 'verdict_cat', 'punishment_cat', 'text']
    expected_sentiment_cols = ['week_date', 'trial_id', 'hardship_sentiment']
    required_mortality_cols = ['week_date', 'year', 'week_of_year', 'deaths']

    if not all(c in structured_df.columns for c in required_structured_cols): raise ValueError(f"structured_df missing cols.")
    if not all(c in sentiment_df.columns for c in expected_sentiment_cols):
         if sentiment_df.empty: logger.warning("Sentiment DF empty.")
         else: raise ValueError(f"sentiment_df missing cols.")
    if not all(c in weekly_mortality_df.columns for c in required_mortality_cols): raise ValueError("mortality_df missing cols.")

    # --- Convert Dates (keep as is) ---
    structured_df['week_date'] = pd.to_datetime(structured_df['week_date'])
    sentiment_df['week_date'] = pd.to_datetime(sentiment_df['week_date'])
    weekly_mortality_df['week_date'] = pd.to_datetime(weekly_mortality_df['week_date'])

    # --- Merge trial data with sentiment (keep as is) ---
    logger.info("Merging structured trial data with hardship sentiment scores...")
    trial_df_merged = pd.merge(
        structured_df[['trial_id', 'week_date', 'offence_cat', 'verdict_cat', 'punishment_cat', 'text']],
        sentiment_df[['trial_id', 'hardship_sentiment']],
        on='trial_id', how='left'
    )
    trial_df_merged['hardship_sentiment'] = pd.to_numeric(trial_df_merged['hardship_sentiment'], errors='coerce')

    # --- Filter for Sentiment Aggregation (keep as is) ---
    trial_df_merged['word_count'] = trial_df_merged['text'].apply(lambda x: len(str(x).split()))
    MIN_TEXT_LENGTH_FOR_SENTIMENT = 30
    valid_sentiment_mask = (trial_df_merged['word_count'] > MIN_TEXT_LENGTH_FOR_SENTIMENT) & trial_df_merged['hardship_sentiment'].notna()
    logger.info(f"Aggregating weekly hardship sentiment from {valid_sentiment_mask.sum()} valid trials.")

    # --- Aggregate Weekly Metrics (keep as is) ---
    grouped_week = trial_df_merged.groupby('week_date')
    VIOLENT_CATS = ['violenttheft', 'kill', 'sexual']; PROPERTY_CATS = ['theft', 'deception', 'damage', 'royaloffences']
    GUILTY_VERDICTS = ['guilty']; NOT_GUILTY_VERDICTS = ['notguilty', 'unknown']
    DEATH_PUNISH = ['death']; TRANSPORT_PUNISH = ['transport']; CORPORAL_PUNISH = ['corporal', 'miscpunish']
    MIN_TRIALS_PER_WEEK = 3

    weekly_metrics = grouped_week.agg(
        total_trials = ('trial_id', 'nunique'),
        violent_trials = ('offence_cat', lambda x: x.isin(VIOLENT_CATS).sum()),
        property_trials = ('offence_cat', lambda x: x.isin(PROPERTY_CATS).sum()),
        guilty_verdicts = ('verdict_cat', lambda x: x.isin(GUILTY_VERDICTS).sum()),
        not_guilty_verdicts = ('verdict_cat', lambda x: x.isin(NOT_GUILTY_VERDICTS).sum()),
        death_sentences = ('punishment_cat', lambda x: x.isin(DEATH_PUNISH).sum()),
        transport_sentences = ('punishment_cat', lambda x: x.isin(TRANSPORT_PUNISH).sum()),
        corporal_sentences = ('punishment_cat', lambda x: x.isin(CORPORAL_PUNISH).sum()),
        hardship_sentiment = ('hardship_sentiment', lambda x: x[valid_sentiment_mask.loc[x.index]].mean() if valid_sentiment_mask.loc[x.index].any() else np.nan)
    ).reset_index()

    # Punishment Score (keep as is)
    def calculate_punishment_score(row):
        if row['punishment_cat'] in DEATH_PUNISH: return 5
        if row['punishment_cat'] in TRANSPORT_PUNISH: return 4
        if row['punishment_cat'] in CORPORAL_PUNISH: return 3
        return 0
    trial_df_merged['punish_score'] = trial_df_merged.apply(calculate_punishment_score, axis=1)
    convicted_df = trial_df_merged[trial_df_merged['verdict_cat'].isin(GUILTY_VERDICTS)]
    weekly_avg_punish_score = convicted_df.groupby('week_date')['punish_score'].mean().reset_index().rename(columns={'punish_score': 'avg_punishment_score'})
    weekly_metrics = pd.merge(weekly_metrics, weekly_avg_punish_score, on='week_date', how='left')

    # Calculate Proportions (keep as is)
    metric_cols_calc = ['violent_crime_prop', 'property_crime_prop', 'conviction_rate',
                        'death_sentence_rate', 'transport_rate', 'avg_punishment_score']
    for col in metric_cols_calc: weekly_metrics[col] = np.nan
    valid_week_mask = weekly_metrics['total_trials'] >= MIN_TRIALS_PER_WEEK
    logger.info(f"Calculating rates/props for {valid_week_mask.sum()} weeks with >= {MIN_TRIALS_PER_WEEK} trials.")
    total_trials_denom_valid = weekly_metrics.loc[valid_week_mask, 'total_trials'].replace(0, np.nan)
    guilty_denom_valid = weekly_metrics.loc[valid_week_mask, 'guilty_verdicts'].replace(0, np.nan)
    valid_verdicts_denom_valid = (weekly_metrics.loc[valid_week_mask, 'guilty_verdicts'] + weekly_metrics.loc[valid_week_mask, 'not_guilty_verdicts']).replace(0, np.nan)

    weekly_metrics.loc[valid_week_mask, 'violent_crime_prop'] = weekly_metrics.loc[valid_week_mask, 'violent_trials'] / total_trials_denom_valid
    weekly_metrics.loc[valid_week_mask, 'property_crime_prop'] = weekly_metrics.loc[valid_week_mask, 'property_trials'] / total_trials_denom_valid
    weekly_metrics.loc[valid_week_mask, 'conviction_rate'] = weekly_metrics.loc[valid_week_mask, 'guilty_verdicts'] / valid_verdicts_denom_valid
    weekly_metrics.loc[valid_week_mask, 'death_sentence_rate'] = weekly_metrics.loc[valid_week_mask, 'death_sentences'] / guilty_denom_valid
    weekly_metrics.loc[valid_week_mask, 'transport_rate'] = weekly_metrics.loc[valid_week_mask, 'transport_sentences'] / guilty_denom_valid
    weekly_metrics['avg_punishment_score'].fillna(0, inplace=True)

    # Impute NaNs (keep as is)
    base_metric_cols_impute = (metric_cols_calc + ['hardship_sentiment'])
    logger.info("Imputing NaNs in calculated/aggregated base metrics...")
    for col in base_metric_cols_impute:
        if col in weekly_metrics.columns and weekly_metrics[col].isnull().any():
             col_median = weekly_metrics[col].median()
             weekly_metrics[col] = weekly_metrics[col].interpolate(method='linear').fillna(col_median).fillna(0)

    # --- Merge Base Metrics with Mortality (keep as is) ---
    logger.info(f"Merging weekly base metrics with weekly mortality data...")
    cols_to_merge = ['week_date', 'total_trials'] + base_metric_cols_impute
    cols_to_merge = [col for col in cols_to_merge if col in weekly_metrics.columns]

    merged_df = pd.merge(weekly_mortality_df, weekly_metrics[cols_to_merge], on='week_date', how='outer')
    merged_df = merged_df.sort_values("week_date").reset_index(drop=True)

    # Impute missing base metrics post-merge (keep as is)
    all_metric_cols_post_merge = ['total_trials'] + base_metric_cols_impute
    for col in all_metric_cols_post_merge:
        if col in merged_df.columns and merged_df[col].isnull().any():
            missing_count = merged_df[col].isnull().sum()
            logger.warning(f"{missing_count} weeks missing '{col}' post-merge. Imputing with interpolation then 0.")
            merged_df[col] = merged_df[col].interpolate(method='linear').fillna(0)

    # --- Log Transform Deaths (and smooth) ---
    if 'deaths' not in merged_df.columns:
        raise ValueError("'deaths' column missing before log transformation step.")
    merged_df['deaths'] = pd.to_numeric(merged_df['deaths'], errors='coerce').fillna(0)

    if APPLY_TARGET_SMOOTHING_EXPERIMENT and TARGET_SMOOTHING_WINDOW > 1:
        logger.warning(f"EXPERIMENTAL: Applying {TARGET_SMOOTHING_WINDOW}-week rolling "
                       f"{TARGET_SMOOTHING_TYPE} to 'deaths' before log transformation.")
        
        # For a true forecasting target, a trailing window is appropriate.
        # center=False makes it a trailing window.
        # min_periods=1 ensures output even at the start of the series.
        if TARGET_SMOOTHING_TYPE == 'median':
            smoothed_deaths = merged_df['deaths'].rolling(
                window=TARGET_SMOOTHING_WINDOW,
                min_periods=1,
                center=False # Trailing median
            ).median()
        elif TARGET_SMOOTHING_TYPE == 'mean':
            smoothed_deaths = merged_df['deaths'].rolling(
                window=TARGET_SMOOTHING_WINDOW,
                min_periods=1,
                center=False # Trailing mean
            ).mean()
        else:
            logger.error(f"Unknown TARGET_SMOOTHING_TYPE: {TARGET_SMOOTHING_TYPE}. Not smoothing.")
            smoothed_deaths = merged_df['deaths'] # No change

        # It's good practice to have a new column for the smoothed deaths if you want to compare
        merged_df['deaths_smoothed_target_exp'] = smoothed_deaths.fillna(0) # Fill NaNs from rolling window start
        merged_df['log_deaths'] = np.log1p(merged_df['deaths_smoothed_target_exp'])
        logger.info("Applied log1p transformation to 'deaths_smoothed_target_exp' -> 'log_deaths'.")
        # Keep original 'deaths' for reference or other analyses if needed
    elif APPLY_TARGET_SMOOTHING_EXPERIMENT: # Window was likely <=1
        logger.info("Target smoothing window <= 1, applying log1p to original 'deaths'.")
        merged_df['log_deaths'] = np.log1p(merged_df['deaths'].fillna(0)) # Original behavior
        logger.info("Applied log1p transformation to original 'deaths' column -> 'log_deaths'.")
    else:
        merged_df['log_deaths'] = np.log1p(merged_df['deaths'].fillna(0))
        logger.info("Applied log1p transformation to 'deaths' column -> 'log_deaths'.")

    # --- Create Specified Lagged Features (keep as is) ---
    lagged_col_names = []
    for base_col, week_lags in lags_to_create.items():
        if base_col in merged_df.columns:
            base_col_median = merged_df[base_col].median()
            for lag in week_lags:
                feature_lag_col_name = f'{base_col}_lag{lag}w'
                lagged_col_names.append(feature_lag_col_name)
                logger.info(f"Creating lagged feature: '{feature_lag_col_name}' ({lag} weeks)...")
                merged_df[feature_lag_col_name] = merged_df[base_col].shift(lag)
                merged_df[feature_lag_col_name] = merged_df[feature_lag_col_name].interpolate(method='linear').fillna(base_col_median).fillna(0)
        else: logger.warning(f"Base column '{base_col}' not found for lagging.")

    # --- Add 'year' and 'week_of_year' for artefact feature (EARLIER than before) ---
    merged_df['year'] = merged_df['week_date'].dt.isocalendar().year.fillna(0).astype(int) # Keep as int for comparison
    merged_df['week_of_year'] = merged_df['week_date'].dt.isocalendar().week.fillna(0).astype(int) # Keep as int

    # +++ NEW: Feature Engineering for Year-End Spikes +++
    # Define weeks that might indicate a reporting spike.
    # Weeks 52, 53 are common. Week 1 could catch delays from the previous year's end.
    SPIKE_WEEKS = [52, 53, 1] # Can adjust this list
    merged_df['is_year_end_spike'] = merged_df['week_of_year'].isin(SPIKE_WEEKS).astype(float) # Ensure float for TFT
    logger.info(f"Created 'is_year_end_spike' feature for weeks: {SPIKE_WEEKS}. "
                f"Count of spike weeks: {merged_df['is_year_end_spike'].sum()}")
    # +++ END NEW FEATURE +++

    # --- Standardize Features ---
    # Standardize base aggregated metrics AND their lagged versions
    features_to_standardize = base_metric_cols_impute + lagged_col_names
    if 'total_trials' in merged_df.columns: features_to_standardize.append('total_trials')
    # DO NOT standardize 'is_year_end_spike' as it's already 0/1 and meant as a flag.
    features_to_standardize = list(set(features_to_standardize))

    standardized_col_names_map = {}
    logger.info(f"Attempting to standardize: {features_to_standardize}")

    for col in features_to_standardize:
        if col in merged_df.columns:
             std_col_name = f'{col}_std'
             standardized_col_names_map[col] = std_col_name
             logger.info(f"Standardizing '{col}' -> '{std_col_name}'.")
             scaler = StandardScaler()
             temp_col_data = merged_df[[col]].copy()
             if temp_col_data.isnull().any().any() or np.isinf(temp_col_data).any().any():
                 col_median = temp_col_data[col].median()
                 temp_col_data[col] = temp_col_data[col].replace([np.inf, -np.inf], np.nan).fillna(col_median).fillna(0)
             try:
                 merged_df[std_col_name] = scaler.fit_transform(temp_col_data)
                 merged_df[std_col_name] = merged_df[std_col_name].astype(float)
             except Exception as e: logger.error(f"StandardScaler failed for '{col}': {e}. Skipping.");
        else: logger.warning(f"Column '{col}' not found for standardization.")

    # --- Prepare Final DataFrame (keep as is regarding other features) ---
    merged_df = merged_df.sort_values("week_date").reset_index(drop=True)
    merged_df["time_idx"] = (merged_df["week_date"] - merged_df["week_date"].min()).dt.days // 7

    # Convert year and week_of_year to string AFTER creating 'is_year_end_spike'
    merged_df['year'] = merged_df['year'].astype(str)
    merged_df['week_of_year'] = merged_df['week_of_year'].astype(str)
    merged_df["series_id"] = "London"

    # Ensure numeric dtypes
    numeric_cols_final = (['time_idx', 'deaths', 'log_deaths', 'total_trials', 'is_year_end_spike'] + # ADDED 'is_year_end_spike'
                           base_metric_cols_impute +
                           lagged_col_names +
                           list(standardized_col_names_map.values()))
    for col in numeric_cols_final:
        if col in merged_df.columns:
            if col == "time_idx":
                merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").astype(int)
            else:
                merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").astype(float)

    # Select final columns
    final_cols = (
        ["week_date", "time_idx", "deaths", "log_deaths", "is_year_end_spike", # ADDED 'is_year_end_spike'
         "year", "week_of_year", "series_id", "total_trials"] +
        base_metric_cols_impute +
        lagged_col_names +
        list(standardized_col_names_map.values())
    )
    final_cols_unique_exist = []
    for col in final_cols:
        if col in merged_df.columns and col not in final_cols_unique_exist: final_cols_unique_exist.append(col)
    merged_df = merged_df[final_cols_unique_exist].copy()

    # --- Final NaN Drop & Clipping (keep as is, but consider 'is_year_end_spike' in critical) ---
    logger.info(f"Shape before final NaN drop and clipping: {merged_df.shape}")
    critical_cols_for_model = ['time_idx', 'log_deaths', 'is_year_end_spike'] + list(standardized_col_names_map.values()) # ADDED
    critical_cols_for_model_exist = [col for col in critical_cols_for_model if col in merged_df.columns]
    logger.info(f"Dropping NaNs based on columns: {critical_cols_for_model_exist}")
    original_rows_before_dropna = len(merged_df)
    merged_df.dropna(subset=critical_cols_for_model_exist, inplace=True)
    rows_dropped_nan = original_rows_before_dropna - len(merged_df)
    if rows_dropped_nan > 0: logger.info(f"Dropped {rows_dropped_nan} rows due to NaNs in critical columns.")
    logger.info(f"Shape after final NaN drop: {merged_df.shape}")

    if merged_df.empty:
        logger.error("DataFrame empty after final NaN drop. Check data processing steps.")
        return pd.DataFrame(columns=final_cols_unique_exist)

    # Clip Ends (keep as is)
    weeks_to_clip = 52
    if not merged_df.empty:
        min_idx_df = merged_df['time_idx'].min(); max_idx_df = merged_df['time_idx'].max()
        original_rows_before_clip = len(merged_df)
        if max_idx_df >= min_idx_df and (max_idx_df - min_idx_df + 1) > 2 * weeks_to_clip :
            merged_df = merged_df[(merged_df['time_idx'] >= min_idx_df + weeks_to_clip) &
                                  (merged_df['time_idx'] <= max_idx_df - weeks_to_clip)].copy()
            rows_clipped = original_rows_before_clip - len(merged_df)
            if rows_clipped > 0: logger.info(f"Clipped ends: Removed {rows_clipped} rows. New shape: {merged_df.shape}.")
        else: logger.warning(f"Not enough data span ({max_idx_df - min_idx_df + 1 if max_idx_df >= min_idx_df else 0} weeks) to clip {weeks_to_clip} weeks. Skipping clipping.")

        if merged_df.empty: logger.error("DataFrame empty after clipping ends."); return pd.DataFrame(columns=final_cols_unique_exist)
        if merged_df["time_idx"].max() - merged_df["time_idx"].min() + 1 < WEEKLY_MAX_ENCODER_LENGTH + WEEKLY_MAX_PREDICTION_LENGTH:
             logger.error(f"Insufficient weekly data span ({merged_df.shape[0]} weeks) for TFT config after clipping."); return pd.DataFrame(columns=final_cols_unique_exist)
    else: logger.error("DataFrame empty before clipping."); return pd.DataFrame(columns=final_cols_unique_exist)

    logger.info(f"Final weekly combined metrics data shape returned: {merged_df.shape}. Time idx range: {merged_df['time_idx'].min()}-{merged_df['time_idx'].max()}")
    logger.info(f"Final Columns Returned: {merged_df.columns.tolist()}")
    nan_check_after = merged_df.isnull().sum()
    if nan_check_after.any(): logger.warning(f"NaNs still present:\n{nan_check_after[nan_check_after > 0]}")

    return merged_df

# -----------------------------------------------------------------------------
# 5. TFT Training and Evaluation (WEEKLY)
# -----------------------------------------------------------------------------


def train_tft_model(df: pd.DataFrame,
                    time_varying_reals_cols: List[str],
                    run_name: str, # <<< ADDED run_name for logging/checkpoints
                    # --- Keep other parameters ---
                    max_epochs: int = MAX_EPOCHS, # <<< Increased default max_epochs
                    batch_size: int = BATCH_SIZE,
                    encoder_length: int = WEEKLY_MAX_ENCODER_LENGTH,
                    pred_length: int = WEEKLY_MAX_PREDICTION_LENGTH,
                    min_val_windows: int = 4,  # <— NEW: leave at least 4 decoder windows for validation
                    lr: float = LEARNING_RATE,
                    hidden_size: int = HIDDEN_SIZE, # Keep complexity same for now
                    attn_heads: int = ATTENTION_HEAD_SIZE, # Keep complexity same for now
                    dropout: float = DROPOUT,
                    hidden_cont_size: int = HIDDEN_CONTINUOUS_SIZE, # Keep complexity same for now
                    clip_val: float = GRADIENT_CLIP_VAL) -> Tuple[Optional[TemporalFusionTransformer], Optional[pl.Trainer], Optional[torch.utils.data.DataLoader], Optional[TimeSeriesDataSet]]:
    """
    Trains the Temporal Fusion Transformer model on WEEKLY data using log-transformed deaths
    and dynamically specified real-valued features. Logs under a specific run name.
                    min_val_windows: minimum number of prediction windows to keep for validation (default 4).
    """

    logger.info(f"--- Starting TFT Training for Run: '{run_name}' ---")
    logger.info(f"Setting up WEEKLY TFT model training (Target: log_deaths)...")
    logger.info(f" Using real features: {time_varying_reals_cols}")
    logger.info(f" Encoder length: {encoder_length} weeks, Prediction length: {pred_length} weeks")

    # --- Data Cutoff ---
    max_idx = df["time_idx"].max()
    training_cutoff = max_idx - (pred_length * max(1, min_val_windows))
    min_idx = df["time_idx"].min()
    logger.info(f"Weekly Data Cutoff for Training: time_idx ≤ {training_cutoff} (Range: {min_idx}-{max_idx}, leaving ≥{min_val_windows} windows for validation)")
    if training_cutoff < min_idx + encoder_length -1:
        logger.error(f"Training cutoff {training_cutoff} doesn't allow full encoder length {encoder_length} from start {min_idx}.")
        return None, None, None, None
    
    df['week_sin'] = np.sin(2*np.pi*df.week_of_year.astype(int)/52)
    df['week_cos'] = np.cos(2*np.pi*df.week_of_year.astype(int)/52)

    # --- Dtype Check ---
    logger.info("Ensuring correct dtypes before TimeSeriesDataSet...")
    try:
        data_for_tft = df.copy()
        required_numeric_cols = ["time_idx", "log_deaths", "deaths"] + time_varying_reals_cols
        required_numeric_cols = list(set(required_numeric_cols))
        for col in required_numeric_cols:
            if col not in data_for_tft.columns: raise ValueError(f"Column '{col}' not found.")
            data_for_tft[col] = pd.to_numeric(data_for_tft[col], errors='coerce')
        # --- Ensure time_idx is integer for TimeSeriesDataSet ---
        if "time_idx" in data_for_tft.columns:
            data_for_tft["time_idx"] = data_for_tft["time_idx"].astype(int)
        categorical_cols = ["series_id", "week_of_year", "year"]
        for col in categorical_cols: data_for_tft[col] = data_for_tft[col].astype(str)
        numeric_cols_to_impute = [col for col in required_numeric_cols if col != 'time_idx']
        if data_for_tft[numeric_cols_to_impute].isnull().any().any():
             nan_counts = data_for_tft[numeric_cols_to_impute].isnull().sum()
             logger.warning(f"NaNs found after casting:\n{nan_counts[nan_counts > 0]}. Imputing with median.")
             for col in numeric_cols_to_impute: data_for_tft[col].fillna(data_for_tft[col].median(), inplace=True)
        logger.info("Dtype check passed.")
    except Exception as e: logger.error(f"Error during dtype check: {e}", exc_info=True); return None, None, None, None

    # --- TimeSeriesDataSet Setup ---
    logger.info("Setting up WEEKLY TimeSeriesDataSet for TFT (Target: log_deaths)...")
    try:
        unknown_reals_for_tft = [col for col in time_varying_reals_cols if col != "log_deaths"]
        logger.info(f"Passing to TimeSeriesDataSet time_varying_unknown_reals: {unknown_reals_for_tft}")
        missing_tft_cols = [col for col in unknown_reals_for_tft if col not in data_for_tft.columns]
        if missing_tft_cols: raise ValueError(f"Columns for TFT `time_varying_unknown_reals` are missing: {missing_tft_cols}")

        training_dataset = TimeSeriesDataSet(
            data_for_tft[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx", target="log_deaths", group_ids=["series_id"],
            max_encoder_length=encoder_length, max_prediction_length=pred_length,
            static_categoricals=[],
            time_varying_known_reals=['time_idx', 'week_sin', 'week_cos', 'is_year_end_spike'], # <--- ADDED HERE
            time_varying_known_categoricals=[],
            time_varying_unknown_categoricals=["year"], # 'week_of_year' can also be here if you use it as categorical
            time_varying_unknown_reals=unknown_reals_for_tft,
            add_target_scales=False, add_encoder_length=True, allow_missing_timesteps=True,
            # Make sure NaNLabelEncoder is appropriate for year/week_of_year if they can have missing values
            categorical_encoders={"year": NaNLabelEncoder(add_nan=True),
                                  "week_of_year": NaNLabelEncoder(add_nan=True)}
        )
        # --- build a sliding‑window validation set, not a single forecast start ---
        val_start_idx = max(min_idx, training_cutoff - encoder_length + 1)
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            data_for_tft[lambda x: x.time_idx >= val_start_idx],   # keep encoder context
            predict=False,                                         # sliding windows
            stop_randomization=True
        )

        effective_batch_size = max(1, min(batch_size, len(training_dataset) // 2 if len(training_dataset) > 1 else 1))
        val_batch_size = max(1, min(effective_batch_size * 2, len(validation_dataset)))
        logger.info(f"Using effective train batch size: {effective_batch_size}, val batch size: {val_batch_size}")

        train_dataloader = training_dataset.to_dataloader(train=True, batch_size=effective_batch_size, num_workers=0, persistent_workers=False)
        # Ensure shuffle=False for validation loader
        val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=val_batch_size, num_workers=0, persistent_workers=False, shuffle=False)

        if len(train_dataloader) == 0 or len(val_dataloader) == 0: logger.error("Empty dataloader(s)."); return None, None, None, None
    except Exception as e: logger.error(f"Error creating TimeSeriesDataSet/Dataloaders: {e}", exc_info=True); logger.error(f"Data info:\n{data_for_tft.info()}"); return None, None, None, None

    logger.info("Configuring TemporalFusionTransformer model...")
    loss_metric = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    try:
        # --- instantiate TFT with AdamW + plateau LR scheduler ---
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=LEARNING_RATE,
            hidden_size=HIDDEN_SIZE,
            attention_head_size=ATTENTION_HEAD_SIZE,
            dropout=DROPOUT,
            optimizer="adamw",
            loss=QuantileLoss([0.1,0.5,0.9]),
            reduce_on_plateau_patience=8,         # wait 7 epochs
            reduce_on_plateau_reduction=1.3,      # /1.3 the LR each time
            reduce_on_plateau_min_lr=6e-7         # do not go below 6e-7
        )
        logger.info(f"TFT model parameters: {tft.size()/1e6:.1f} million")
    except Exception as e: logger.error(f"Error initializing TFT: {e}", exc_info=True); return None, None, val_dataloader, validation_dataset

    # --- Use run_name for logger ---
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=20, verbose=True, mode="min") # <<< Increased patience
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    accelerator, devices = ('cpu', 1)
    logger.info(f"Configuring Trainer (Accelerator: {accelerator}, Devices: {devices})...")
    from pytorch_lightning.loggers import TensorBoardLogger
    # Use run_name to create distinct log directories
    tb_logger = TensorBoardLogger(save_dir="lightning_logs/", name=f"tft_{run_name}_weekly_log_target")
    csv_logger = CSVLogger(save_dir=PLOT_DIR, name="tft_logs")
    trainer = pl.Trainer(
        max_epochs=max_epochs, # Use increased max_epochs from args
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=clip_val,
        callbacks=[lr_monitor, early_stop_callback],
        logger=[tb_logger, csv_logger],
        enable_progress_bar=True
    )

    logger.info(f"Starting TFT model training for run '{run_name}'...")
    start_train_time = time.time()
    best_tft = None # Initialize best_tft
    try:
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        # ---- Plot training & validation loss curves ----
        try:
            metrics_file = os.path.join(csv_logger.log_dir, "metrics.csv")
            if os.path.exists(metrics_file):
                metrics = pd.read_csv(metrics_file)
                loss_cols = [c for c in metrics.columns if c.endswith("_loss") and "step" not in c]
                if "epoch" in metrics.columns and loss_cols:
                    fig, ax = plt.subplots()
                    (metrics
                        .groupby("epoch")[loss_cols]
                        .mean()
                        .plot(ax=ax))
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title("TFT – training vs validation loss")
                    out_path = os.path.join(PLOT_DIR, "tft_training_curve.png")
                    fig.savefig(out_path, bbox_inches="tight")
                    plt.close(fig)
                    logger.info(f"Saved training-curve plot → {out_path}")
                else:
                    logger.warning("Loss columns not found in metrics.csv – skipping curve plot.")
            else:
                logger.warning("metrics.csv not written – skipping curve plot.")
        except Exception as e:
            logger.warning(f"Could not generate training-curve plot: {e}")
        logger.info(f"TFT training finished for '{run_name}' in {(time.time() - start_train_time)/60:.2f} minutes.")
        best_model_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, "checkpoint_callback") and trainer.checkpoint_callback else None

        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"Loading best model for '{run_name}' from checkpoint: {best_model_path}")
            # Load onto the globally defined DEVICE
            best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path, map_location=DEVICE)
            logger.info(f"Best model loaded successfully for run '{run_name}'.")
            return best_tft, trainer, val_dataloader, validation_dataset
        else:
            logger.warning(f"Best checkpoint not found or invalid path for '{run_name}'. Attempting to return last model state.")
            # Try to get the last model state from the trainer
            last_model = trainer.model if hasattr(trainer, 'model') and trainer.model is not None else tft
            if last_model:
                 last_model.to(DEVICE) # Ensure it's on the correct device
                 logger.info(f"Returning last model state for run '{run_name}'.")
                 return last_model, trainer, val_dataloader, validation_dataset
            else:
                 logger.error(f"Could not retrieve last model state for run '{run_name}'.")
                 return None, trainer, val_dataloader, validation_dataset

    except Exception as e:
        logger.error(f"Error during TFT fitting for run '{run_name}': {e}", exc_info=True)
        # Attempt to return last model state even on error
        last_model = trainer.model if hasattr(trainer, 'model') and trainer.model is not None else tft
        if last_model: last_model.to(DEVICE)
        return last_model, trainer, val_dataloader, validation_dataset

# Place this function definition before evaluate_model and evaluate_against_baselines

def convert_time_idx_to_date_labels(time_indices: pd.Series,
                                    min_time_idx_ref: int,
                                    min_week_date_ref: datetime,
                                    label_type: str = "year_week", # "year_week", "year", "date"
                                    tick_frequency: int = 5 # Show label every N ticks
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a Series of time_idx values to date-based labels for plot ticks.
    """
    if not isinstance(time_indices, pd.Series): # Ensure it's a Series for .unique()
        time_indices = pd.Series(time_indices)

    if time_indices.empty:
        return np.array([]), np.array([])

    unique_sorted_time_idx = np.sort(time_indices.unique())
    
    tick_positions = []
    tick_labels = []

    for i, current_time_idx in enumerate(unique_sorted_time_idx):
        if i % tick_frequency == 0:
            weeks_offset = current_time_idx - min_time_idx_ref
            current_date = min_week_date_ref + timedelta(weeks=int(weeks_offset)) # Ensure weeks_offset is int
            
            tick_positions.append(current_time_idx)
            if label_type == "year_week":
                tick_labels.append(f"{current_date.year}-W{current_date.isocalendar().week:02d}")
            elif label_type == "year":
                tick_labels.append(str(current_date.year))
            elif label_type == "date":
                tick_labels.append(current_date.strftime('%Y-%m-%d'))
            else:
                tick_labels.append(str(current_time_idx))
                
    return np.array(tick_positions), np.array(tick_labels)
# --- evaluate_model function (Correct for Forecasting Task) ---
# Ensure these imports are present at the top of your script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss # Or specific metric classes used
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import os
import torch

# Make sure DEVICE is defined globally (e.g., DEVICE = torch.device("cpu"))
# Make sure logger is defined globally (e.g., logger = logging.getLogger(__name__))

# --- evaluate_model function (Correct for Forecasting Task - Revised Unpacking) ---
# Modify the arguments of evaluate_model:
def evaluate_model(model: TemporalFusionTransformer,
                   dataloader: torch.utils.data.DataLoader,
                   # REMOVE: dataset: TimeSeriesDataSet,
                   val_index_df_with_dates: pd.DataFrame, # ADD: Contains 'time_idx', 'week_date' for the validation set slice
                   min_overall_time_idx: int,             # ADD
                   min_overall_week_date: datetime,       # ADD
                   plot_dir: str, run_name: str) -> Dict[str, float]:
    """
    Evaluates TFT model on log_deaths, returns metrics (MAE, MSE, SMAPE) on original death scale,
    saves plots with run_name prefix, and acknowledges confidence-based metrics.
    Includes revised predict() output handling.
    """
    logger.info(f"Evaluating model performance for run '{run_name}'...")
    # Fallback if the model object lacks max_prediction_length
    if not hasattr(model.hparams, "max_prediction_length"):
        model.hparams.max_prediction_length = WEEKLY_MAX_PREDICTION_LENGTH
    results = {}
    if model is None or dataloader is None or len(dataloader) == 0:
        logger.error("Model/Dataloader missing for evaluation.")
        return {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan}

    try:
        eval_device = next(model.parameters()).device
    except Exception:
        eval_device = torch.device(DEVICE) # Fallback to global DEVICE
        try:
            model.to(eval_device)
        except Exception as device_err:
            logger.error(f"Could not move model to device {eval_device}: {device_err}")
            eval_device = torch.device("cpu") # Force CPU if move fails
            model.to(eval_device)
            logger.warning("Forcing evaluation on CPU.")
    logger.info(f"Evaluation device: {eval_device}")

    actuals_log_list, preds_log_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(iter(dataloader)):
            try:
                x_gpu = {k: v.to(eval_device) for k, v in x.items() if isinstance(v, torch.Tensor)}
                target_log = y[0].to(eval_device)
                preds = model(x_gpu)["prediction"] # Get predictions from model forward pass
                preds_log_list.append(preds.cpu())
                actuals_log_list.append(target_log.cpu())
            except Exception as batch_err:
                 logger.error(f"Error processing evaluation batch {i}: {batch_err}", exc_info=True)
                 continue # Skip problematic batch

    if not preds_log_list:
        logger.error("No predictions collected during evaluation.")
        return {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan}

    # --- Metric Calculation ---
    try:
        actuals_log_all = torch.cat(actuals_log_list).numpy()
        preds_log_all = torch.cat(preds_log_list).numpy()

        actuals_log_flat = actuals_log_all.flatten()
        # Ensure preds_log_all has the expected 3 dimensions (batch, time, quantiles)
        if preds_log_all.ndim != 3 or preds_log_all.shape[2] != 3:
             logger.error(f"Prediction tensor has unexpected shape: {preds_log_all.shape}. Expected (..., 3). Cannot calculate metrics.")
             return {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan}
        preds_log_median_flat = preds_log_all[:, :, 1].flatten() # Use median (p50) - index 1

        min_len_m = min(len(actuals_log_flat), len(preds_log_median_flat))
        if len(actuals_log_flat) != len(preds_log_median_flat):
            logger.warning(f"Metric length mismatch ({len(actuals_log_flat)} vs {len(preds_log_median_flat)}): Truncating.")
            preds_log_median_flat = preds_log_median_flat[:min_len_m]
            actuals_log_flat = actuals_log_flat[:min_len_m]

        # Inverse transform to original scale
        actuals_orig_flat = np.expm1(actuals_log_flat)
        preds_orig_median_flat = np.maximum(0, np.expm1(preds_log_median_flat)) # Ensure non-negative

        val_mae = mean_absolute_error(actuals_orig_flat, preds_orig_median_flat)
        val_mse = mean_squared_error(actuals_orig_flat, preds_orig_median_flat)

        # Calculate SMAPE carefully, avoiding division by zero
        denominator = (np.abs(actuals_orig_flat) + np.abs(preds_orig_median_flat)) / 2.0
        # Handle cases where both actual and prediction are near zero
        smape_mask = denominator > 1e-9 # Use a small threshold instead of exact zero
        val_smape = np.mean(
            np.abs(preds_orig_median_flat[smape_mask] - actuals_orig_flat[smape_mask]) /
            denominator[smape_mask]
            ) * 100 if np.any(smape_mask) else 0.0 # Return 0 if all denominators are zero

        results = {"MAE": val_mae, "MSE": val_mse, "SMAPE": val_smape}
        logger.info(f"[Validation Metrics ({run_name}, Original Scale)] MAE={val_mae:.3f} MSE={val_mse:.3f} SMAPE={val_smape:.3f}%")
        logger.info("Note: While standard forecasting metrics (MAE, MSE, SMAPE) are reported, evaluating nuanced historical sentiment ideally involves confidence-based metrics like cPrecision/cRecall (Yacouby & Axman, 2020), which were beyond the scope of direct implementation for this forecasting task.")

    except Exception as metric_err:
        logger.error(f"Error calculating evaluation metrics: {metric_err}", exc_info=True)
        results = {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan} # Ensure results dict exists

    logger.info(f"Generating weekly evaluation plots for run '{run_name}' (showing original death scale)...")
    plot_fig = None; plot_fig_res = None
    try:
        # --- Simplest predict call for plotting data ---
        # This should return quantiles by default, along with x and index
        logger.info("Calling model.predict() for plotting...")
        predictions = model.predict(
            dataloader,
            return_x=True,
            return_index=True,
            mode="quantiles"      # <-- keep only this extra flag
        )
        logger.info(f"model.predict() output type: {type(predictions)}")
        # --- End Simplification ---

        # --- Direct Unpacking Attempt ---
        if not (isinstance(predictions, (list, tuple)) and len(predictions) == 3):
             # Check if it's a dict (less common direct return)
             if isinstance(predictions, dict) and 'prediction' in predictions and 'x' in predictions and 'index' in predictions:
                 logger.warning("Predict returned a dict, unpacking components.")
                 raw_preds_container = predictions
                 x_output = predictions['x']
                 index_df = predictions['index']
                 # Need to extract the prediction tensor itself if mode was raw/prediction
                 if 'prediction' in raw_preds_container:
                     raw_preds = raw_preds_container['prediction']
                 else:
                     logger.error("Prediction tensor missing in returned dict. Skipping plots.")
                     return results
             else:
                 logger.error(f"Predict output did not return expected tuple/list of 3 or usable dict. Got {type(predictions)}. Skipping plots.")
                 return results # Return metrics calculated earlier
        else:
            # Standard tuple/list unpacking
            raw_preds, x_output, index_df = predictions # Unpack directly

        # Validate components (more detailed checks)
        # Check raw_preds: should be tensor [samples, time, quantiles] for mode=quantiles (default)
        if not isinstance(raw_preds, torch.Tensor) or raw_preds.ndim != 3 or raw_preds.shape[2] != 3:
            logger.error(f"Unpacked prediction component is not a valid quantile tensor. Shape: {raw_preds.shape if isinstance(raw_preds, torch.Tensor) else type(raw_preds)}. Skipping plots.")
            return results
        # Check x_output
        if not isinstance(x_output, dict) or 'decoder_target' not in x_output:
             logger.error(f"Unpacked x_output is not a dict or missing 'decoder_target'. Type: {type(x_output)}. Skipping plots.")
             return results
        # Check index_df
        if not isinstance(index_df, pd.DataFrame) or 'time_idx' not in index_df.columns:
             logger.error(f"Unpacked index_df is not a DataFrame or missing 'time_idx'. Type: {type(index_df)}. Skipping plots.")
             return results
        # --- End Validation ---

        # Proceed with plotting
        preds_log_tensor = raw_preds.cpu()
        actuals_log_tensor = x_output['decoder_target'].cpu()
        time_idx_flat = index_df['time_idx'].values

        # Flatten log predictions (p10, p50, p90)
        preds_log_p10_flat = preds_log_tensor[:, :, 0].flatten().numpy()
        preds_log_p50_flat = preds_log_tensor[:, :, 1].flatten().numpy()
        preds_log_p90_flat = preds_log_tensor[:, :, 2].flatten().numpy()
        actuals_log_flat_plot = actuals_log_tensor.flatten().numpy()

        all_val_time_indices_for_plot = []
        # index_df_from_predict contains starting time_idx for each sequence in the batch from dataloader
        for start_idx_pred_seq in index_df['time_idx']:
            all_val_time_indices_for_plot.extend(list(range(start_idx_pred_seq, start_idx_pred_seq + model.hparams.max_prediction_length)))

        min_len_for_plot = min(len(all_val_time_indices_for_plot), len(actuals_log_flat_plot), len(preds_log_p50_flat))
        if len(all_val_time_indices_for_plot) != min_len_for_plot or \
        len(actuals_log_flat_plot) != min_len_for_plot or \
        len(preds_log_p50_flat) != min_len_for_plot:
            logger.warning(f"Plot length mismatch in evaluate_model ({len(all_val_time_indices_for_plot)} vs "
                        f"{len(actuals_log_flat_plot)} vs {len(preds_log_p50_flat)}): Truncating to {min_len_for_plot}.")

        time_idx_for_plot_flat_np = np.array(all_val_time_indices_for_plot[:min_len_for_plot])
        # Actuals and predictions already flattened and possibly truncated
        actuals_log_plot_final = actuals_log_flat_plot[:min_len_for_plot]
        preds_log_p10_plot_final = preds_log_p10_flat[:min_len_for_plot]
        preds_log_p50_plot_final = preds_log_p50_flat[:min_len_for_plot]
        preds_log_p90_plot_final = preds_log_p90_flat[:min_len_for_plot]

        # Create a DataFrame to sort and handle potential duplicates if validation windows overlap
        plot_df = pd.DataFrame({
            'time_idx': time_idx_for_plot_flat_np,
            'actuals_log': actuals_log_plot_final,
            'preds_log_p10': preds_log_p10_plot_final,
            'preds_log_p50': preds_log_p50_plot_final,
            'preds_log_p90': preds_log_p90_plot_final,
        })
        # If predict=False in val_ds, multiple sequences might predict the same time_idx.
        # We usually take the first occurrence for actuals (as they are non-random)
        # and might average predictions or take first for simplicity here.
        # For actuals, keep='first' is good. For predictions from different contexts, mean might be better.
        # Let's assume for plotting, first is okay, or this means val_ds was created with predict=True or non-overlapping.
        plot_df = plot_df.drop_duplicates(subset=['time_idx'], keep='first').sort_values('time_idx')

        # Inverse transform to original scale
        actuals_orig_plot = np.expm1(plot_df['actuals_log'].values)
        p10_orig_plot = np.maximum(0, np.expm1(plot_df['preds_log_p10'].values))
        p50_orig_plot = np.maximum(0, np.expm1(plot_df['preds_log_p50'].values))
        p90_orig_plot = np.maximum(0, np.expm1(plot_df['preds_log_p90'].values))
        time_idx_sorted_final = plot_df['time_idx'].values # These are the final unique time_idx to plot

        try:
            # Determine candidate indices to drop
            drop_mask = pd.Series(False, index=actuals_val_orig.index)
            # If week_date is available – use ISO calendar week == 53
            if 'week_date' in index_df_plot.columns and not index_df_plot['week_date'].isnull().all():
                iso_weeks = pd.to_datetime(index_df_plot.set_index('time_idx')['week_date']).dt.isocalendar().week
                drop_mask |= (iso_weeks == 53)
            # Always consider the single last point a spike candidate
            drop_mask.loc[actuals_val_orig.index.max()] = True
            # Apply mask if at least one point flagged
            if drop_mask.any():
                logger.info(f"Excluding {drop_mask.sum()} week‑53/last‑week point(s) from plots & metrics.")
                # Filter all validation‑aligned objects
                actuals_val_orig = actuals_val_orig.loc[~drop_mask]
                tft_preds_val_orig = tft_preds_val_orig.loc[~drop_mask]
                index_df_plot = index_df_plot[~index_df_plot['time_idx'].isin(drop_mask[drop_mask].index)]
        except Exception as wk53_err:
            logger.warning(f"Week‑53 filtering failed (continuing without filter): {wk53_err}")


        # --- Generate Forecast Plot ---
        plot_fig_eval, ax_eval = plt.subplots(figsize=(18, 7)) # Use a different fig variable name

        # Use time_idx_sorted_final for x-data for predictions
        ax_eval.plot(time_idx_sorted_final, actuals_orig_plot, label="Actual Deaths", marker='.', linestyle='-', alpha=0.7, color='black', markersize=3, linewidth=0.8)
        ax_eval.plot(time_idx_sorted_final, p50_orig_plot, label="Predicted Median (p50)", linestyle='--', alpha=0.9, color='tab:orange', linewidth=1.2)
        ax_eval.fill_between(time_idx_sorted_final, p10_orig_plot, p90_orig_plot, color='tab:orange', alpha=0.3, label='p10-p90 Quantiles')

        plot_title = f"TFT Forecast vs Actuals ({run_name})\nMAE={val_mae:.2f}, SMAPE={val_smape:.2f}%" # val_mae, val_smape are from overall eval
        ax_eval.set_title(plot_title, fontsize=14)

        # Use helper for x-axis labels
        # Ensure time_idx_sorted_final is what you want to label
        tick_positions, tick_labels = convert_time_idx_to_date_labels(
            pd.Series(time_idx_sorted_final), # Pass the actual time_idx values being plotted
            min_overall_time_idx,
            min_overall_week_date,
            label_type="year_week", # Or "date" for YYYY-MM-DD
            tick_frequency=max(1, len(time_idx_sorted_final) // 10) # Aim for ~10 labels
        )
        if len(tick_positions) > 0:
            ax_eval.set_xticks(tick_positions)
            ax_eval.set_xticklabels(tick_labels, rotation=45, ha='right')
            x_axis_label_str = "Time (Year-Week)"
        else: # Fallback if no ticks generated
            ax_eval.set_xlabel("Time Index (Weeks)", fontsize=12)
            x_axis_label_str = "Time Index (Weeks)"
        ax_eval.set_xlabel(x_axis_label_str, fontsize=12)

        ax_eval.set_ylabel("Weekly Deaths (Original Scale)", fontsize=12)
        ax_eval.legend(fontsize=10); ax_eval.grid(True, linestyle=':', alpha=0.7)
        
        # Adjust layout carefully if using autofmt_xdate
        plt.tight_layout() # Try this first
        is_date_like = False
        if len(tick_positions) > 0:
            try:
                # attempt to parse; raise on failure so we can flag non-date labels
                pd.to_datetime(tick_labels[0], errors="raise")
                is_date_like = True
            except Exception:
                is_date_like = False

        if is_date_like:
            try:
                plot_fig_eval.autofmt_xdate(rotation=30, ha='right') # Apply AFTER tight_layout if issues
            except Exception as e_fmt:
                logger.warning(f"autofmt_xdate failed: {e_fmt}")

        plot_file = os.path.join(plot_dir, f"tft_val_forecast_{run_name}_original_scale.png") # Keep original name or change
        plt.savefig(plot_file); logger.info(f"Saved forecast plot for '{run_name}' to {plot_file}");
        plt.close(plot_fig_eval)


        # --- Residual Plot ---
        residuals_plot = actuals_orig_plot - p50_orig_plot # Use the aligned data
        plot_fig_res_eval, ax_res_eval = plt.subplots(figsize=(10, 6)) # Different fig var name
        ax_res_eval.scatter(p50_orig_plot, residuals_plot, alpha=0.3, s=15, color='tab:blue', edgecolors='k', linewidth=0.5)
        ax_res_eval.axhline(0, color='red', linestyle='--', linewidth=1)
        ax_res_eval.set_title(f'Residual Plot ({run_name}, Original Scale)', fontsize=14)
        ax_res_eval.set_xlabel('Predicted Median Deaths (Original Scale)', fontsize=12)
        ax_res_eval.set_ylabel('Residuals (Original Scale)', fontsize=12)
        ax_res_eval.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
        save_path_res = os.path.join(plot_dir, f"residuals_{run_name}_original_scale_plot.png")
        plt.savefig(save_path_res); logger.info(f"Saved residual plot for '{run_name}' to {save_path_res}");
        plt.close(plot_fig_res) # Close this specific figure

    except Exception as e:
        logger.warning(f"Evaluation plotting failed: {e}", exc_info=True)
    finally:
        # Attempt to close any figures that might still be open
        if 'plot_fig' in locals() and plot_fig is not None and plt.fignum_exists(plot_fig.number): plt.close(plot_fig)
        if 'plot_fig_res' in locals() and plot_fig_res is not None and plt.fignum_exists(plot_fig_res.number): plt.close(plot_fig_res)
        plt.close('all') # General cleanup

    return results

# -----------------------------------------------------------------------------
# 6. Enhanced Plotting Functions (UPDATED FOR WEEKLY)
# -----------------------------------------------------------------------------
# (Make sure plt is imported: import matplotlib.pyplot as plt)

def plot_time_series(df: pd.DataFrame, time_col: str, value_col: str, title: str,
                     ylabel: str, filename: str, plot_dir: str,
                     zoom_ylim: Optional[Tuple[float, float]] = None,
                     mark_zero_threshold: float = 0.01):
    """
    Plots a simple time series, optionally zooming the y-axis and marking near-zero points.

    Args:
        df: DataFrame containing the data.
        time_col: Name of the column for the x-axis (e.g., 'week_date').
        value_col: Name of the column for the y-axis.
        title: Plot title.
        ylabel: Y-axis label.
        filename: Name for the saved plot file.
        plot_dir: Directory to save the plot.
        zoom_ylim: Optional tuple (min_y, max_y) to set y-axis limits.
        mark_zero_threshold: If zoom_ylim is set, mark points below this threshold.
    """
    logger.info(f"Generating plot: {title}")
    if time_col not in df.columns or value_col not in df.columns:
        logger.error(f"Plot fail '{filename}': Missing columns '{time_col}' or '{value_col}'.")
        return
    if df.empty:
        logger.warning(f"DataFrame empty for plot '{filename}'. Skipping.")
        return

    try:
        fig, ax = plt.subplots(figsize=(18, 6)) # Wider plot for weekly data

        # Plot the main series
        ax.plot(df[time_col], df[value_col], marker='.', linestyle='-',
                markersize=1.5, alpha=0.6, linewidth=0.8, label=ylabel)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(time_col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)

        # Handle y-axis zoom and zero marking
        if zoom_ylim is not None:
            try:
                min_y, max_y = zoom_ylim
                # Find points below the threshold
                zero_mask = df[value_col] < mark_zero_threshold
                if zero_mask.any():
                    # Plot markers for zero/near-zero points just below the zoomed area
                    # Adjust marker_y_pos if min_y can be negative
                    marker_y_pos = min_y - (max_y - min_y) * 0.02 # Position slightly below min_y
                    ax.plot(df.loc[zero_mask, time_col],
                            [marker_y_pos] * zero_mask.sum(),
                            marker='|', linestyle='None', markersize=5, color='red', alpha=0.5,
                            label=f'< {mark_zero_threshold:.2f} threshold')
                    ax.legend() # Show legend including the threshold markers

                logger.info(f"Applying y-axis zoom: {zoom_ylim}")
                ax.set_ylim(zoom_ylim)
            except Exception as e:
                logger.error(f"Failed to apply zoom/mark zeros for '{filename}': {e}")
                # Fallback to default y-limits if zoom fails
                pass # Let matplotlib auto-scale

        plt.tight_layout()
        # Add rotation for dates if time_col is week_date
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
             fig.autofmt_xdate(rotation=45, ha='right')

        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path)
        logger.info(f"Saved plot: {save_path}")

    except Exception as e:
        logger.error(f"Plot fail '{filename}': {e}", exc_info=True)
    finally:
        # Ensure figure is closed
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        else:
            plt.close() # Close the current figure implicitly created

def plot_dual_axis(df: pd.DataFrame, time_col: str, col1: str, col2: str, label1: str, label2: str, title: str, filename: str, plot_dir: str):
    """Plots two time series on a dual-axis chart (weekly focus)."""
    logger.info(f"Generating plot: {title}")
    if not all(c in df.columns for c in [time_col, col1, col2]): logger.error(f"Plot fail '{filename}': Missing cols."); return
    fig, ax1 = plt.subplots(figsize=(18, 6)) # Wider plot
    try:
        time_data = df[time_col]; x_label = time_col.replace('_', ' ').title()
        if pd.api.types.is_datetime64_any_dtype(time_data): pass
        else: start_date_str = df['week_date'].min().strftime('%Y-%m-%d'); x_label = f"Time Index (Weeks since {start_date_str})"
        color1 = 'tab:blue'; ax1.set_xlabel(x_label, fontsize=12); ax1.set_ylabel(label1, color=color1, fontsize=12)
        line1 = ax1.plot(time_data, df[col1], color=color1, label=label1, alpha=0.8, linewidth=1.2)
        ax1.tick_params(axis='y', labelcolor=color1); ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
        ax2 = ax1.twinx(); color2 = 'tab:red'; ax2.set_ylabel(label2, color=color2, fontsize=12)
        line2 = ax2.plot(time_data, df[col2], color=color2, label=label2, linestyle='--', alpha=0.8, linewidth=1.2)
        ax2.tick_params(axis='y', labelcolor=color2); fig.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.title(title, fontsize=14);
        lines = line1 + line2; labels = [l.get_label() for l in lines]; ax1.legend(lines, labels, loc='upper left')
        plt.savefig(os.path.join(plot_dir, filename)); logger.info(f"Saved plot: {filename}")
    except Exception as e: logger.error(f"Plot fail {filename}: {e}", exc_info=True)
    finally: plt.close(fig)

def plot_scatter_fear_vs_deaths(df: pd.DataFrame, fear_col: str, death_col: str, title: str, filename: str, plot_dir: str):
    """Plots a scatter plot of fear score vs deaths (weekly focus)."""
    logger.info(f"Generating plot: {title}")
    if not all(c in df.columns for c in [fear_col, death_col]): logger.error(f"Plot fail '{filename}': Missing cols."); return
    try:
        plt.figure(figsize=(8, 8)); sns.scatterplot(data=df, x=fear_col, y=death_col, alpha=0.2, s=10, edgecolor=None) # More transparency for weekly
        corr = df[fear_col].corr(df[death_col]); plt.title(f"{title}\n(Correlation: {corr:.2f})")
        plt.xlabel(fear_col.replace('_', ' ').title()); plt.ylabel(death_col.replace('_', ' ').title())
        plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout(); plt.savefig(os.path.join(plot_dir, filename)); logger.info(f"Saved plot: {filename}")
    except Exception as e: logger.error(f"Plot fail {filename}: {e}", exc_info=True)
    finally: plt.close()

def plot_weekly_boxplot(df: pd.DataFrame, column: str, title: str, filename: str, plot_dir: str):
    """Plots a boxplot grouped by week of year."""
    logger.info(f"Generating plot: {title}")
    if 'week_of_year' not in df.columns or column not in df.columns: logger.error(f"Plot fail '{filename}': Missing 'week_of_year' or '{column}'."); return
    try:
        df['week_num'] = df['week_of_year'].astype(int)
        week_order = [str(i) for i in range(1, 54)] # Weeks 1 to 53
        plt.figure(figsize=(18, 7)) # Wider plot
        sns.boxplot(x='week_of_year', y=column, data=df, order=week_order, showfliers=False, palette="coolwarm")
        plt.title(title, fontsize=14); plt.xlabel("Week of Year", fontsize=12); plt.ylabel(column.replace('_', ' ').title(), fontsize=12)
        # Reduce number of x-axis labels shown
        tick_freq = 5
        plt.xticks(ticks=range(0, 53, tick_freq), labels=[str(i) for i in range(1, 54, tick_freq)], rotation=45, ha='right')
        plt.tight_layout(); plt.savefig(os.path.join(plot_dir, filename)); logger.info(f"Saved plot: {filename}")
    except Exception as e: logger.error(f"Failed weekly boxplot {filename}: {e}", exc_info=True)
    finally: plt.close()


def plot_sentiment_with_rolling_stats(df: pd.DataFrame, time_col: str, value_col: str,
                                      window: int, title: str, ylabel: str,
                                      filename: str, plot_dir: str):
    """
    Plots a sentiment time series along with its rolling mean and rolling standard deviation.

    Args:
        df: DataFrame containing the data.
        time_col: Name of the column for the x-axis (e.g., 'week_date').
        value_col: Name of the sentiment column for the y-axis.
        window: Integer window size for rolling statistics.
        title: Plot title.
        ylabel: Y-axis label for the raw score.
        filename: Name for the saved plot file.
        plot_dir: Directory to save the plot.
    """
    logger.info(f"Generating rolling stats plot: {title}")
    if time_col not in df.columns or value_col not in df.columns:
        logger.error(f"Plot fail '{filename}': Missing columns '{time_col}' or '{value_col}'.")
        return
    if df.empty:
        logger.warning(f"DataFrame empty for plot '{filename}'. Skipping.")
        return

    try:
        # Calculate rolling statistics
        rolling_mean = df[value_col].rolling(window=window, center=True, min_periods=1).mean()
        rolling_std = df[value_col].rolling(window=window, center=True, min_periods=1).std()

        fig, ax1 = plt.subplots(figsize=(18, 7)) # Slightly taller for dual axis

        color1 = 'lightblue'
        ax1.set_xlabel(time_col.replace('_', ' ').title(), fontsize=12)
        ax1.set_ylabel(ylabel, color=color1, fontsize=12)
        # Plot raw score lightly in the background
        ax1.plot(df[time_col], df[value_col], color=color1, label=f'{ylabel} (Raw)', alpha=0.4, linewidth=0.5, marker='.', markersize=1)
        # Plot rolling mean prominently
        line1 = ax1.plot(df[time_col], rolling_mean, color='blue', label=f'{ylabel} ({window}w Rolling Mean)', linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
        y_min, y_max = rolling_mean.min(), rolling_mean.max()
        y_range = y_max - y_min
        ax1.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range) # Auto-adjust ylim based on mean

        # Create a second y-axis for rolling std dev
        ax2 = ax1.twinx()
        color2 = 'darkorange'
        ax2.set_ylabel(f'{window}w Rolling Std Dev', color=color2, fontsize=12)
        line2 = ax2.plot(df[time_col], rolling_std, color=color2, label=f'{ylabel} ({window}w Rolling Std Dev)', linestyle='--', linewidth=1.2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(bottom=0) # Std dev cannot be negative

        fig.suptitle(title, fontsize=14)
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout considering suptitle

        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
             fig.autofmt_xdate(rotation=45, ha='right')

        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path)
        logger.info(f"Saved rolling stats plot: {save_path}")

    except Exception as e:
        logger.error(f"Plot fail '{filename}': {e}", exc_info=True)
    finally:
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        else:
            plt.close()

def plot_training_history(log_dir: str, run_name: str, plot_dir: str):
    """
    Plots training and validation loss curves from TensorBoard metrics.csv.

    Args:
        log_dir: Path to the specific version directory inside lightning_logs
                 (e.g., 'lightning_logs/my_run_name/version_0').
        run_name: Name of the training run (for plot title/filename).
        plot_dir: Directory to save the plot.
    """
    metrics_path = os.path.join(log_dir, "metrics.csv")
    logger.info(f"Attempting to plot training history for '{run_name}' from: {metrics_path}")

    if not os.path.exists(metrics_path):
        logger.warning(f"Metrics file not found at {metrics_path}. Skipping training history plot.")
        return

    plot_fig_train = None # Initialize figure variable
    try:
        metrics_df = pd.read_csv(metrics_path)

        # Filter for epoch-level metrics if step metrics are also present
        # Plot 'val_loss' and 'train_loss_epoch' if available
        epochs = metrics_df[metrics_df["val_loss"].notna()]["epoch"]
        val_loss = metrics_df[metrics_df["val_loss"].notna()]["val_loss"]

        # Check if train_loss_epoch exists, otherwise try train_loss_step (less ideal)
        if "train_loss_epoch" in metrics_df.columns:
            train_loss = metrics_df[metrics_df["train_loss_epoch"].notna()]["train_loss_epoch"]
            train_epochs = metrics_df[metrics_df["train_loss_epoch"].notna()]["epoch"]
            train_loss_label = "Train Loss (Epoch)"
        elif "train_loss_step" in metrics_df.columns:
             # Aggregate step loss by epoch (simple mean)
             logger.warning("train_loss_epoch not found, using mean train_loss_step per epoch.")
             train_loss_agg = metrics_df.dropna(subset=["train_loss_step"]).groupby("epoch")["train_loss_step"].mean()
             train_loss = train_loss_agg.values
             train_epochs = train_loss_agg.index
             train_loss_label = "Train Loss (Step Avg)"
        else:
            logger.warning("No training loss found in metrics.csv. Plotting validation loss only.")
            train_loss = None

        if val_loss.empty and (train_loss is None or len(train_loss) == 0) :
            logger.warning("No valid loss data found in metrics file. Skipping plot.")
            return

        plot_fig_train, ax = plt.subplots(figsize=(12, 6))

        if not val_loss.empty:
            ax.plot(epochs, val_loss, label="Validation Loss", marker='o', linestyle='-')
        if train_loss is not None and len(train_loss) > 0:
             # Ensure train_epochs aligns if using aggregated step loss
             if len(train_epochs) == len(train_loss):
                 ax.plot(train_epochs, train_loss, label=train_loss_label, marker='x', linestyle='--')
             else:
                 logger.warning("Length mismatch between train_epochs and train_loss. Skipping train loss plot.")


        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Quantile Loss)")
        ax.set_title(f"Training History - {run_name}")
        ax.legend()
        ax.grid(True, linestyle=':')
        plt.tight_layout()

        plot_filename = f"training_history_{run_name}.png"
        save_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(save_path)
        logger.info(f"Saved training history plot for '{run_name}' to {save_path}")

    except FileNotFoundError:
         logger.warning(f"Metrics file not found at {metrics_path}. Cannot plot training history.")
    except KeyError as e:
         logger.warning(f"Could not find expected columns ('val_loss', 'epoch', etc.) in {metrics_path}: {e}. Skipping plot.")
    except Exception as e:
        logger.error(f"Failed to plot training history for '{run_name}': {e}", exc_info=True)
    finally:
        if plot_fig_train is not None and plt.fignum_exists(plot_fig_train.number):
            plt.close(plot_fig_train)
        plt.close('all') # Close any other figures

def plot_distribution(series: pd.Series, title: str, xlabel: str, filename: str, plot_dir: str, bins: int = 50, filter_zeros: bool = True): # Added filter_zeros parameter
    """
    Plots a histogram and KDE of a given series, marking the mean.
    Optionally filters out exact zero values before plotting.
    """
    logger.info(f"Generating distribution plot: {title} (Filter Zeros: {filter_zeros})")
    if series.empty:
        logger.warning(f"Series empty for distribution plot '{filename}'. Skipping.")
        return
    if series.isnull().all():
        logger.warning(f"Series contains only NaNs for distribution plot '{filename}'. Skipping.")
        return

    dist_fig = None
    try:
        dist_fig, ax = plt.subplots(figsize=(10, 6))
        
        # Drop NaNs first
        series_cleaned = series.dropna()

        # *** NEW: Filter out zeros if requested ***
        if filter_zeros:
            original_count = len(series_cleaned)
            series_cleaned = series_cleaned[series_cleaned != 0]
            num_filtered = original_count - len(series_cleaned)
            if num_filtered > 0:
                logger.info(f"Filtered out {num_filtered} zero values for distribution plot '{filename}'.")
        
        if series_cleaned.empty:
            logger.warning(f"Series empty after dropping NaNs (and optionally zeros) for distribution plot '{filename}'. Skipping.")
            if dist_fig: # Close the figure if it was created
                plt.close(dist_fig)
            return

        sns.histplot(series_cleaned, kde=True, ax=ax, bins=bins, stat="density", color="skyblue", edgecolor="black", linewidth=0.5)
        
        mean_val = series_cleaned.mean()
        std_val = series_cleaned.std()
        
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.3f}')
        
        ax.set_title(f"{title}\n(Std Dev: {std_val:.3f}){' (Zeros Filtered)' if filter_zeros and num_filtered > 0 else ''}", fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path)
        logger.info(f"Saved distribution plot: {save_path}")

    except Exception as e:
        logger.error(f"Distribution Plot fail '{filename}': {e}", exc_info=True)
    finally:
        if dist_fig is not None and plt.fignum_exists(dist_fig.number):
            plt.close(dist_fig)
        plt.close('all')

def plot_ccf(df: pd.DataFrame, var1: str, var2: str, max_lags: int,
    title: str, filename: str, plot_dir: str):
    """Plots the Cross-Correlation Function (CCF) between two variables."""
    logger.info(f"Generating CCF plot: {title}")
    if var1 not in df.columns or var2 not in df.columns:
        logger.error(f"CCF Plot fail '{filename}': Missing columns '{var1}' or '{var2}'.")
        return
    if df[[var1, var2]].isnull().any().any():
        logger.warning(f"NaNs found in columns for CCF '{filename}'. Dropping NaNs for calculation.")
        df_ccf = df[[var1, var2]].dropna()
    else:
        df_ccf = df[[var1, var2]]

    if len(df_ccf) < max_lags * 2:
        logger.warning(f"Not enough data points ({len(df_ccf)}) for CCF plot '{filename}' with max_lags={max_lags}. Skipping.")
        return

    ccf_fig = None # Initialize figure variable
    try:
        # Calculate CCF - Note: ccf(x, y) calculates corr(x_{t+k}, y_t)
        # We want corr(var1_{t+k}, var2_t) -> var1 leads var2 for positive k
        # Or corr(var1_t, var2_{t+k}) -> var2 leads var1 for positive k
        # Let's calculate corr(var1_{t+k}, var2_t) -> var1 is leading for positive k
        correlation = ccf(df_ccf[var1], df_ccf[var2], adjusted=False) # Calculate raw correlation

        nlags = max_lags
        lags = np.arange(-nlags, nlags + 1)
        # Extract relevant lags from ccf output (it calculates for positive lags only relative to first series)
        # We need to reconstruct for negative lags as well
        ccf_vals = np.zeros(len(lags))
        # Positive lags (k >= 0): corr(var1_{t+k}, var2_t)
        ccf_vals[nlags:] = correlation[:nlags+1]
        # Negative lags (k < 0): corr(var1_{t-|k|}, var2_t) = corr(var1_t, var2_{t+|k|})
        # This requires calculating ccf in the other direction
        correlation_rev = ccf(df_ccf[var2], df_ccf[var1], adjusted=False)
        ccf_vals[:nlags] = correlation_rev[1:nlags+1][::-1] # Get lags 1 to nlags and reverse

        # Calculate confidence intervals (approximate for large samples)
        conf_level = 1.96 / np.sqrt(len(df_ccf))

        ccf_fig, ax = plt.subplots(figsize=(12, 5))
        # Use stem plot for correlations
        markerline, stemlines, baseline = ax.stem(lags, ccf_vals, linefmt='grey', markerfmt='o', basefmt='black')
        plt.setp(markerline, 'color', 'blue', 'markersize', 4)
        plt.setp(stemlines, 'color', 'blue', 'linewidth', 0.5)

        # Plot confidence intervals
        ax.axhline(conf_level, color='grey', linestyle='--', linewidth=0.8)
        ax.axhline(-conf_level, color='grey', linestyle='--', linewidth=0.8)
        ax.fill_between(lags, -conf_level, conf_level, color='grey', alpha=0.15)

        ax.set_xlabel(f"Lag (Weeks) - Positive lag means '{var1}' leads '{var2}'")
        ax.set_ylabel("Cross-correlation")
        ax.set_title(title)
        ax.grid(True, linestyle=':')
        plt.tight_layout()
        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path); logger.info(f"Saved plot: {save_path}")

    except Exception as e:
        logger.error(f"CCF Plot fail '{filename}': {e}", exc_info=True)
    finally:
        if ccf_fig is not None and plt.fignum_exists(ccf_fig.number):
            plt.close(ccf_fig)
        plt.close('all')


def plot_rolling_correlation(df: pd.DataFrame, time_col: str, var1: str, var2: str,
                           window: int, title: str, filename: str, plot_dir: str):
    """Plots the rolling correlation between two variables."""
    logger.info(f"Generating rolling correlation plot: {title}")
    if time_col not in df.columns or var1 not in df.columns or var2 not in df.columns:
        logger.error(f"Rolling Correlation fail '{filename}': Missing columns.")
        return
    if df[[var1, var2]].isnull().any().any():
        logger.warning(f"NaNs found in columns for rolling correlation '{filename}'. Calculation might produce NaNs.")

    roll_fig = None
    try:
        rolling_corr = df[var1].rolling(window=window, center=True, min_periods=window // 2).corr(df[var2])

        roll_fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(df[time_col], rolling_corr, label=f'{window//52}y Rolling Correlation', linewidth=1.5)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.set_xlabel(time_col.replace('_', ' ').title())
        ax.set_ylabel("Correlation Coefficient")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle=':')
        plt.tight_layout()
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            roll_fig.autofmt_xdate(rotation=45, ha='right')
        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path); logger.info(f"Saved plot: {save_path}")

    except Exception as e:
        logger.error(f"Rolling Correlation fail '{filename}': {e}", exc_info=True)
    finally:
        if roll_fig is not None and plt.fignum_exists(roll_fig.number):
            plt.close(roll_fig)
        plt.close('all')

def plot_acf_pacf(series: pd.Series, lags: int, title_prefix: str, filename_suffix: str, plot_dir: str):
    """Plots ACF and PACF for a given time series."""
    logger.info(f"Generating ACF/PACF plots for: {title_prefix}")
    if series.isnull().any():
        logger.warning(f"NaNs found in series for ACF/PACF '{title_prefix}'. Dropping NaNs.")
        series = series.dropna()
    if len(series) < lags * 2:
         logger.warning(f"Not enough data points ({len(series)}) for ACF/PACF plot '{title_prefix}' with lags={lags}. Skipping.")
         return

    acf_fig = None; pacf_fig = None
    try:
        # ACF Plot
        acf_fig = plt.figure(figsize=(12, 5))
        plot_acf(series, lags=lags, ax=acf_fig.gca(), title=f'{title_prefix} - Autocorrelation (ACF)')
        plt.tight_layout()
        acf_filename = f"acf_{filename_suffix}.png"
        plt.savefig(os.path.join(plot_dir, acf_filename)); logger.info(f"Saved plot: {acf_filename}")
        plt.close(acf_fig)

        # PACF Plot
        pacf_fig = plt.figure(figsize=(12, 5))
        # Method 'ywm' is often more stable for PACF
        plot_pacf(series, lags=lags, ax=pacf_fig.gca(), title=f'{title_prefix} - Partial Autocorrelation (PACF)', method='ywm')
        plt.tight_layout()
        pacf_filename = f"pacf_{filename_suffix}.png"
        plt.savefig(os.path.join(plot_dir, pacf_filename)); logger.info(f"Saved plot: {pacf_filename}")
        plt.close(pacf_fig)

    except Exception as e:
        logger.error(f"ACF/PACF Plot fail for '{title_prefix}': {e}", exc_info=True)
    finally:
         if acf_fig is not None and plt.fignum_exists(acf_fig.number): plt.close(acf_fig)
         if pacf_fig is not None and plt.fignum_exists(pacf_fig.number): plt.close(pacf_fig)
         plt.close('all')

def analyze_bivariate_var_irf(df: pd.DataFrame, target_col: str, predictor_col: str,
                              max_lags_var: int, irf_periods: int,
                              plot_dir: str, run_name: str):
    """
    (REVISED) Fits a BIVARIATE VAR model using FIRST DIFFERENCES of the input columns 
    (target_col, predictor_col) from the provided DataFrame. Includes BIC lag selection. 
    Plots three key orthogonalized Impulse Response Functions (IRFs):
    1. Response of target to shock in predictor
    2. Response of predictor to its own shock
    3. Response of predictor to shock in target (feedback)

    Args:
        df: DataFrame containing the ORIGINAL UNSMOOTHED time series data.
        target_col: Name of the primary target column (e.g., 'log_deaths').
        predictor_col: Name of the predictor/impulse column (e.g., 'hardship_sentiment').
        max_lags_var: Maximum lags to consider for VAR model order selection.
        irf_periods: Number of periods (weeks) ahead to plot the impulse response.
        plot_dir: Directory to save plots.
        run_name: Base name for the analysis run (used in filenames).
    """
    bivar_run_id = f"{predictor_col}_vs_{target_col}"
    logger.info(f"--- Running BIVARIATE VAR/IRF (on Differences): {bivar_run_id} ---")

    if target_col not in df.columns or predictor_col not in df.columns:
        logger.error(f"Bivariate VAR/IRF fail: Missing columns '{target_col}' or '{predictor_col}'.")
        return

    # Select ONLY the two columns from the original dataframe
    data_orig = df[[target_col, predictor_col]].copy()
    # Drop rows with NaNs in EITHER column BEFORE differencing
    data_orig.dropna(inplace=True) 

    if data_orig.shape[0] < max_lags_var + irf_periods + 5: # Need enough data for lags + IRF horizon
        logger.error(f"Not enough non-NaN paired observations ({data_orig.shape[0]}) for VAR/IRF {bivar_run_id}.")
        return
        
    # --- Difference the data for stationarity ---
    try:
        data_diff = data_orig.diff().dropna() # First difference
        if data_diff.shape[0] < max_lags_var + irf_periods + 5: # Check length AFTER differencing
            logger.error(f"Not enough data points ({data_diff.shape[0]}) after differencing for VAR/IRF {bivar_run_id}.")
            return
        # Check for constant columns AFTER differencing
        if (data_diff[target_col].nunique() <= 1) or (data_diff[predictor_col].nunique() <= 1):
            logger.error(f"Data for '{target_col}' or '{predictor_col}' became constant after differencing. Bivariate VAR impossible for {bivar_run_id}.")
            return
        logger.info(f"Using differenced data for VAR model: {bivar_run_id}")
    except Exception as e:
        logger.error(f"Differencing failed for Bivariate VAR {bivar_run_id}: {e}")
        return
    # --- End Differencing ---

    # Initialize figure variables to prevent reference errors in finally block
    var_fig_pred_to_target = None; var_fig_self = None; var_fig_target_to_pred = None
    
    try:
        # Fit the BIVARIATE VAR model on DIFFERENCED data
        # Ensure the order of columns passed to VAR matches how you want to interpret impulse/response
        model_data = data_diff[[target_col, predictor_col]] 
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ValueWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=EstimationWarning)

            model = VAR(model_data)
            
            # Lag Order Selection using BIC
            chosen_lags = None
            try:
                 selected_lags_result = model.select_order(maxlags=max_lags_var)
                 chosen_lags = selected_lags_result.bic # Use Bayesian Information Criterion
                 # Validate selected lag
                 if not isinstance(chosen_lags, int) or not (0 < chosen_lags <= max_lags_var):
                      logger.warning(f"BIC selected lag {chosen_lags} is invalid or out of range (1-{max_lags_var}). Defaulting to lag {LAG_WEEKS}.")
                      chosen_lags = LAG_WEEKS 
                 else:
                      logger.info(f"Bivariate VAR ({bivar_run_id}) using BIC selected lag order: {chosen_lags}")
            except Exception as lag_err:
                 logger.warning(f"Bivariate VAR lag selection failed for {bivar_run_id}: {lag_err}. Defaulting to lag {LAG_WEEKS}.")
                 chosen_lags = LAG_WEEKS

            # Fit the model with the chosen number of lags
            results = model.fit(chosen_lags)
            logger.info(f"Bivariate VAR model ({bivar_run_id}) fitted with lags={results.k_ar}")

        # Calculate Orthogonalized Impulse Response Functions 
        # Orthogonalized IRFs account for contemporaneous correlation by ordering variables (Cholesky decomposition)
        # The default order is based on the column order in the VAR model input (model_data).
        # Shocking the first variable assumes it's contemporaneously independent of the second.
        # Shocking the second allows for immediate response from the first.
        # Here, target_col is first, predictor_col is second.
        irf = results.irf(periods=irf_periods)

        # --- Plot 1: Response of target_col to impulse in predictor_col ---
        impulse_var = predictor_col
        response_var = target_col
        logger.info(f"Plotting Orth. Bivar IRF: Impulse={impulse_var}, Response={response_var}")
        var_fig_pred_to_target = irf.plot(impulse=impulse_var, response=response_var, stderr_type='mc', signif=0.05, plot_stderr=True, orth=True)
        var_fig_pred_to_target.suptitle(f'Orth. IR (Bivariate): Shock to {impulse_var} -> Response in {response_var}\n(Based on Differenced Data)', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout for suptitle
        filename_response = f"irf_bivar_{impulse_var}_to_{response_var}_{run_name}.png"
        save_path_response = os.path.join(plot_dir, filename_response)
        var_fig_pred_to_target.savefig(save_path_response)
        logger.info(f"Saved IRF plot to {save_path_response}")
        plt.close(var_fig_pred_to_target) # Close the specific figure

        # --- Plot 2: Response of predictor_col to its OWN impulse ---
        impulse_var = predictor_col
        response_var = predictor_col
        logger.info(f"Plotting Orth. Bivar IRF: Impulse={impulse_var}, Response={response_var}")
        var_fig_self = irf.plot(impulse=impulse_var, response=response_var, stderr_type='mc', signif=0.05, plot_stderr=True, orth=True)
        var_fig_self.suptitle(f'Orth. IR (Bivariate): Shock to {impulse_var} -> Response in {response_var}\n(Based on Differenced Data)', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        filename_self = f"irf_bivar_{impulse_var}_to_self_{run_name}.png"
        save_path_self = os.path.join(plot_dir, filename_self)
        var_fig_self.savefig(save_path_self)
        logger.info(f"Saved Self-IRF plot to {save_path_self}")
        plt.close(var_fig_self)

        # --- Plot 3: Response of predictor_col to impulse in target_col (Feedback) ---
        impulse_var = target_col
        response_var = predictor_col
        logger.info(f"Plotting Orth. Bivar IRF: Impulse={impulse_var}, Response={response_var}")
        var_fig_target_to_pred = irf.plot(impulse=impulse_var, response=response_var, stderr_type='mc', signif=0.05, plot_stderr=True, orth=True)
        var_fig_target_to_pred.suptitle(f'Orth. IR (Bivariate): Shock to {impulse_var} -> Response in {response_var}\n(Based on Differenced Data)', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        filename_target_to_pred = f"irf_bivar_{impulse_var}_to_{response_var}_{run_name}.png"
        save_path_target_to_pred = os.path.join(plot_dir, filename_target_to_pred)
        var_fig_target_to_pred.savefig(save_path_target_to_pred)
        logger.info(f"Saved Target->Predictor IRF plot to {save_path_target_to_pred}")
        plt.close(var_fig_target_to_pred)

    except InfeasibleTestError as infeasible_e:
         logger.error(f"VAR model fitting failed for {bivar_run_id} (InfeasibleTestError): {infeasible_e}. Check data for collinearity after differencing.")
    except Exception as e:
        logger.error(f"Bivariate VAR/IRF Analysis failed for {bivar_run_id}: {e}", exc_info=True)
    finally:
        # Ensure all potentially created figures are closed
        if var_fig_pred_to_target is not None and plt.fignum_exists(var_fig_pred_to_target.number): plt.close(var_fig_pred_to_target)
        if var_fig_self is not None and plt.fignum_exists(var_fig_self.number): plt.close(var_fig_self)
        if var_fig_target_to_pred is not None and plt.fignum_exists(var_fig_target_to_pred.number): plt.close(var_fig_target_to_pred)
        plt.close('all') # Cleanup general figures

def plot_lag_scatter(df: pd.DataFrame, target_col: str, predictor_col: str, lag: int,
                       title: str, filename: str, plot_dir: str):
    """
    Plots a scatter plot of the target variable vs. a lagged predictor variable.

    Args:
        df: DataFrame containing the data.
        target_col: Name of the target column (e.g., 'log_deaths').
        predictor_col: Name of the predictor column (e.g., 'hardship_sentiment').
        lag: The number of time steps (weeks) to lag the predictor.
        title: Plot title.
        filename: Name for the saved plot file.
        plot_dir: Directory to save the plot.
    """
    logger.info(f"Generating Lag Scatter plot: {title} (Lag={lag})")
    if target_col not in df.columns or predictor_col not in df.columns:
        logger.error(f"Lag Scatter fail '{filename}': Missing columns '{target_col}' or '{predictor_col}'.")
        return
    if df.empty:
        logger.warning(f"DataFrame empty for lag scatter plot '{filename}'. Skipping.")
        return
    if lag <= 0:
        logger.error(f"Lag must be positive for lag scatter plot. Got {lag}. Skipping '{filename}'.")
        return

    lag_scatter_fig = None # Initialize figure variable
    try:
        # Create a temporary DataFrame with the lagged predictor
        df_lagged = df[[target_col, predictor_col]].copy()
        lagged_predictor_col_name = f"{predictor_col}_lag{lag}"
        df_lagged[lagged_predictor_col_name] = df_lagged[predictor_col].shift(lag)

        # Drop NaNs introduced by shifting
        df_lagged.dropna(subset=[target_col, lagged_predictor_col_name], inplace=True)

        if df_lagged.empty:
             logger.warning(f"No data remaining after creating lag={lag} for '{predictor_col}'. Skipping scatter plot.")
             return

        # Create the scatter plot
        lag_scatter_fig = plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df_lagged, x=lagged_predictor_col_name, y=target_col,
                        alpha=0.2, s=10, edgecolor=None) # Same styling as previous scatter

        # Calculate correlation for the title
        try:
             # Use numpy for correlation after dropping NaNs
             correlation = np.corrcoef(df_lagged[target_col], df_lagged[lagged_predictor_col_name])[0, 1]
             plot_title = f"{title}\n({target_col} vs {predictor_col} at Lag {lag}w)\n(Correlation: {correlation:.2f})"
        except Exception as corr_err:
             logger.warning(f"Could not calculate correlation for lag scatter: {corr_err}")
             plot_title = f"{title}\n({target_col} vs {predictor_col} at Lag {lag}w)"


        plt.title(plot_title, fontsize=12) # Smaller fontsize maybe
        plt.xlabel(f"{predictor_col.replace('_', ' ').title()} (Lag {lag}w)")
        plt.ylabel(target_col.replace('_', ' ').title())
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path)
        logger.info(f"Saved lag scatter plot: {save_path}")

    except Exception as e:
        logger.error(f"Lag Scatter Plot fail '{filename}': {e}", exc_info=True)
    finally:
        if lag_scatter_fig is not None and plt.fignum_exists(lag_scatter_fig.number):
            plt.close(lag_scatter_fig)
        plt.close('all')

# -----------------------------------------------------------------------------
# 7. Interpretation & Granger Causality (Unchanged Functions)
# -----------------------------------------------------------------------------
def interpret_tft(model: TemporalFusionTransformer, val_dataloader: torch.utils.data.DataLoader,
                  plot_dir: str, run_name: str):
    """
    Calculates and saves TFT feature importance plots. Handles dictionary of plots.
    """
    logger.info(f"Calculating TFT feature importance for run '{run_name}'...")
    if model is None or val_dataloader is None or len(val_dataloader) == 0:
        logger.warning("Model/Dataloader missing, skip interpretation.")
        return

    matplotlib_imported = False
    # Initialize fig_imp to None or an empty list to store multiple figures if needed
    figures_to_close = []
    try:
        import matplotlib.pyplot as plt
        matplotlib_imported = True
    except ImportError:
        logger.warning("matplotlib not found or import error. Skipping interpretation plot generation.")

    try:
        interpret_device = "cpu"; model.to(interpret_device)
        logger.info(f"Running interpretation on: {interpret_device}")

        # Predict using mode="raw"
        prediction_output = model.predict(val_dataloader, mode="raw")
        logger.info(f"predict(mode='raw') output type: {type(prediction_output)}")

        raw_predictions_dict = None
        # --- MODIFIED UNPACKING LOGIC ---
        if isinstance(prediction_output, dict):
            raw_predictions_dict = prediction_output
        elif hasattr(prediction_output, 'to_dict') and callable(getattr(prediction_output, 'to_dict')):
            logger.info("Predict output is TupleOutputMixIn-like, calling .to_dict()")
            raw_predictions_dict = prediction_output.to_dict()
        elif hasattr(prediction_output, "_asdict") and callable(getattr(prediction_output, "_asdict")):
            logger.info("Predict output is namedtuple-like, converting via _asdict()")
            raw_predictions_dict = prediction_output._asdict()
        elif hasattr(prediction_output, "prediction"):
            logger.info("Predict output exposes 'prediction' attribute, building dict manually.")
            raw_predictions_dict = {"prediction": prediction_output.prediction}
        elif isinstance(prediction_output, (list, tuple)) and len(prediction_output) >= 1 and isinstance(prediction_output[0], dict):
            logger.warning("Predict() returned list/tuple, using first element for interpretation.")
            raw_predictions_dict = prediction_output[0]
        else:
            logger.error(f"Could not extract valid raw prediction dictionary. Type: {type(prediction_output)}. Skip interpretation.")
            return
        # --- END MODIFIED UNPACKING ---

        if 'prediction' not in raw_predictions_dict:
            logger.error("Key 'prediction' missing in raw_predictions_dict. Skip interpretation.")
            return

        raw_predictions_dict_cpu = {k: v.to(interpret_device) if isinstance(v, torch.Tensor) else v
                                    for k, v in raw_predictions_dict.items()}

        if matplotlib_imported:
            interpretation = model.interpret_output(raw_predictions_dict_cpu, reduction="mean")

            # plot_interpretation can return a single figure or a dictionary of figures
            interpretation_plots = model.plot_interpretation(interpretation) # Removed plot_type to get default(s)

            if isinstance(interpretation_plots, dict):
                for plot_name, fig_obj in interpretation_plots.items():
                    if hasattr(fig_obj, 'suptitle'): # Check if it's a figure object
                        fig_obj.suptitle(f"TFT Interpretation: {plot_name.replace('_', ' ').title()} ({run_name})")
                        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
                        save_path_imp = os.path.join(plot_dir, f"tft_interpretation_{plot_name}_{run_name}.png")
                        fig_obj.savefig(save_path_imp)
                        logger.info(f"Saved interpretation plot '{plot_name}' to {save_path_imp}")
                        figures_to_close.append(fig_obj) # Add to list for closing
                    else:
                        logger.warning(f"Item '{plot_name}' in interpretation_plots is not a figure object.")
            elif hasattr(interpretation_plots, 'suptitle'): # It's a single figure object
                interpretation_plots.suptitle(f"TFT Feature Importance ({run_name})")
                plt.tight_layout(rect=[0, 0.05, 1, 0.96])
                save_path_imp = os.path.join(plot_dir, f"tft_interpretation_importance_{run_name}.png")
                interpretation_plots.savefig(save_path_imp)
                logger.info(f"Saved interpretation plot to {save_path_imp}")
                figures_to_close.append(interpretation_plots)
            else:
                logger.warning(f"plot_interpretation did not return a recognized figure object or dict. Type: {type(interpretation_plots)}")
        else:
            logger.warning("Skipping interpretation plot generation as matplotlib could not be imported.")
        # === NEW: variable‑specific attention for hardship lag features ===
        try:
            encoder_sel = raw_predictions_dict.get("encoder_variable_selection")   # [batch, time_enc, vars]
            if encoder_sel is None:
                logger.info("No encoder_variable_selection in raw output → skip per‑feature attention.")
            else:
                enc_sel_mean = encoder_sel.mean(dim=0).detach().cpu().numpy()      # → [time_enc, vars]
                # get feature list from the validation TimeSeriesDataSet
                var_names = getattr(getattr(val_dataloader, "dataset", None), "reals", [])                                # list of real‑valued inputs
                hardship_lags = [
                    "hardship_sentiment_lag2w_std",
                    "hardship_sentiment_lag3w_std",
                    "hardship_sentiment_lag4w_std",
                    "hardship_sentiment_lag5w_std",
                    "hardship_sentiment_lag6w_std",
                    "hardship_sentiment_lag8w_std",
                ]
                hardship_lags = [v for v in hardship_lags if v in var_names]
                t_axis = np.arange(-enc_sel_mean.shape[0] + 1, 1)                   # weeks back

                for v in hardship_lags:
                    idx = var_names.index(v)
                    w   = enc_sel_mean[:, idx]                                      # length = encoder_len

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(t_axis, w)
                    ax.set(title=f"Attention for {v} (run {run_name})",
                        xlabel="Weeks back", ylabel="Attention weight")
                    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
                    out = os.path.join(PLOT_DIR, f"tft_attention_{v}_{run_name}.png")
                    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
                    logger.info(f"Saved: {out}")
        except Exception as e:
            logger.error(f"Variable‑specific attention failed: {e}", exc_info=True)
    except AttributeError as e: logger.error(f"AttributeError during interpretation: {e}.", exc_info=True)
    except Exception as e: logger.error(f"Error during interpretation: {e}", exc_info=True)
    finally:
        if matplotlib_imported:
            for fig_to_close in figures_to_close:
                if fig_to_close is not None and plt.fignum_exists(fig_to_close.number):
                    plt.close(fig_to_close)
            plt.close('all') # General cleanup
        model.to(DEVICE)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculates Symmetric Mean Absolute Percentage Error (SMAPE)."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    numerator = np.abs(predicted - actual)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    # Handle cases where both actual and prediction are near zero
    smape_mask = denominator > 1e-9 # Use a small threshold
    valid_smape_terms = numerator[smape_mask] / denominator[smape_mask]
    return np.mean(valid_smape_terms) * 100 if np.any(smape_mask) else 0.0

def naive_forecast(train_series: pd.Series, n_forecast: int) -> np.ndarray:
    """
    Naive forecast: last observed value is repeated.
    Args:
        train_series: pd.Series of historical values used for training the main model.
                      The baselines will use the end of this to "predict" the validation period.
        n_forecast: Number of steps to forecast ahead (length of validation set).
    Returns:
        Array of naive forecasts.
    """
    if train_series.empty:
        return np.full(n_forecast, np.nan)
    last_value = train_series.iloc[-1]
    return np.full(n_forecast, last_value)

def seasonal_naive_forecast(train_series: pd.Series, n_forecast: int, seasonality: int = 52) -> np.ndarray:
    """
    Seasonal Naive forecast: value from the same season in the last year.
    Args:
        train_series: pd.Series of historical values.
        n_forecast: Number of steps to forecast.
        seasonality: The seasonal period (e.g., 52 for weekly data with annual seasonality).
    Returns:
        Array of seasonal naive forecasts.
    """
    if len(train_series) < seasonality:
        logger.warning(f"Train series length ({len(train_series)}) is less than seasonality ({seasonality}). "
                       f"Falling back to Naive forecast for Seasonal Naive.")
        return naive_forecast(train_series, n_forecast)

    forecasts = np.empty(n_forecast)
    for i in range(n_forecast):
        # Index in the training series for the last known seasonal value
        # We need to go back `seasonality` steps from the point `i` steps *before* the forecast starts.
        # The last point of train_series is effectively index -1 relative to the start of forecast.
        # So for the first forecast point (i=0), we look at train_series.iloc[-seasonality].
        # For the second (i=1), we look at train_series.iloc[-seasonality+1], etc.
        # This means we need `seasonality - i` back from the end of the training series.
        # Or, more simply, `train_series.iloc[-(seasonality) + i % seasonality]` can work if train_series is long enough.
        # Let's use a simpler approach: for each step i, look back `seasonality` from the corresponding point
        # in the *previous cycle*.
        idx = len(train_series) - seasonality + (i % seasonality)
        if idx < 0: # Not enough history for this specific seasonal point
            # Fallback for very start if n_forecast > seasonality
            # This should ideally not happen if train_series is long enough relative to seasonality
            logger.debug(f"Seasonal Naive fallback for step {i} due to insufficient history for full cycle.")
            forecasts[i] = train_series.iloc[-seasonality + (i % seasonality)] # take from the first available cycle
        else:
            forecasts[i] = train_series.iloc[idx]
    return forecasts


def average_forecast(train_series: pd.Series, n_forecast: int) -> np.ndarray:
    """
    Average forecast: mean of the historical (training) data.
    Args:
        train_series: pd.Series of historical values.
        n_forecast: Number of steps to forecast.
    Returns:
        Array of average forecasts.
    """
    if train_series.empty:
        return np.full(n_forecast, np.nan)
    mean_value = train_series.mean()
    return np.full(n_forecast, mean_value)

def get_tft_predictions_and_actuals_orig_scale(
    model: TemporalFusionTransformer,
    dataloader: torch.utils.data.DataLoader,
    dataset: TimeSeriesDataSet # The validation TimeSeriesDataSet
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
    """
    Gets TFT predictions (p50) and actuals from the dataloader,
    transforms them to original 'deaths' scale, and aligns them by time_idx.

    Returns:
        Tuple: (
            index_df_for_plot: DataFrame with 'time_idx', 'week_date' for the validation period,
            actuals_orig_flat: pd.Series of actual deaths, indexed by validation time_idx,
            preds_orig_p50_flat: pd.Series of TFT p50 predicted deaths, indexed by validation time_idx
        )
    """
    logger.info("Getting TFT predictions and actuals for baseline comparison...")
    if model is None or dataloader is None or len(dataloader) == 0 or dataset is None:
        logger.error("Model, Dataloader or Dataset missing for getting TFT preds.")
        return None, None, None

    try:
        eval_device = next(model.parameters()).device
    except Exception:
        eval_device = torch.device(DEVICE)
        try: model.to(eval_device)
        except Exception: eval_device = torch.device("cpu"); model.to(eval_device)

    # Get predictions and actuals in log scale
    predictions_output = model.predict(dataloader, return_x=True, return_index=True, mode="quantiles")

    if not (isinstance(predictions_output, (list, tuple)) and len(predictions_output) == 3):
        if isinstance(predictions_output, dict) and 'prediction' in predictions_output and 'x' in predictions_output and 'index' in predictions_output:
            raw_preds_tensor, x_output, index_df_val = predictions_output['prediction'], predictions_output['x'], predictions_output['index']
        else:
            logger.error("TFT predict output format not recognized. Cannot extract data for baselines.")
            return None, None, None
    else:
        raw_preds_tensor, x_output, index_df_val = predictions_output

    if not isinstance(raw_preds_tensor, torch.Tensor) or raw_preds_tensor.ndim != 3 or \
       not isinstance(x_output, dict) or 'decoder_target' not in x_output or \
       not isinstance(index_df_val, pd.DataFrame) or 'time_idx' not in index_df_val.columns:
        logger.error("Problem with unpacked TFT prediction components.")
        return None, None, None

    preds_log_p50_tensor = raw_preds_tensor[:, :, 1].cpu() # Median (p50)
    actuals_log_tensor = x_output['decoder_target'].cpu()   # These are 'log_deaths'

    # Flatten and convert to numpy
    preds_log_p50_flat = preds_log_p50_tensor.flatten().numpy()
    actuals_log_flat = actuals_log_tensor.flatten().numpy()

    # Align with time_idx from index_df_val
    # index_df_val contains time_idx for EACH forecast step of EACH sample in the validation set.
    # We need to ensure it aligns with the flattened predictions.
    # The number of forecast steps is model.max_prediction_length
    # The number of samples in val_dataloader is len(dataloader.dataset) -> len(dataset)

    # We need to map these back to the original dataframe's time_idx and 'deaths'
    # `index_df_val` gives the starting `time_idx` of each prediction sequence.
    # We need to get the full sequence of `time_idx` for all predicted points.

    all_val_time_indices = []
    for start_idx in index_df_val['time_idx']:
        all_val_time_indices.extend(
            range(start_idx, start_idx + dataset.max_prediction_length)
        )

    if len(all_val_time_indices) != len(actuals_log_flat):
        logger.warning(f"Length mismatch between expanded time_idx ({len(all_val_time_indices)}) "
                       f"and actuals ({len(actuals_log_flat)}). Taking minimum length.")
        min_len = min(len(all_val_time_indices), len(actuals_log_flat), len(preds_log_p50_flat))
        all_val_time_indices = all_val_time_indices[:min_len]
        actuals_log_flat = actuals_log_flat[:min_len]
        preds_log_p50_flat = preds_log_p50_flat[:min_len]

    # Create Series for easier manipulation and merging
    actuals_log_series = pd.Series(actuals_log_flat, index=all_val_time_indices).sort_index()
    preds_log_p50_series = pd.Series(preds_log_p50_flat, index=all_val_time_indices).sort_index()

    # Inverse transform to original 'deaths' scale
    actuals_orig_series = np.expm1(actuals_log_series)
    preds_orig_p50_series = np.maximum(0, np.expm1(preds_log_p50_series)) # Ensure non-negative

    # Deduplicate if overlapping windows created same time_idx multiple times (take mean for preds)
    # Actuals should be unique if dataloader setup correctly for val.
    actuals_orig_series = actuals_orig_series[~actuals_orig_series.index.duplicated(keep='first')]
    preds_orig_p50_series = preds_orig_p50_series.groupby(preds_orig_p50_series.index).mean() # Average preds for same time_idx

    # Align them to ensure they cover the exact same time indices
    common_index = actuals_orig_series.index.intersection(preds_orig_p50_series.index)
    actuals_orig_final = actuals_orig_series.loc[common_index]
    preds_orig_p50_final = preds_orig_p50_series.loc[common_index]


    # Get week_date for plotting — fall back gracefully if it is missing
    if hasattr(dataset, "data") and "time_idx" in dataset.data and "week_date" in dataset.data:
        try:
            mapping_df = pd.DataFrame({
                "time_idx": dataset.data["time_idx"],
                "week_date": pd.to_datetime(dataset.data["week_date"])
            })
            mapping_df = mapping_df.drop_duplicates(subset=["time_idx"]).set_index("time_idx")
            index_df_for_plot = (
                mapping_df
                  .reindex(common_index)  # align to validation indices
                  .reset_index()
            )
        except Exception as map_err:
            logger.warning(f"Could not build time_idx→week_date mapping: {map_err}. Using time_idx only.")
            index_df_for_plot = pd.DataFrame({"time_idx": common_index})
    else:
        logger.warning("dataset.data lacks 'time_idx' or 'week_date'; using time_idx only for plots.")
        index_df_for_plot = pd.DataFrame({"time_idx": common_index})


    logger.info(f"Successfully extracted and aligned TFT actuals/predictions. Count: {len(actuals_orig_final)}")
    return index_df_for_plot, actuals_orig_final, preds_orig_p50_final

def evaluate_against_baselines(
    tft_model: TemporalFusionTransformer,
    tft_val_dataloader: torch.utils.data.DataLoader,
    tft_val_dataset: TimeSeriesDataSet,
    val_index_df_with_dates: pd.DataFrame, # You added this, good
    full_df_cropped: pd.DataFrame,
    min_overall_time_idx: int,             # <<< ADDED
    min_overall_week_date: datetime,       # <<< ADDED
    plot_dir: str,
    run_name: str,
    seasonality_period: int = 52
):
    """
    Evaluates TFT against Naive, Seasonal Naive, and Average baselines.
    Plots forecasts and reports MAE, MSE, SMAPE metrics on original 'deaths' scale.

    Args:
        tft_model: Trained TFT model.
        tft_val_dataloader: Dataloader for the validation set for TFT.
        tft_val_dataset: The TimeSeriesDataSet for validation.
        full_df_cropped: The DataFrame from which training and validation data were split.
                         Must contain 'time_idx' and 'deaths'.
        plot_dir: Directory to save plots.
        run_name: Name for the run (used in plot titles/filenames).
        seasonality_period: Seasonality for Seasonal Naive forecast.
    """
    logger.info(f"--- Evaluating TFT vs Baselines for Run: {run_name} ---")

    # --- 1. Get TFT predictions and actuals for the validation period ---
    index_df_plot, actuals_val_orig, tft_preds_val_orig = get_tft_predictions_and_actuals_orig_scale(
        tft_model, tft_val_dataloader, tft_val_dataset
    )

    if actuals_val_orig is None or tft_preds_val_orig is None or index_df_plot is None:
        logger.error("Failed to get TFT predictions/actuals. Aborting baseline comparison.")
        return

    if actuals_val_orig.empty:
        logger.error("No actual values retrieved for validation. Aborting baseline comparison.")
        return

    # ------------------------------------------------------------------
    # OPTIONAL: drop the final ISO week‑53 spike (often an artificial tail
    # week in yearly aggregates). We identify it as the *last* validation
    # point, provided its ISO week == 53 or it is the max time_idx.
    # ------------------------------------------------------------------
    try:
        # Determine candidate indices to drop
        drop_mask = pd.Series(False, index=actuals_val_orig.index)
        # If week_date is available – use ISO calendar week == 53
        if 'week_date' in index_df_plot.columns and not index_df_plot['week_date'].isnull().all():
            iso_weeks = pd.to_datetime(index_df_plot.set_index('time_idx')['week_date']).dt.isocalendar().week
            drop_mask |= (iso_weeks == 53)
        # Always consider the single last point a spike candidate
        drop_mask.loc[actuals_val_orig.index.max()] = True
        # Apply mask if at least one point flagged
        if drop_mask.any():
            logger.info(f"Excluding {drop_mask.sum()} week‑53/last‑week point(s) from plots & metrics.")
            # Filter all validation‑aligned objects
            actuals_val_orig = actuals_val_orig.loc[~drop_mask]
            tft_preds_val_orig = tft_preds_val_orig.loc[~drop_mask]
            index_df_plot = index_df_plot[~index_df_plot['time_idx'].isin(drop_mask[drop_mask].index)]
    except Exception as wk53_err:
        logger.warning(f"Week‑53 filtering failed (continuing without filter): {wk53_err}")

    # --- 2. Prepare data for baseline models ---
    # Baselines need "training" data up to the start of the validation period.
    # The validation period starts at the first time_idx in `actuals_val_orig.index`
    first_val_time_idx = actuals_val_orig.index.min()

    # Ensure full_df_cropped is sorted and has 'time_idx' and 'deaths'
    if 'time_idx' not in full_df_cropped.columns or 'deaths' not in full_df_cropped.columns:
        logger.error("full_df_cropped is missing 'time_idx' or 'deaths'. Cannot run baselines.")
        return
    
    # Ensure full_df_cropped is sorted by time_idx
    full_df_cropped = full_df_cropped.sort_values(by='time_idx').set_index('time_idx')


    train_data_for_baselines = full_df_cropped.loc[full_df_cropped.index < first_val_time_idx, 'deaths'].dropna()

    if train_data_for_baselines.empty:
        logger.error("No training data available for baselines (all data might be in validation).")
        return

    n_forecast_steps = len(actuals_val_orig)

    # --- 3. Generate Baseline Forecasts ---
    logger.info("Generating baseline forecasts...")
    naive_preds = naive_forecast(train_data_for_baselines, n_forecast_steps)
    snaive_preds = seasonal_naive_forecast(train_data_for_baselines, n_forecast_steps, seasonality=seasonality_period)
    avg_preds = average_forecast(train_data_for_baselines, n_forecast_steps)

    # Align baseline predictions with the validation index for consistent metric calculation and plotting
    naive_preds_series = pd.Series(naive_preds, index=actuals_val_orig.index)
    snaive_preds_series = pd.Series(snaive_preds, index=actuals_val_orig.index)
    avg_preds_series = pd.Series(avg_preds, index=actuals_val_orig.index)


    # --- 4. Calculate Metrics ---
    metrics_summary = {}
    models_to_evaluate = {
        "TFT": tft_preds_val_orig,
        "Naive": naive_preds_series,
        "Seasonal Naive": snaive_preds_series,
        "Average": avg_preds_series
    }

    for model_name, preds in models_to_evaluate.items():
        if preds is None or (isinstance(preds, pd.Series) and preds.isnull().all()) or \
           (isinstance(preds, np.ndarray) and np.isnan(preds).all()):
            logger.warning(f"Predictions for {model_name} are all NaNs. Skipping metrics.")
            metrics_summary[model_name] = {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan, "RMSE": np.nan}
            continue

        # Ensure actuals and preds are aligned and have the same length for metrics
        common_idx_metric = actuals_val_orig.index.intersection(preds.index)
        actuals_metric = actuals_val_orig.loc[common_idx_metric].values
        preds_metric = preds.loc[common_idx_metric].values
        
        if len(actuals_metric) == 0:
            logger.warning(f"No common data points for metrics for {model_name}. Skipping.")
            metrics_summary[model_name] = {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan, "RMSE": np.nan}
            continue


        mae = mean_absolute_error(actuals_metric, preds_metric)
        mse = mean_squared_error(actuals_metric, preds_metric)
        rmse = np.sqrt(mse)
        smape = calculate_smape(actuals_metric, preds_metric)
        metrics_summary[model_name] = {"MAE": mae, "MSE": mse, "SMAPE": smape, "RMSE": rmse}
        logger.info(f"Metrics for {model_name} ({run_name}): "
                    f"MAE={mae:.2f}, MSE={mse:.2f}, RMSE={rmse:.2f}, SMAPE={smape:.2f}%")

    metrics_df = pd.DataFrame(metrics_summary).T
    logger.info(f"\n--- Metrics Summary ({run_name}) ---\n{metrics_df}\n-----------------------------")
    try:
        metrics_df.to_csv(os.path.join(plot_dir, f"baseline_comparison_metrics_{run_name}.csv"))
        logger.info(f"Saved baseline metrics summary to CSV for {run_name}.")
    except Exception as e_csv:
        logger.warning(f"Could not save baseline metrics CSV: {e_csv}")


    # --- 5. Generate Plots ---
    if 'week_date' in index_df_plot.columns and not index_df_plot['week_date'].isnull().all():
        x_axis_data_for_plot = index_df_plot['week_date']
        x_axis_is_datetime = True
        x_plot_label = "Week Date (Validation Period)"
    else: # Fallback if week_date is missing or all NaNs
        logger.warning("week_date missing or all NaNs in index_df_plot. Plotting against time_idx.")
        x_axis_data_for_plot = index_df_plot['time_idx']
        x_axis_is_datetime = False
        x_plot_label = "Time Index (Validation Period)"

    # Determine the number of points to plot
    # For full validation range:
    num_plot_points_to_show = len(x_axis_data_for_plot)
    # Or to show a fixed number like before:
    # num_plot_points_to_show = min(len(x_axis_data_for_plot), 150)
    
    # Ensure actuals and preds are sliced consistently
    actuals_to_plot = actuals_val_orig.iloc[:num_plot_points_to_show]
    
    fig_comp, ax_comp = plt.subplots(figsize=(20, 8)) # Wider figure
    ax_comp.plot(x_axis_data_for_plot.iloc[:num_plot_points_to_show],
                 actuals_to_plot,
                 label="Actual Deaths", color='black', linewidth=1.5, marker='.', markersize=4, zorder=5)
    
    plot_styles = {
        "TFT": {'color': 'red', 'linestyle': '-', 'linewidth': 1.2},
        "Naive": {'color': 'blue', 'linestyle': '--', 'linewidth': 1.0},
        "Seasonal Naive": {'color': 'green', 'linestyle': ':', 'linewidth': 1.0},
        "Average": {'color': 'purple', 'linestyle': '-.', 'linewidth': 1.0}
    }

    for model_name, preds_series in models_to_evaluate.items():
        if preds_series is not None and not preds_series.isnull().all():
            style = plot_styles.get(model_name, {})
            # preds_series should already be aligned with actuals_val_orig's index
            preds_to_plot = preds_series.iloc[:num_plot_points_to_show]
            ax_comp.plot(x_axis_data_for_plot.iloc[:num_plot_points_to_show],
                         preds_to_plot,
                         label=f"{model_name} Forecast", **style, zorder=3 if model_name != "TFT" else 4)

    title_text = f"TFT vs Baseline Forecasts ({run_name}, Original Scale)"
    if num_plot_points_to_show < len(x_axis_data_for_plot):
        title_text += f" - First {num_plot_points_to_show} points"
    ax_comp.set_title(title_text, fontsize=14)
    ax_comp.set_ylabel("Weekly Deaths", fontsize=12)
    ax_comp.legend(fontsize=10, loc='upper left')
    ax_comp.grid(True, linestyle=':', alpha=0.7)

    if x_axis_is_datetime:
        import matplotlib.dates as mdates
        ax_comp.set_xlabel(x_plot_label, fontsize=12)
        # Auto-format dates; adjust locator and formatter for density
        num_major_ticks = 10 # Aim for about 10 major ticks
        if num_plot_points_to_show <= num_major_ticks * 2 : # If few points, show all or more frequently
            ax_comp.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=num_plot_points_to_show//2, maxticks=num_plot_points_to_show))
            ax_comp.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%W%U')) # Year-Week
        else: # More points, space out ticks
            locator_interval = max(1, num_plot_points_to_show // (num_major_ticks * 7)) # roughly weekly if many years
            if num_plot_points_to_show / 52 > 5 : # If more than 5 years, maybe monthly or quarterly ticks
                 ax_comp.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, (num_plot_points_to_show // 52) // 4 ))) # ~Quarterly for many years
                 ax_comp.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            else:
                 ax_comp.xaxis.set_major_locator(mdates.WeekdayLocator(interval=max(1, num_plot_points_to_show // (num_major_ticks * 2) ))) # ~Bi-weekly to Monthly
                 ax_comp.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%W%U'))

        fig_comp.autofmt_xdate(rotation=30, ha='right')
    else: # x_axis is time_idx - use convert_time_idx_to_date_labels
        # You need min_overall_time_idx, min_overall_week_date passed here too
        if 'min_overall_time_idx' in locals() and 'min_overall_week_date' in locals(): # Check if passed
            tick_pos, tick_lab = convert_time_idx_to_date_labels(
                x_axis_data_for_plot.iloc[:num_plot_points_to_show],
                min_overall_time_idx, min_overall_week_date,
                label_type="year_week",
                tick_frequency=max(1, num_plot_points_to_show // 10)
            )
            if len(tick_pos)>0: ax_comp.set_xticks(tick_pos); ax_comp.set_xticklabels(tick_lab, rotation=45, ha='right')
        ax_comp.set_xlabel(x_plot_label, fontsize=12)


    plt.tight_layout()
    # --- Save and close the comparison figure ---
    try:
        save_path_comp = os.path.join(plot_dir, f"baseline_forecast_comparison_{run_name}.png")
        safe_name = run_name.strip() or "run"
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path_comp = os.path.join(
                plot_dir,
                f"baseline_forecast_comparison_{safe_name}.png"
        )
        fig_comp.savefig(save_path_comp, bbox_inches="tight")
        logger.info(f"Saved baseline comparison plot to {save_path_comp}")
        # ------------------------------------------------------------------
        # Extra diagnostics: residual time‑series, actual‑vs‑predicted scatter,
        # and residuals vs. lagged hardship predictors
        # ------------------------------------------------------------------
        try:
            # === 1) Residual time‑series plot ===
            tft_residuals = actuals_val_orig - tft_preds_val_orig
            if not tft_residuals.isnull().all():
                fig_res_ts, ax_res_ts = plt.subplots(figsize=(18, 5))
                ax_res_ts.plot(x_axis_data_for_plot.iloc[:num_plot_points_to_show],
                               tft_residuals.iloc[:num_plot_points_to_show],
                               label='TFT Forecast Residuals', color='teal', marker='o',
                               linestyle='-', markersize=3)
                ax_res_ts.axhline(0, color='red', linestyle='--', linewidth=1)
                ax_res_ts.set_title(f'TFT Forecast Residuals Over Time ({run_name})', fontsize=14)
                ax_res_ts.set_xlabel(x_plot_label, fontsize=12)
                ax_res_ts.set_ylabel('Residual (Actual − Predicted)', fontsize=12)
                ax_res_ts.legend(); ax_res_ts.grid(True)
                if x_axis_is_datetime:
                    fig_res_ts.autofmt_xdate()
                plt.tight_layout()
                res_ts_path = os.path.join(plot_dir, f"tft_residuals_timeseries_{safe_name}_.png")
                fig_res_ts.savefig(res_ts_path, bbox_inches="tight"); plt.close(fig_res_ts)
                logger.info(f"Saved residual time‑series plot to {res_ts_path}")
            # === 2) Actual vs. Predicted scatter ===
            fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))
            ax_scatter.scatter(actuals_val_orig.values, tft_preds_val_orig.values,
                               alpha=0.3, s=15, label='Predictions')
            min_val = float(min(actuals_val_orig.min(), tft_preds_val_orig.min()))
            max_val = float(max(actuals_val_orig.max(), tft_preds_val_orig.max()))
            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Forecast (y=x)')
            ax_scatter.set_xlabel('Actual Deaths (Original Scale)', fontsize=12)
            ax_scatter.set_ylabel('TFT Predicted Median Deaths', fontsize=12)
            ax_scatter.set_title(f'TFT: Actual vs. Predicted ({run_name})', fontsize=14)
            ax_scatter.set_aspect('equal', adjustable='box'); ax_scatter.grid(True); ax_scatter.legend()
            plt.tight_layout()
            scatter_path = os.path.join(plot_dir, f"tft_actual_vs_predicted_scatter_{safe_name}_.png")
            fig_scatter.savefig(scatter_path, bbox_inches="tight"); plt.close(fig_scatter)
            logger.info(f"Saved actual‑vs‑predicted scatter to {scatter_path}")

            # === 3) Residuals vs. lagged hardship predictors ===
            # Identify available hardship lag cols ( *_lagXw_std )
            hardship_lag_cols = [c for c in full_df_cropped.columns if c.startswith('hardship_sentiment_lag') and c.endswith('_std')]

            if hardship_lag_cols:
                # `full_df_cropped` should ALREADY have 'time_idx' as its index
                # from the preparation step for baseline models earlier in this function.
                # So, we can directly use it. Let's ensure it's the case or handle.
                if not isinstance(full_df_cropped.index, pd.RangeIndex) and full_df_cropped.index.name == 'time_idx':
                    full_df_idxed_for_preds = full_df_cropped # Already indexed by time_idx
                elif 'time_idx' in full_df_cropped.columns: # If somehow it got reset to a column
                    full_df_idxed_for_preds = full_df_cropped.set_index('time_idx')
                else:
                    logger.error("Cannot find 'time_idx' as index or column in full_df_cropped for predictor alignment. Skipping residuals vs predictors plot.")
                    # Optionally raise an error or return
                    raise ValueError("time_idx setup issue for residuals vs predictors plot")


                for lag_col_name in hardship_lag_cols: # Changed variable name for clarity
                    if lag_col_name in full_df_idxed_for_preds.columns:
                        # tft_residuals is a Series indexed by time_idx (same as actuals_val_orig.index)
                        # We need to align predictor values from full_df_idxed_for_preds with these residual time_idx
                        
                        # Get the predictor values corresponding to the time_idx of the residuals
                        # This ensures we only take predictor values for which we have residuals
                        common_indices_for_rvp = tft_residuals.index.intersection(full_df_idxed_for_preds.index)
                        
                        if common_indices_for_rvp.empty:
                            logger.warning(f"No common time_idx between residuals and predictor '{lag_col_name}'. Skipping plot.")
                            continue

                        predictor_values_aligned = full_df_idxed_for_preds.loc[common_indices_for_rvp, lag_col_name]
                        residuals_aligned = tft_residuals.loc[common_indices_for_rvp]

                        if predictor_values_aligned.empty or residuals_aligned.empty:
                            logger.warning(f"Empty series after alignment for predictor '{lag_col_name}'. Skipping plot.")
                            continue
                        
                        # Drop any NaNs that might have arisen from partial loc
                        combined_df_rvp = pd.DataFrame({'predictor': predictor_values_aligned, 'residual': residuals_aligned}).dropna()
                        if combined_df_rvp.empty:
                            logger.warning(f"Empty data after dropna for predictor '{lag_col_name}'. Skipping plot.")
                            continue

                        fig_rvp, ax_rvp = plt.subplots(figsize=(6, 6))
                        ax_rvp.scatter(combined_df_rvp['predictor'], combined_df_rvp['residual'], alpha=0.3, s=15)
                        ax_rvp.axhline(0, color='red', linestyle='--', linewidth=1)
                        ax_rvp.set_xlabel(lag_col_name, fontsize=11) # Use lag_col_name
                        ax_rvp.set_ylabel('Residual (Actual − Predicted)', fontsize=11)
                        ax_rvp.set_title(f'Residuals vs {lag_col_name} ({run_name})', fontsize=12) # Use lag_col_name
                        ax_rvp.grid(True)
                        plt.tight_layout()
                        rvp_path = os.path.join(plot_dir, f"residuals_vs_{lag_col_name.replace('/','_')}_{safe_name}_.png") # Sanitize col name for filename
                        fig_rvp.savefig(rvp_path, bbox_inches="tight"); plt.close(fig_rvp)
                        logger.info(f"Saved residuals‑vs‑{lag_col_name} scatter to {rvp_path}")
                    else:
                        logger.warning(f"Predictor column '{lag_col_name}' not found in full_df_idxed_for_preds. Skipping plot.")
            else:
                logger.info("No hardship lag columns found for 'Residuals vs. predictors' plots.")
        except Exception as diag_err:
            logger.error(f"Extra diagnostic plotting failed: {diag_err}")
        finally:
            plt.close('all')
    except Exception as save_err:
        logger.error(f"Could not save baseline comparison plot: {save_err}")
    finally:
        if fig_comp is not None and plt.fignum_exists(fig_comp.number):
            plt.close(fig_comp)
        plt.close('all')
# Make sure InfeasibleTestError is imported if not already done
from statsmodels.tools.sm_exceptions import InfeasibleTestError

def run_granger_causality(df: pd.DataFrame, var1: str, var2: str, max_lag: int = 12) -> Optional[Dict[int, float]]:
    """
    Performs Granger Causality tests using first differences.
    Attempts test on levels if differenced series is constant (with warning).
    Returns {lag: p_value}.
    """
    logger.info(f"Running Granger Causality test: '{var1}' -> '{var2}'? (Max lag: {max_lag} weeks)")
    if var1 not in df.columns or var2 not in df.columns:
        logger.error(f"Granger columns missing ('{var1}' or '{var2}'). Skipping test.")
        return None

    data = df[[var1, var2]].copy()
    # Drop rows with NaNs *before* checking for constant values
    data.dropna(inplace=True)

    if data.shape[0] < max_lag + 5: # Check after dropping NaNs
        logger.error(f"Not enough data ({data.shape[0]}) for Granger test after dropna. Skipping test.")
        return None

    # Check for constant columns in the original data
    if (data[var1].nunique() <= 1) or (data[var2].nunique() <= 1):
        logger.error(f"Input data for '{var1}' or '{var2}' is constant before differencing. Granger test impossible. Skipping test.")
        return None

    # Proceed with differencing
    try:
        data_diff = data.diff().dropna()
    except Exception as e:
        logger.error(f"Granger differencing error: {e}. Skipping test.")
        return None

    if data_diff.shape[0] < max_lag + 5: # Check after differencing
        logger.error(f"Not enough data ({data_diff.shape[0]}) after differencing. Skipping test.")
        return None

    # Check for constant columns *after* differencing
    test_on_diff = True
    if (data_diff[var1].nunique() <= 1) or (data_diff[var2].nunique() <= 1):
        logger.warning(f"Constant column found after differencing for '{var1}' or '{var2}'.")
        # Attempt test on levels as fallback
        logger.warning(f"Attempting Granger test on raw levels for '{var1}' -> '{var2}' (use results with caution).")
        test_on_diff = False
        test_data = data[[var2, var1]] # Use original data
        if test_data.shape[0] < max_lag + 5: # Re-check length for levels data
             logger.error("Not enough data for levels test. Skipping.")
             return None
    else:
        logger.info("Using first differences for Granger test.")
        test_data = data_diff[[var2, var1]] # Use differenced data


    # Perform the test
    try:
        gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
        p_values = {lag: gc_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)}

        # Log results clearly
        significant_lags = [lag for lag, p in p_values.items() if p < 0.05]
        test_type = "(Differences)" if test_on_diff else "(Levels - CAUTION!)"
        if significant_lags:
            logger.info(f" Granger Result ('{var1}' -> '{var2}') {test_type}: Significant at lags: {significant_lags} (p < 0.05).")
        else:
            logger.info(f" Granger Result ('{var1}' -> '{var2}') {test_type}: Not significant up to lag {max_lag} (p >= 0.05).")
        return p_values

    except InfeasibleTestError as e: # Catch specific error
         logger.error(f"Granger test error for '{var1}' -> '{var2}' (Infeasible): {e}. Likely constant values even in levels.")
         return None
    except Exception as e:
        logger.error(f"Granger test error for '{var1}' -> '{var2}': {e}", exc_info=True)
        return None

@memory.cache
def calculate_sentiment_scores_dataframe(df: pd.DataFrame, text_col: str = "text", batch_size: int = 32) -> pd.DataFrame:
    """Calculates multiple MacBERTh sentiment scores for the text column."""
    logger.info(f"Calculating multiple MacBERTh sentiment scores for '{text_col}'...")
    if text_col not in df.columns: raise ValueError(f"Column '{text_col}' not found.")

    df_copy = df.copy()
    df_copy[text_col] = df_copy[text_col].astype(str).fillna('') # Ensure string type

    scorer = MacBERThSentimentScorer() # Uses the class defined above
    texts_to_score = df_copy[text_col].tolist()

    # Get the dictionary of scores {score_name: [list_of_scores]}
    sentiment_scores_dict = scorer.calculate_sentiment_scores(texts_to_score, batch_size=batch_size)

    # Add each score list as a new column
    score_cols_added = []
    for score_name, scores_list in sentiment_scores_dict.items():
        if scores_list: # Only add if scores were calculated
             df_copy[score_name] = scores_list
             score_cols_added.append(score_name) # Keep track of added columns
             logger.info(f"Added sentiment scores for '{score_name}'. Stats: Min={np.min(scores_list):.3f}, Max={np.max(scores_list):.3f}, Mean={np.mean(scores_list):.3f}, Std={np.std(scores_list):.3f}")
        else:
             logger.warning(f"No scores calculated for '{score_name}', column not added.")

    logger.info("Multiple sentiment scoring complete.")
    # Return identifier columns PLUS the newly added score columns
    id_cols = ['week_date', 'doc_id', 'trial_id'] # Keep identifiers
    # Ensure we only select columns that actually exist in the df_copy
    final_cols = [col for col in id_cols + score_cols_added if col in df_copy.columns]
    return df_copy[final_cols]

def plot_granger_causality_results(p_values_dict: Dict[str, Optional[Dict[int, float]]], title_prefix: str, plot_dir: str):
    """Plots the p-values from Granger Causality tests."""
    if not p_values_dict: logger.warning("No Granger results to plot."); return
    valid_results = {k: v for k, v in p_values_dict.items() if v is not None}
    num_tests = len(valid_results)
    if num_tests == 0: logger.warning("No valid Granger results to plot."); return
    fig, axes = plt.subplots(nrows=num_tests, ncols=1, figsize=(10, 4 * num_tests), squeeze=False); axes = axes.flatten()
    fig.suptitle(f"{title_prefix} Granger Causality P-Values (Weekly Lags)", fontsize=14, y=1.02)
    for i, (test_description, p_values) in enumerate(valid_results.items()):
        ax = axes[i]; lags = list(p_values.keys()); pvals = list(p_values.values())
        bars = ax.bar(lags, pvals, color='lightblue', edgecolor='black'); ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='p = 0.05 Threshold')
        for lag_idx, p in enumerate(pvals):
            if p < 0.05: bars[lag_idx].set_color('salmon'); bars[lag_idx].set_edgecolor('red')
        ax.set_title(f"Test: {test_description}"); ax.set_xlabel("Lag (Weeks)"); ax.set_ylabel("P-value"); ax.set_xticks(lags); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    filename = f"{title_prefix.lower().replace(' ','_')}_granger_causality_weekly.png"
    save_path = os.path.join(plot_dir, filename); plt.savefig(save_path); logger.info(f"Saved Granger plot to {save_path}"); plt.close(fig)

def main():
    """
    (REVISED V2 - NO SMOOTHING in pipeline)
    Runs analysis pipeline: parses data, calculates hardship sentiment, 
    aggregates metrics weekly, performs lagging & standardization on UNSMOOTHED data,
    trains/evaluates TFT, runs Granger causality on UNSMOOTHED data.
    Optionally plots smoothed trends for visualization only.
    """
    logger.info("--- REVISED V2 (NO SMOOTHING): Starting Historical Analysis Script ---")
    script_start_time = time.time()
    final_df_cropped = None # Initialize
    best_model = None; trainer = None; val_dl = None; val_ds = None # Initialize TFT vars

    # --- Configuration ---
    # Define rolling correlation window HERE
    rolling_window_years = 5
    rolling_window_weeks = rolling_window_years * 52
    logger.info(f"Using rolling correlation window: {rolling_window_years} years ({rolling_window_weeks} weeks)")

    # Define key lags based on potential analysis needs (can be adjusted)
    hardship_lags_list = [2, 4, 5, 6, 8] # Lags to create for hardship
    property_crime_lags_list = [4] # Lags to create for property crime
    logger.info(f"Selected lags to create: Hardship={hardship_lags_list}, PropertyCrime={property_crime_lags_list}")

    # Define lags dictionary for the aggregation function (using BASE names)
    lags_for_aggregation = {
        'hardship_sentiment': hardship_lags_list,
        'property_crime_prop': property_crime_lags_list,
        # Add other base columns if you want their lags created e.g.
        # 'conviction_rate': [4],
    }

    # Define smoothing window JUST for visualization plots if desired
    VISUAL_SMOOTHING_WINDOW = 8 

    try:
        os.makedirs(PLOT_DIR, exist_ok=True); logger.info(f"Plots directory: {PLOT_DIR}")

        ANALYSIS_START_DATE = '1719-01-01'
        ANALYSIS_END_DATE = '1829-12-31'
        logger.info(f"*** Analysis Period: {ANALYSIS_START_DATE} to {ANALYSIS_END_DATE} ***")

        # --- Step 1: Parse Old Bailey Data ---
        logger.info("--- Step 1: Parsing/Loading Old Bailey Sessions Papers ---")
        ob_df_parsed_full = parse_old_bailey_papers(ob_dir=OLD_BAILEY_DIR, start_year=START_YEAR, end_year=END_YEAR)
        if ob_df_parsed_full.empty or 'text' not in ob_df_parsed_full.columns or 'trial_id' not in ob_df_parsed_full.columns:
            raise ValueError("Step 1 Failed.")
        logger.info(f"Parsed/Loaded {len(ob_df_parsed_full)} trial accounts.")

        # --- Step 2: Calculate Hardship Sentiment Scores ---
        logger.info("--- Step 2: Calculating/Loading Hardship Sentiment Scores per Trial ---")
        import joblib
        # direct load from explicit safekeeping path if available
        safe_cache = "/Users/sebo/Desktop/AUC/Semester 6/Capstone/Seb Olsen Capstone Repo/Fear-and-Mortality-Capstone/sentimentanalysisoutput.pkl"
        if os.path.exists(safe_cache):
            logger.info(f"Loading sentiment scores from safekeeping cache file: {safe_cache}")
            ob_df_sentiment_scores = joblib.load(safe_cache)
            # --- Attach original text and run qualitative validation/sampling ---
            if not ob_df_sentiment_scores.empty and 'trial_id' in ob_df_sentiment_scores.columns:
                ob_df_sentiment_scores = ob_df_sentiment_scores.merge(
                    ob_df_parsed_full[['trial_id', 'text']], on='trial_id', how='left'
                )
                validate_trial_sentiment_scores(
                    ob_df_sentiment_scores,
                    text_col_original="text",
                    text_col_processed="processed_text" if 'processed_text' in ob_df_sentiment_scores.columns else "text",
                    n_samples_spot_check=15,
                    min_text_length=30,
                )
        # # direct load from known cache path using sentiment_cache_dir
        # ob_df_sentiment_scores = calculate_sentiment_scores_dataframe(ob_df_parsed_full, text_col='text', batch_size=BATCH_SIZE // 2)
        # if ob_df_sentiment_scores.empty or 'hardship_sentiment' not in ob_df_sentiment_scores.columns:
        #     logger.warning("Step 2 Warning: Hardship sentiment scoring failed or DF empty.")
        #     ob_df_sentiment_scores = pd.DataFrame(columns=['week_date', 'doc_id', 'trial_id', 'hardship_sentiment'])
        # else: logger.info(f"Hardship sentiment scores calculated/loaded.")

        # -----------------------------------------------------------
        # Qualitative validation + sampling of hardship sentiment
        # -----------------------------------------------------------
        

        # --- Step 2.1 & 2.5: Diagnostics & Validation (Trial Level) ---
        # (Keep the diagnostic code block from previous step here - it operates before aggregation)
        # Ensure it plots the trial-level hardship distribution and logs correlation/vector info
        logger.info("--- Step 2.5: Preparing & Running Trial-Level Diagnostics ---")
        if not ob_df_sentiment_scores.empty and 'hardship_sentiment' in ob_df_sentiment_scores.columns:
             # --- (Include the diagnostic code block from previous response here) ---
             # It should:
             # 1. Merge sentiment with text from ob_df_parsed_full
             # 2. Filter by word count
             # 3. Plot trial-level hardship distribution
             # 4. Run validate_trial_sentiment_scores (for spot checks/keyword corr on hardship)
             # 5. Log the reference vector check (will show only hardship is present)
            pass # Placeholder: Add the full diagnostic block code here
        else: logger.warning("Skipping trial-level diagnostics as sentiment scores are missing.")
        # --- (End of Diagnostic Block Placeholder) ---

        # --- Step 3: Load Weekly Mortality ---
        logger.info("--- Step 3: Loading Weekly Mortality Data ---")
        weekly_mortality_df = load_and_aggregate_weekly_mortality(file_path=COUNTS_FILE, start_year=START_YEAR, end_year=END_YEAR)
        if weekly_mortality_df.empty: raise ValueError("Step 3 Failed.")

        # --- Step 4: Aggregate Weekly Metrics (NO SMOOTHING in pipeline) ---
        logger.info(f"--- Step 4: Aggregating Weekly Metrics (NO SMOOTHING), Lagging, Standardizing ---")
        # Pass the dictionary defining which lags to create for which base columns
        final_df_full_processed = aggregate_weekly_combined_metrics(
            structured_df=ob_df_parsed_full,
            sentiment_df=ob_df_sentiment_scores,
            weekly_mortality_df=weekly_mortality_df,
            lags_to_create = lags_for_aggregation # Pass the dict
        )
        if final_df_full_processed is None or final_df_full_processed.empty:
             raise ValueError("Step 4 Failed: Aggregation/Processing returned empty DataFrame.")
        logger.info(f"Full processed (unsmoothed) DataFrame shape (before final crop): {final_df_full_processed.shape}")
        del ob_df_parsed_full, ob_df_sentiment_scores, weekly_mortality_df # Free memory

        _temp_df_for_ref = final_df_full_processed.sort_values("week_date").reset_index(drop=True)
        if _temp_df_for_ref.empty:
            raise ValueError("Full processed DataFrame is empty after sorting for time references.")
        
        # Assuming 'time_idx' is (week_date - week_date.min()) / 7 days
        # So, the time_idx for the min_week_date should be 0 if calculated on the full df.
        # For robustness, let's find the row with the minimum week_date.
        min_date_row = _temp_df_for_ref.loc[_temp_df_for_ref['week_date'].idxmin()]
        min_overall_week_date = pd.to_datetime(min_date_row["week_date"])
        min_overall_time_idx = int(min_date_row["time_idx"]) # Ensure this time_idx corresponds to min_overall_week_date
        
        logger.info(f"Global time reference: min_overall_time_idx={min_overall_time_idx} "
                    f"corresponds to min_overall_week_date={min_overall_week_date.strftime('%Y-%m-%d')}")

        # --- Step 5: CROP the Processed DataFrame ---
        logger.info(f"--- Step 5: Cropping Processed Data to {ANALYSIS_START_DATE} - {ANALYSIS_END_DATE} ---")
        if 'week_date' not in final_df_full_processed.columns: raise ValueError("Processed DF missing 'week_date'.")
        final_df_cropped = final_df_full_processed[
            (final_df_full_processed['week_date'] >= pd.to_datetime(ANALYSIS_START_DATE)) &
            (final_df_full_processed['week_date'] <= pd.to_datetime(ANALYSIS_END_DATE))
        ].copy()
        if final_df_cropped.empty: raise ValueError(f"No data remains after cropping final processed DataFrame.")
        logger.info(f"Cropped final DataFrame shape: {final_df_cropped.shape}")
        del final_df_full_processed

        # --- Step 5.5: Save Cropped Data ---
        try:
            save_filename = f"final_weekly_data_UNSMOOTHED_lagged_{ANALYSIS_START_DATE[:4]}_{ANALYSIS_END_DATE[:4]}.csv" # Reflects content
            final_df_cropped.to_csv(os.path.join(PLOT_DIR, save_filename), index=False)
            logger.info(f"Saved CROPPED final weekly data (unsmoothed) to {save_filename}")
        except Exception as e: logger.warning(f"Could not save final data CSV: {e}")

        # --- Step 6: Generate Plots (using CROPPED final_df with UNSMOOTHED analytical data) ---
        logger.info("--- Step 6: Generating Key Data Exploration Plots (Cropped, Unsmoothed Analytical) ---")

        # Define key BASE columns from the processed dataframe
        time_col = 'week_date'
        target_col = 'log_deaths'
        hardship_col = 'hardship_sentiment'
        prop_crime_col = 'property_crime_prop'
        conv_rate_col = 'conviction_rate'
        # Define lagged columns based on what was created in lags_for_aggregation
        lagged_hardship_cols = [f'{hardship_col}_lag{lag}w' for lag in hardship_lags_list]
        lagged_prop_crime_cols = [f'{prop_crime_col}_lag{lag}w' for lag in property_crime_lags_list]


        logger.info("--- Step 6: Generating Key Data Exploration Plots (Cropped, Unsmoothed Analytical) ---")

        # Define key BASE columns for overview plots
        # Adjust this list based on columns actually present and most relevant
        key_cols_overview = [
            'log_deaths', 
            'hardship_sentiment', 
            'property_crime_prop', 
            'violent_crime_prop', # Assuming you have this
            'conviction_rate', 
            'avg_punishment_score', # Assuming you have this
            'total_trials' 
        ]
        # Filter down to only columns that actually exist in the dataframe
        key_cols_overview = [col for col in key_cols_overview if col in final_df_cropped.columns]
        logger.info(f"Using columns for overview plots: {key_cols_overview}")

        if final_df_cropped is not None and not final_df_cropped.empty:

            # --- 1. Correlation Heatmap ---
            logger.info("Generating Correlation Heatmap...")
            heatmap_fig = None
            try:
                if len(key_cols_overview) > 1:
                    corr_matrix = final_df_cropped[key_cols_overview].corr()
                    heatmap_fig = plt.figure(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
                    plt.title('Correlation Heatmap of Key Weekly Variables (Unsmoothed)', fontsize=14)
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    save_path_heatmap = os.path.join(PLOT_DIR, "corr_heatmap_unsmoothed.png")
                    plt.savefig(save_path_heatmap); logger.info(f"Saved plot: {save_path_heatmap}")
                else:
                    logger.warning("Not enough columns available for heatmap.")
            except Exception as e:
                logger.error(f"Heatmap Plot fail: {e}", exc_info=True)
            finally:
                if heatmap_fig is not None and plt.fignum_exists(heatmap_fig.number): plt.close(heatmap_fig)
                plt.close('all')

            # --- 2. Pair Plot (Select fewer variables if too crowded) ---
            logger.info("Generating Pair Plot...")
            pairplot_vars = ['log_deaths', 'hardship_sentiment', 'property_crime_prop', 'conviction_rate'] # Select key vars
            pairplot_vars = [col for col in pairplot_vars if col in final_df_cropped.columns] # Ensure they exist
            pairplot_fig = None
            try:
                if len(pairplot_vars) > 1:
                    pairplot_fig = sns.pairplot(final_df_cropped[pairplot_vars].dropna(), kind='scatter', diag_kind='kde', plot_kws={'alpha':0.1, 's': 10}, corner=True) # Use corner=True for efficiency
                    pairplot_fig.fig.suptitle('Pair Plot Matrix of Key Weekly Variables (Unsmoothed)', y=1.02, fontsize=14)
                    save_path_pairplot = os.path.join(PLOT_DIR, "pairplot_unsmoothed.png")
                    pairplot_fig.savefig(save_path_pairplot); logger.info(f"Saved plot: {save_path_pairplot}")
                else:
                     logger.warning("Not enough columns available for pair plot.")
            except Exception as e:
                 logger.error(f"Pair Plot fail: {e}", exc_info=True)
            finally:
                # Seaborn pairplot returns PairGrid, not a simple figure handle to check with fignum_exists
                plt.close('all') # Close all figures as a precaution after pairplot

            # --- 3. Standardized Time Series Overlay ---
            logger.info("Generating Standardized Time Series Overlay Plot...")
            std_overlay_fig = None
            cols_to_standardize_plot = ['log_deaths', 'hardship_sentiment', 'property_crime_prop']
            cols_to_standardize_plot = [col for col in cols_to_standardize_plot if col in final_df_cropped.columns]
            try:
                if len(cols_to_standardize_plot) > 0:
                    df_std_plot = final_df_cropped[['week_date'] + cols_to_standardize_plot].copy()
                    scaler = StandardScaler()
                    # Standardize only the numeric columns, keep week_date
                    df_std_plot[cols_to_standardize_plot] = scaler.fit_transform(df_std_plot[cols_to_standardize_plot])
                    
                    std_overlay_fig, ax = plt.subplots(figsize=(18, 6))
                    for col in cols_to_standardize_plot:
                        ax.plot(df_std_plot['week_date'], df_std_plot[col], label=f"{col.replace('_',' ').title()} (Std)", alpha=0.7, linewidth=1.2)
                    
                    ax.set_title('Standardized Weekly Time Series Comparison (Unsmoothed)', fontsize=14)
                    ax.set_xlabel('Week Date', fontsize=12)
                    ax.set_ylabel('Standardized Value (Z-score)', fontsize=12)
                    ax.legend()
                    ax.grid(True, linestyle=':')
                    std_overlay_fig.autofmt_xdate()
                    plt.tight_layout()
                    save_path_std_overlay = os.path.join(PLOT_DIR, "standardized_overlay_unsmoothed.png")
                    plt.savefig(save_path_std_overlay); logger.info(f"Saved plot: {save_path_std_overlay}")
                else:
                    logger.warning("No columns available for standardized overlay plot.")
            except Exception as e:
                 logger.error(f"Standardized Overlay Plot fail: {e}", exc_info=True)
            finally:
                if std_overlay_fig is not None and plt.fignum_exists(std_overlay_fig.number): plt.close(std_overlay_fig)
                plt.close('all')
            
            # --- 4. More Rolling Correlation Plots ---
            logger.info("Generating Additional Rolling Correlation Plots...")
            # Assumes plot_rolling_correlation function and rolling_window_weeks are defined
            # Assumes target_col = 'log_deaths'
            target_col_roll = 'log_deaths'
            pairs_for_rolling = [
                (target_col_roll, 'hardship_sentiment'),
                ('hardship_sentiment', 'property_crime_prop'),
                (target_col_roll, 'conviction_rate')
                # Add other pairs if desired
            ]
            for var1, var2 in pairs_for_rolling:
                 if var1 in final_df_cropped.columns and var2 in final_df_cropped.columns:
                     plot_rolling_correlation(
                         df=final_df_cropped, 
                         time_col='week_date', 
                         var1=var1, 
                         var2=var2, 
                         window=rolling_window_weeks, 
                         title=f"{rolling_window_years}y Rolling Correlation: {var1.replace('_',' ').title()} vs {var2.replace('_',' ').title()} (Unsmoothed)", 
                         filename=f"roll_corr_{var1}_vs_{var2}_unsmoothed.png", 
                         plot_dir=PLOT_DIR
                     )
                 else:
                     logger.warning(f"Skipping rolling correlation for {var1} vs {var2}: Column(s) missing.")

            # --- 5. Seasonality Plots for Predictors ---
            logger.info("Generating Weekly Seasonality Boxplots for Predictors...")
            # Assumes plot_weekly_boxplot function is defined
            predictors_for_seasonality = ['hardship_sentiment', 'property_crime_prop', 'conviction_rate', 'total_trials']
            for pred_col in predictors_for_seasonality:
                 if pred_col in final_df_cropped.columns:
                     plot_weekly_boxplot(
                         df=final_df_cropped, 
                         column=pred_col, 
                         title=f"Weekly Distribution of {pred_col.replace('_',' ').title()} (by Week of Year)", 
                         filename=f"boxplot_{pred_col}_weekly_unsmoothed.png", 
                         plot_dir=PLOT_DIR
                     )
                 else:
                     logger.warning(f"Skipping weekly boxplot for {pred_col}: Column missing.")
                     
            
            # --- OPTIONAL: Create and plot SMOOTHED versions locally JUST FOR VISUALIZATION ---
            logger.info(f"--- Step 6b: Generating SMOOTHED plots for VISUALIZATION ONLY (Window: {VISUAL_SMOOTHING_WINDOW}w) ---")
            smoothed_cols_for_plot = {}
            for base_col_name in [target_col, hardship_col, prop_crime_col, conv_rate_col]:
                if base_col_name in final_df_cropped.columns:
                    smooth_col_name = f"{base_col_name}_smooth_VIS" # Add suffix to distinguish
                    smoothed_cols_for_plot[base_col_name] = smooth_col_name
                    final_df_cropped[smooth_col_name] = final_df_cropped[base_col_name].rolling(window=VISUAL_SMOOTHING_WINDOW, center=True, min_periods=1).mean()
                    # Plot the smoothed version
                    plot_time_series(final_df_cropped, time_col, smooth_col_name, f"Weekly {base_col_name.replace('_',' ').title()} (Smoothed - Viz Only)", f"Smoothed Score/Rate", f"{smooth_col_name}_timeseries.png", PLOT_DIR)
            # ---------------------------------------------------------------------------

            # --- Visualize Specific Granger Lags Over Time ---
            logger.info("--- Visualizing Specific Granger Lag Relationships ---")
            target_col_viz = 'log_deaths' 
            predictor_col_viz = 'hardship_sentiment'
            key_lag_viz = 4 # Choose a key significant lag from Granger

            lagged_col_name_viz = f'{predictor_col_viz}_lag{key_lag_viz}w'
            
            if target_col_viz in final_df_cropped.columns and lagged_col_name_viz in final_df_cropped.columns:
                 plot_dual_axis(
                     df=final_df_cropped, 
                     time_col='week_date', 
                     col1=target_col_viz, 
                     col2=lagged_col_name_viz, 
                     label1=target_col_viz.replace('_',' ').title(), 
                     label2=f"{predictor_col_viz.replace('_',' ').title()} (Lag {key_lag_viz}w)", 
                     title=f"{target_col_viz.replace('_',' ').title()} vs. Lagged {predictor_col_viz.replace('_',' ').title()}", 
                     filename=f"dual_axis_{target_col_viz}_vs_{lagged_col_name_viz}.png", 
                     plot_dir=PLOT_DIR
                 )
            else:
                 logger.warning(f"Skipping dual axis plot: Missing '{target_col_viz}' or '{lagged_col_name_viz}'.")

            # Plot distributions of UNSMOOTHED hardship sentiment
            if hardship_col in final_df_cropped.columns:
                 plot_distribution(final_df_cropped[hardship_col], title=f"Distribution of Weekly (Unsmoothed) Hardship Sentiment", xlabel=f"Hardship Sentiment Score", filename=f"{hardship_col}_unsmoothed_distribution.png", plot_dir=PLOT_DIR, filter_zeros=True)

            # Plot ACF/PACF for key UNSMOOTHED series
            plot_lags_acf = 20
            if target_col in final_df_cropped.columns: plot_acf_pacf(final_df_cropped[target_col].dropna(), plot_lags_acf, f"Weekly {target_col.replace('_', ' ').title()}", f"{target_col}_unsmoothed_acf_pacf", PLOT_DIR)
            if hardship_col in final_df_cropped.columns: plot_acf_pacf(final_df_cropped[hardship_col].dropna(), plot_lags_acf, f"Weekly Hardship Sentiment", f"{hardship_col}_unsmoothed_acf_pacf", PLOT_DIR)
            if prop_crime_col in final_df_cropped.columns: plot_acf_pacf(final_df_cropped[prop_crime_col].dropna(), plot_lags_acf, f"Weekly Property Crime Prop", f"{prop_crime_col}_unsmoothed_acf_pacf", PLOT_DIR)

            # Plot CCF of UNSMOOTHED Series
            if target_col in final_df_cropped.columns and hardship_col in final_df_cropped.columns:
                 plot_ccf(final_df_cropped, hardship_col, target_col, max_lags=20, title=f"CCF: Weekly Hardship Sentiment vs Log(Deaths)", filename=f"ccf_{hardship_col}_vs_{target_col}_unsmoothed.png", plot_dir=PLOT_DIR)
            if target_col in final_df_cropped.columns and prop_crime_col in final_df_cropped.columns:
                 plot_ccf(final_df_cropped, prop_crime_col, target_col, max_lags=20, title=f"CCF: Weekly Property Crime Prop vs Log(Deaths)", filename=f"ccf_{prop_crime_col}_vs_{target_col}_unsmoothed.png", plot_dir=PLOT_DIR)

            # Plot Key Lag Scatter Plots (UNSMOOTHED target vs LAGGED UNSMOOTHED predictor)
            logger.info("--- Generating Specific Lag Scatter Plots (Unsmoothed) ---")
            if target_col in final_df_cropped.columns:
                # Example: Plotting for the first lag created for hardship and property crime
                first_lag_hardship = lagged_hardship_cols[0] if lagged_hardship_cols else None
                first_lag_prop_crime = lagged_prop_crime_cols[0] if lagged_prop_crime_cols else None
                
                if first_lag_hardship and first_lag_hardship in final_df_cropped.columns:
                    lag_val = int(first_lag_hardship.split('lag')[1].split('w')[0]) # Extract lag number
                    plot_lag_scatter(final_df_cropped, target_col, hardship_col, lag=lag_val, title=f"LogDeaths vs Lagged Weekly Hardship", filename=f"lag_scatter_{target_col}_vs_{first_lag_hardship}_unsmoothed.png", plot_dir=PLOT_DIR)
                if first_lag_prop_crime and first_lag_prop_crime in final_df_cropped.columns:
                    lag_val = int(first_lag_prop_crime.split('lag')[1].split('w')[0])
                    plot_lag_scatter(final_df_cropped, target_col, prop_crime_col, lag=lag_val, title=f"LogDeaths vs Lagged Weekly Property Crime", filename=f"lag_scatter_{target_col}_vs_{first_lag_prop_crime}_unsmoothed.png", plot_dir=PLOT_DIR)

            # Plot Rolling Correlation (Using UNSMOOTHED variables)
            if target_col in final_df_cropped.columns and prop_crime_col in final_df_cropped.columns:
                 plot_rolling_correlation(final_df_cropped, time_col, target_col, prop_crime_col, window=rolling_window_weeks, title=f"{rolling_window_years}y Rolling Correlation: LogDeaths vs Property Crime Prop", filename=f"roll_corr_{target_col}_vs_{prop_crime_col}_unsmoothed.png", plot_dir=PLOT_DIR)

        else: logger.warning("Cropped final_df empty, skipping data plots.")


        # === Step 7: Train TFT Model (using CROPPED final_df & standardized UNSMOOTHED features) ===
        run_name = f"" # Reflects input
        logger.info(f"--- Step 7: Training and Evaluating TFT Model ({run_name}) ---")

        if final_df_cropped is not None and not final_df_cropped.empty:
            # Get standardized columns (_std suffix) which are based on unsmoothed base/lagged data
            tft_real_features_exist = [col for col in final_df_cropped.columns if col.endswith('_std')]
            tft_real_features_exist_FULL = [col for col in final_df_cropped.columns if col.endswith('_std')]
            logger.info(f"Using these standardized features for TFT ({run_name}): {tft_real_features_exist}")

            if not tft_real_features_exist:
                logger.error("No standardized features found in cropped df. Skipping training.")
            else:
                best_model, trainer, val_dl, val_ds = train_tft_model(
                    df=final_df_cropped,
                    time_varying_reals_cols=tft_real_features_exist,
                    run_name=run_name,
                    max_epochs=MAX_EPOCHS, # Keep increased epochs
                )
                
                current_max_idx = final_df_cropped["time_idx"].max()
                current_pred_length = WEEKLY_MAX_PREDICTION_LENGTH # Param for this run
                current_min_val_windows = 4 # Param for this run (e.g. DEFAULT_MIN_VAL_WINDOWS_TFT)
                current_encoder_length = WEEKLY_MAX_ENCODER_LENGTH # Param for this run

                # training_cutoff based on the data that will go into TimeSeriesDataSet
                current_training_cutoff = current_max_idx - (current_pred_length * current_min_val_windows) - current_pred_length
                
                # The validation samples will start having predictable parts from current_training_cutoff + 1
                validation_start_time_idx = current_training_cutoff + 1
                
                validation_data_for_index = final_df_cropped[
                    final_df_cropped['time_idx'] >= validation_start_time_idx
                ][['time_idx', 'week_date']].drop_duplicates().sort_values('time_idx')

                if validation_data_for_index.empty:
                    logger.error(f"Calculated validation_data_for_index is empty for run {run_name}. "
                                f"Check cutoffs: max_idx={current_max_idx}, training_cutoff={current_training_cutoff}")
                    # Potentially skip evaluation or raise error
                else:
                    logger.info(f"validation_data_for_index created for run {run_name} with {len(validation_data_for_index)} points, "
                                f"from time_idx {validation_data_for_index['time_idx'].min()} to {validation_data_for_index['time_idx'].max()}")

                # Evaluate Model
                if best_model and val_dl and val_ds:
                    logger.info(f"--- Evaluating Best Model for Run: {run_name} ---")
                    eval_metrics = evaluate_model(
                        best_model,
                        val_dl,
                        val_index_df_with_dates=validation_data_for_index, # Crucial for correct dates
                        min_overall_time_idx=min_overall_time_idx,       # Pass global ref
                        min_overall_week_date=min_overall_week_date,     # Pass global ref
                        plot_dir=PLOT_DIR,
                        run_name=run_name
                    )
                    logger.info(f"Final Validation Metrics ({run_name}): {eval_metrics}")
                    # Plot Training History (handling potential list of loggers)
                    if trainer and hasattr(trainer, 'logger'):
                        logger_obj = trainer.logger
                        log_dir_path = None
                        if isinstance(logger_obj, list): # Handle multiple loggers
                            csv_loggr = next((lgr for lgr in logger_obj if isinstance(lgr, CSVLogger)), None)
                            if csv_loggr and hasattr(csv_loggr, 'log_dir'): log_dir_path = csv_loggr.log_dir
                        elif isinstance(logger_obj, CSVLogger) and hasattr(logger_obj, 'log_dir'): # Single CSV logger
                            log_dir_path = logger_obj.log_dir
                        
                        if log_dir_path:
                            logger.info(f"--- Plotting Training History for Run: {run_name} ---")
                            plot_training_history(log_dir_path, run_name, PLOT_DIR)
                        else: logger.warning("CSVLogger log_dir not found. Cannot plot training history.")
                    else: logger.warning("Trainer or logger object missing/invalid. Cannot plot training history.")

                else: logger.error(f"TFT Model training/loading failed for run '{run_name}'.")

                if best_model and val_dl and val_ds and final_df_cropped is not None and not final_df_cropped.empty:
                    logger.info(f"--- Running Baseline Comparison for: {run_name} ---")
                    evaluate_against_baselines(
                        tft_model=best_model,
                        tft_val_dataloader=val_dl,
                        tft_val_dataset=val_ds,
                        val_index_df_with_dates=validation_data_for_index, # You added this
                        full_df_cropped=final_df_cropped.copy(),
                        min_overall_time_idx=min_overall_time_idx,     # <<< PASS IT
                        min_overall_week_date=min_overall_week_date, # <<< PASS IT
                        plot_dir=PLOT_DIR,
                        run_name=run_name, # This should be final_run_name if in final block
                        seasonality_period=52
                    )
                else:
                    logger.warning("Skipping baseline comparison due to missing TFT model, data, or dataloaders.")

                # run_name_no_trials = f"TFT_NoTotalTrials_{ANALYSIS_START_DATE[:4]}_{ANALYSIS_END_DATE[:4]}" # New run name
                # logger.info(f"--- Step 7b: Training and Evaluating TFT Model WITHOUT total_trials ({run_name_no_trials}) ---")

                # # Create feature list excluding total_trials_std
                # tft_real_features_no_trials = [col for col in tft_real_features_exist_FULL if col != 'total_trials_std']
                
                # if not tft_real_features_no_trials:
                #     logger.error("No standardized features remaining after removing total_trials_std. Skipping training.")
                # elif 'total_trials_std' not in tft_real_features_exist_FULL:
                #     logger.warning("'total_trials_std' was not in the original feature list. Skipping NoTotalTrials run.")
                # else:
                #     logger.info(f"Using these standardized features for NO TRIALS model ({run_name_no_trials}): {tft_real_features_no_trials}")

                #     # Train the new model
                #     best_model_no_trials, trainer_no_trials, val_dl_no_trials, val_ds_no_trials = train_tft_model(
                #         df=final_df_cropped,
                #         time_varying_reals_cols=tft_real_features_no_trials, # Pass the filtered list
                #         run_name=run_name_no_trials, # Use the new run name
                #         max_epochs=75, 
                #     )

                #     # Evaluate the new model
                #     if best_model_no_trials and val_dl_no_trials and val_ds_no_trials:
                #         logger.info(f"--- Evaluating Best Model for Run: {run_name_no_trials} ---")
                #         eval_metrics_no_trials = evaluate_model(best_model_no_trials, val_dl_no_trials, val_ds_no_trials, PLOT_DIR, run_name=run_name_no_trials)
                #         logger.info(f"Final Validation Metrics ({run_name_no_trials}): {eval_metrics_no_trials}")
                    
                #         # Interpret the new model
                #         logger.info(f"--- Running TFT Interpretation ({run_name_no_trials}) ---")
                #         interpret_tft(best_model_no_trials, val_dl_no_trials, PLOT_DIR, run_name=run_name_no_trials)
                #     else:
                #         logger.error(f"TFT Model training/loading failed for run '{run_name_no_trials}'.")
        else:
            logger.error("Cropped Final DataFrame empty, cannot train TFT model.")

        if APPLY_TARGET_SMOOTHING_EXPERIMENT and 'deaths_original_unsmoothed' in final_df_cropped and 'deaths_smoothed_for_target' in final_df_cropped:
            logger.info("Plotting original vs. smoothed deaths for validation period...")
            
            # Get the validation segment of final_df_cropped
            # Assuming validation_data_for_index is already defined and contains 'time_idx' and 'week_date' for validation
            if 'validation_data_for_index' in locals() and not validation_data_for_index.empty:
                plot_df_smoothing_effect = pd.merge(
                    validation_data_for_index[['time_idx', 'week_date']],
                    final_df_cropped[['time_idx', 'deaths_original_unsmoothed', 'deaths_smoothed_for_target']],
                    on='time_idx',
                    how='left'
                ).dropna().sort_values('week_date')

                if not plot_df_smoothing_effect.empty:
                    fig_smooth, ax_smooth = plt.subplots(figsize=(18, 6))
                    ax_smooth.plot(plot_df_smoothing_effect['week_date'], plot_df_smoothing_effect['deaths_original_unsmoothed'], label='Original Deaths', color='black', alpha=0.7, linewidth=1)
                    ax_smooth.plot(plot_df_smoothing_effect['week_date'], plot_df_smoothing_effect['deaths_smoothed_for_target'], label=f'Smoothed Deaths ({TARGET_SMOOTHING_WINDOW}w {TARGET_SMOOTHING_TYPE})', color='red', linestyle='--', linewidth=1.2)
                    ax_smooth.set_title(f'Effect of Target Smoothing on Validation Data ({run_name})') # Use appropriate run name
                    ax_smooth.set_xlabel('Week Date')
                    ax_smooth.set_ylabel('Weekly Deaths')
                    ax_smooth.legend()
                    ax_smooth.grid(True)
                    fig_smooth.autofmt_xdate()
                    plt.tight_layout()
                    plt.savefig(os.path.join(PLOT_DIR, f"target_smoothing_effect_{run_name}.png"))
                    plt.close(fig_smooth)

        if final_df_cropped is not None and not final_df_cropped.empty:
            logger.info("--- Starting TFT Model Run with ONLY Hardship Sentiment Features ---")

            # --- Define features for THIS specific run ---
            # Select only the hardship sentiment related _std columns for unknown reals
            hardship_only_unknown_reals = [
                col for col in final_df_cropped.columns
                if col.startswith('hardship_sentiment') and col.endswith('_std')
            ]
            # Also include the base (unlagged) hardship_sentiment_std if it exists
            if 'hardship_sentiment_std' in final_df_cropped.columns and 'hardship_sentiment_std' not in hardship_only_unknown_reals:
                hardship_only_unknown_reals.append('hardship_sentiment_std')
            
            # Ensure no duplicates if base was already caught by startswith
            hardship_only_unknown_reals = list(dict.fromkeys(hardship_only_unknown_reals))


            if not hardship_only_unknown_reals:
                logger.error("No hardship sentiment '_std' features found. Skipping hardship-only TFT run.")
            else:
                logger.info(f"Features for Hardship-Only TFT (time_varying_unknown_reals): {hardship_only_unknown_reals}")

                run_name_hardship_only = f"TFT_HardshipOnly_{ANALYSIS_START_DATE[:4]}_{ANALYSIS_END_DATE[:4]}"
                
                # Use your best-known hyperparameters or a reasonable default set
                # Example: using parameters from your successful unsmoothed run
                current_best_lr = 3e-5 # Or from Optuna if that was run on a broader set
                current_best_hidden_size = 32
                current_best_attn_heads = 4
                current_best_dropout = 0.36
                current_best_hidden_cont_size = 8
                current_best_encoder_length = 30 # Or 40, whatever was best for unsmoothed
                current_best_grad_clip = 0.1

                # Train the hardship-only model
                best_model_hs_only, trainer_hs_only, val_dl_hs_only, val_ds_hs_only = train_tft_model(
                    df=final_df_cropped.copy(), # Pass a fresh copy of the data
                    time_varying_reals_cols=hardship_only_unknown_reals, # <<< KEY CHANGE
                    run_name=run_name_hardship_only,
                    # Pass your chosen hyperparameters for this run
                    lr=current_best_lr,
                    hidden_size=current_best_hidden_size,
                    attn_heads=current_best_attn_heads,
                    dropout=current_best_dropout,
                    hidden_cont_size=current_best_hidden_cont_size,
                    encoder_length=current_best_encoder_length,
                    clip_val=current_best_grad_clip,
                    max_epochs=MAX_EPOCHS, # Use your standard max epochs for a full run
                    batch_size=BATCH_SIZE,
                    # Ensure find_lr_mode is 'skip' if you're setting LR manually based on prior knowledge
                )

                validation_data_for_index = final_df_cropped[
                    final_df_cropped['time_idx'] >= validation_start_time_idx
                ][['time_idx', 'week_date']].drop_duplicates().sort_values('time_idx')

                if validation_data_for_index.empty:
                    logger.error(f"Calculated validation_data_for_index is empty for run {run_name}. "
                                f"Check cutoffs: max_idx={current_max_idx}, training_cutoff={current_training_cutoff}")
                    # Potentially skip evaluation or raise error
                else:
                    logger.info(f"validation_data_for_index created for run {run_name} with {len(validation_data_for_index)} points, "
                                f"from time_idx {validation_data_for_index['time_idx'].min()} to {validation_data_for_index['time_idx'].max()}")

                # Evaluate the hardship-only model
                if best_model_hs_only and val_dl_hs_only:
                    # Ensure validation_data_for_index and min_overall references are available
                    # You might need to recalculate validation_data_for_index if encoder_length changed for this run
                    # or pass them as they were for the main model run for consistency.
                    # For simplicity, let's assume they are consistent for now.
                    
                    logger.info(f"--- Evaluating Hardship-Only Model: {run_name_hardship_only} ---")
                    eval_metrics_hs_only = evaluate_model(
                        best_model_hs_only,
                        val_dl_hs_only,
                        val_index_df_with_dates=validation_data_for_index, # Re-use from main setup
                        min_overall_time_idx=min_overall_time_idx,
                        min_overall_week_date=min_overall_week_date,
                        plot_dir=PLOT_DIR,
                        run_name=run_name_hardship_only
                    )
                    logger.info(f"Hardship-Only Model Validation Metrics ({run_name_hardship_only}): {eval_metrics_hs_only}")

                    # Compare against baselines
                    evaluate_against_baselines(
                        tft_model=best_model_hs_only,
                        tft_val_dataloader=val_dl_hs_only,
                        tft_val_dataset=val_ds_hs_only, # Pass the dataset object for this specific run
                        val_index_df_with_dates=validation_data_for_index,
                        full_df_cropped=final_df_cropped.copy(),
                        min_overall_time_idx=min_overall_time_idx,
                        min_overall_week_date=min_overall_week_date,
                        plot_dir=PLOT_DIR,
                        run_name=run_name_hardship_only,
                        seasonality_period=52
                    )

                    # Interpret the hardship-only model
                    logger.info(f"--- Running TFT Interpretation for Hardship-Only Model: {run_name_hardship_only} ---")
                    interpret_tft(best_model_hs_only, val_dl_hs_only, PLOT_DIR, run_name=run_name_hardship_only)
                else:
                    logger.error(f"Hardship-Only TFT Model training/loading failed for run '{run_name_hardship_only}'.")
        else:
            logger.error("final_df_cropped is empty. Skipping all TFT runs.")

        # === Step 8: Interpretation & Granger Causality (using CROPPED final_df & UNSMOOTHED features) ===
        logger.info("--- Step 8: Model Interpretation & Granger Causality (Cropped Period - Using Unsmoothed) ---")
        if final_df_cropped is not None and not final_df_cropped.empty:
             # Interpretation
            if best_model and val_dl:
                logger.info(f"--- Running TFT Interpretation ({run_name}) ---")
                interpret_tft(best_model, val_dl, PLOT_DIR, run_name=run_name)
            else: logger.warning("Skipping TFT interpretation as best_model or val_dl not available.")

            # Granger Causality - Test the UNSMOOTHED variables and their specific lags
            logger.info("--- Running Granger Causality Tests (Unsmoothed Vars & Specific Lags) ---")
            granger_results = {}
            max_weekly_lag = 12
            target_col_granger = target_col # Use base 'log_deaths'

            # Define metrics to test (key BASE metrics + key LAGGED BASE metrics)
            metrics_to_test_granger = {}
            key_base_unsmoothed = [hardship_col, prop_crime_col, conv_rate_col] # Example key base metrics
            key_lagged_unsmoothed = lagged_hardship_cols + lagged_prop_crime_cols # Use the lists of lagged col names

            for col in key_base_unsmoothed + key_lagged_unsmoothed:
                 if col in final_df_cropped.columns: metrics_to_test_granger[col] = col.replace('_', ' ').title()

            metrics_to_test_granger_final = {k: v for k, v in metrics_to_test_granger.items()} # Already checked existence
            logger.info(f"Metrics selected for Granger testing (Unsmoothed): {list(metrics_to_test_granger_final.keys())}")

            for metric_col, metric_label in metrics_to_test_granger_final.items():
                if target_col_granger in final_df_cropped.columns:
                    logger.info(f"Granger tests: '{metric_col}' vs '{target_col_granger}'")
                    # run_granger_causality applies differencing internally, suitable for unsmoothed
                    desc1 = f"'{metric_label}' -> '{target_col_granger.replace('_',' ').title()}'"
                    granger_results[desc1] = run_granger_causality(final_df_cropped, metric_col, target_col_granger, max_lag=max_weekly_lag)
                    desc2 = f"'{target_col_granger.replace('_',' ').title()}' -> '{metric_label}'"
                    granger_results[desc2] = run_granger_causality(final_df_cropped, target_col_granger, metric_col, max_lag=max_weekly_lag)
                else: logger.warning(f"Target '{target_col_granger}' missing for Granger."); break

            if granger_results:
                 plot_granger_causality_results(granger_results, title_prefix=f"Unsmoothed_Metrics_vs_LogDeaths", plot_dir=PLOT_DIR) # Updated title
            else: logger.warning("No valid Granger results to plot for unsmoothed data.")

            # # (Inside Step 8 after Granger tests)
            # logger.info("--- Running Bivariate VAR/IRF Analysis (Using Differenced Unsmoothed Data) ---")
            # var_lags_select = 12 # Max lags for VAR model selection
            # irf_horizon = 20 # Weeks ahead for IRF plot

            # target_col_var = 'log_deaths' 
            # predictors_for_var = ['hardship_sentiment', 'property_crime_prop', 'conviction_rate'] # Example predictors

            # # Prepare differenced data for VAR
            # cols_for_var = [target_col_var] + predictors_for_var
            # cols_for_var = [col for col in cols_for_var if col in final_df_cropped.columns] # Check existence
            
            # if len(cols_for_var) >= 2: # Need at least target and one predictor
            #     df_for_var = final_df_cropped[cols_for_var].copy()
            #     df_diff_var = df_for_var.diff().dropna() # Create differenced data
                
            #     if not df_diff_var.empty and df_diff_var.shape[0] > var_lags_select + 15:
            #         # Loop through predictors for BIVARIATE analysis
            #         for predictor_col_var in predictors_for_var:
            #             if predictor_col_var in df_diff_var.columns:
            #                 logger.info(f"--- Running VAR/IRF for: {predictor_col_var} vs {target_col_var} ---")
            #                 # Call the existing function, passing the DIFFERENCED data subset
            #                 analyze_bivariate_var_irf(
            #                     df=final_df_cropped, # Pass the original unsmoothed df
            #                     target_col=target_col_var,
            #                     predictor_col=predictor_col_var,
            #                     max_lags_var=var_lags_select,
            #                     irf_periods=irf_horizon,
            #                     plot_dir=PLOT_DIR,
            #                     run_name=f"VAR_Diff_{predictor_col_var}" 
            #                 )
            #             else:
            #                  logger.warning(f"Predictor '{predictor_col_var}' not available in differenced data for VAR.")
            #     else:
            #         logger.warning("Not enough data after differencing for VAR analysis.")
            # else:
            #     logger.warning("Not enough columns available for VAR analysis.")

        else: logger.warning("Cropped Final DataFrame empty, skipping interpretation & Granger.")

    # --- Error Handling & Script End ---
    except FileNotFoundError as e: logger.error(f"Data file not found: {e}.")
    except ValueError as e: logger.error(f"Data processing/config error: {e}", exc_info=True)
    except ImportError as e: logger.error(f"Missing library: {e}.")
    except Exception as e: logger.error(f"Unexpected script error: {e}", exc_info=True)

    script_end_time = time.time()
    logger.info(f"--- Script finished in {(script_end_time - script_start_time):.2f} seconds ({(script_end_time - script_start_time)/60:.2f} minutes) ---")

# --- Run Main ---
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    main()
