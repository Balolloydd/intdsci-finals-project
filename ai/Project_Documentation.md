# Seoul Bike-Sharing Demand Prediction
### A Supervised Machine-Learning Project — Final Report

**Dataset:** `SeoulBikeData-1.csv` (hourly bike-rental counts, Seoul Metropolitan bike system)
**Companion code:** `Seoul_Bike_ML_Project.ipynb` (reproducible Jupyter notebook)
**Task type:** Supervised learning — **binary classification**
**Evaluation:** Confusion matrix, classification report, and majority-class baseline

---

## A. Introduction

### A.1 What the problem is about

The Seoul Metropolitan Government operates a public bike-sharing network. The dataset records,
**hour by hour for a full year (1 December 2017 to 30 November 2018)**, how many bikes were rented
together with the prevailing weather and calendar conditions: temperature, humidity, wind speed,
visibility, dew-point temperature, solar radiation, rainfall, snowfall, the season, whether the day
is a holiday, and whether it is a functioning (working) day.

Forecasting bike demand is a practical operations problem. A useful demand model lets the city
(i) pre-position more bikes at stations before peak hours, (ii) schedule maintenance during
low-demand hours, and (iii) plan for weather-driven swings in usage.

### A.2 What the project produces

Because the assignment requires performance to be reported with a **confusion matrix**, the problem
is framed as a **supervised binary classification** task:

> **Given the weather and calendar conditions for a particular hour, will bike demand be HIGH or
> LOW?**

* **Target (label):** `high_demand` — `1` (High) when the hourly rental count is at or above the
  dataset median, `0` (Low) otherwise. The median is used because it yields a balanced,
  interpretable split ("above-average demand").
* **Input features:** weather measurements plus calendar features. Some features are *engineered*
  from the date (day of week, weekend flag) and *encoded* (season, holiday, functioning day).

The output of the project is a trained **Logistic Regression** classifier, an 80/20 train/test
evaluation, a confusion matrix, and a comparison against a majority-class baseline — all
reproducible with a fixed random seed (`random_state = 42`).

### A.3 How the report maps to the required sections

| Required section | Where it is covered |
|---|---|
| A. Introduction | This section |
| B. Data Collection & Preprocessing | Section B |
| C. Feature Engineering | Section C |
| D. Data Splitting | Section D |
| E. Model Selection & Training | Section E |
| F. Model Evaluation | Section F |
| Limitations | Section F.4 |

---

## B. Data Collection and Preprocessing

### B.1 Data collection

The data is loaded from `SeoulBikeData-1.csv`. Two collection details are important:

1. **File encoding** — the file is saved in **CP949** (the Windows Korean code page). Reading it
   with the default UTF-8 encoding fails, so the encoding is specified explicitly
   (`pd.read_csv(..., encoding="cp949")`).
2. **Header text** — a few column names contain a non-ASCII degree symbol, which reads back as a
   garbled character. These headers are normalised to clean, ASCII names during preprocessing.

**Dataset overview:** 8,760 rows × 14 columns = every hour of 365 days (24 × 365 = 8,760).

| Column | Meaning | Type |
|---|---|---|
| `Date` | Date (day/month/year) | string → datetime |
| `Rented Bike Count` | Number of bikes rented that hour (target source) | int |
| `Hour` | Hour of day (0–23) | int |
| `Temperature` | Air temperature (°C) | float |
| `Humidity` | Relative humidity (%) | int |
| `Wind speed` | Wind speed (m/s) | float |
| `Visibility` | Visibility (10 m units) | int |
| `Dew point temperature` | Dew point (°C) | float |
| `Solar Radiation` | Solar radiation (MJ/m²) | float |
| `Rainfall` | Rainfall (mm) | float |
| `Snowfall` | Snowfall (cm) | float |
| `Seasons` | Winter / Spring / Summer / Autumn | string |
| `Holiday` | No Holiday / Holiday | string |
| `Functioning Day` | Yes / No | string |

### B.2 Exploratory Data Analysis (EDA) and interpretation

Basic inspection (shape, types, missing values, descriptive statistics) and four visualisations were
produced in the notebook.

**Findings and interpretation:**

* **No missing values** (0 across all columns) and **no duplicated rows** — the data is clean, so
  heavy cleaning is unnecessary.
* **Strong daily (diurnal) cycle.** Demand is lowest in the small hours (≈ 3–5 a.m.) and peaks at
  two commuting times: a **morning peak (≈ 7–9 a.m.)** and a larger **evening peak (≈ 17–19 h)**.
  `Hour` is therefore expected to be highly informative.
* **Strong seasonal effect.** Summer has by far the highest average demand; Winter is the lowest.
  `Season` and `Temperature` should be highly informative.
* **Right-skewed target.** The rental-count distribution is skewed right with a long tail to
  ≈ 3,500 and a floor of 0. A single median cut gives a clean, balanced binary target.
* **Target statistics:** mean ≈ 705, median = 504.5, min = 0, max = 3,556.

### B.3 Preprocessing techniques (with output after each)

Three preparation steps were applied, each shown with its output in the notebook:

1. **Rename columns** to clean, ASCII names. The mapping matches on the ASCII text before any
   parenthesis, so the garbled degree symbol is never used directly.
   *Output:* the 14 columns are renamed, e.g. `Rented Bike Count → rented_bikes`,
   `Temperature(℃) → temperature`, `Functioning Day → functioning_day`.
2. **Parse the `Date` column** into a real datetime. The dates are **day-first**
   (`01/12/2017` = 1 December 2017), which the `Seasons` column confirms (Winter).
   *Output:* `date` becomes `datetime64`; confirmed range **2017-12-01 to 2018-11-30**.
3. **Re-check data quality.**
   *Output:* 0 remaining missing values, 0 duplicated rows, 8,760 rows retained.

No rows were dropped: the dataset was already complete, so no imputation was required.

---

## C. Feature Engineering

### C.1 Methods used

Three groups of derived features were created:

1. **Time features from the date**
   * `day_of_week` — 0 = Monday … 6 = Sunday (from `date.dt.dayofweek`).
   * `weekend` — 1 for Saturday/Sunday, else 0.
   Rationale: bike usage on weekends differs from weekdays, so these add signal the raw date
   string does not expose.
2. **Binary encodings** of the two categorical columns:
   * `holiday` — No Holiday = 0, Holiday = 1.
   * `functioning_day` — Yes = 1, No = 0.
3. **One-hot encoding** of `season` with *drop-first* (to avoid the dummy-variable/collinearity
   problem), producing `season_Spring`, `season_Summer`, `season_Winter` (Autumn is the reference).

### C.2 Feature selection and justification (correlation analysis)

Feature selection was justified with **Pearson correlation** of each feature against the raw rental
count, plus a full correlation heatmap to check for redundancy.

Correlation with `rented_bikes` (sorted by strength):

| Feature | Correlation |
|---|---|
| temperature | **+0.54** |
| season_Winter | −0.43 |
| hour | **+0.41** |
| dew_point | **+0.38** |
| season_Summer | +0.30 |
| solar_radiation | +0.26 |
| functioning_day | +0.20 |
| humidity | −0.20 |
| visibility | +0.20 |
| snowfall | −0.14 |
| rainfall | −0.12 |
| wind_speed | +0.12 |
| holiday | −0.07 |
| weekend | −0.04 |
| day_of_week | −0.03 |
| season_Spring | +0.02 |

**Justification for the selected features:**

* The strongest signals are **temperature** (warmer → more riding), **hour** (commuting peaks) and
  **dew point / solar radiation** (mild, sunny weather encourages cycling).
* **humidity, rainfall and snowfall** show a *negative* correlation — wetter/colder conditions
  suppress demand, a sensible reason to keep them.
* **functioning_day** and **season** capture calendar and seasonal behaviour that `hour` alone does
  not.
* All **16** engineered features are retained: the heatmap shows no near-perfect duplicate pairs,
  and each feature has a plausible physical or behavioural reason to matter. (Note:
  `temperature`/`dew_point` and `day_of_week`/`weekend` are moderately correlated, which is expected
  and acceptable for the model.)

The final modelling matrix `X` has **16 features**; the target is `y = high_demand`.

---

## D. Data Splitting

The data is split **80% training / 20% testing** with **`random_state = 42`**, exactly as the
assignment requires. `stratify = y` is used so that both the training and testing sets preserve the
original 50/50 class balance (this does not change the split ratio or the seed).

* **Training set:** 7,008 rows (3,504 High / 3,504 Low)
* **Testing set:** 1,752 rows (876 High / 876 Low)

The test set is **held out** and is used only once, at the end, for evaluation — it never
participates in training or in any preprocessing step.

---

## E. Model Selection and Training

### E.1 Algorithm

**Logistic Regression** was chosen as the learning model.

### E.2 Why this algorithm

The assignment asks for a *simple* model, and Logistic Regression is the right fit:

* It is a standard, **interpretable linear classifier**: it outputs the probability that an hour is
  High-demand, and we threshold at 0.5 to obtain the class.
* It is a strong, transparent **baseline workhorse** for tabular, moderate-size data and is fast to
  train and debug.
* Its **coefficients are directly interpretable** (direction and relative strength of each feature),
  which lets us *explain* the model — one of the project's learning outcomes.
* Because it assumes roughly linear, additive effects, it is an honest "simple" model: good
  performance is not the product of an over-powerful black box.

Configuration: `LogisticRegression(max_iter=1000, random_state=42)`. `max_iter` is raised to
guarantee solver convergence; `random_state` keeps results reproducible.

### E.3 Training (no data leakage)

Features are standardised with a `StandardScaler` that is **fit on the training set only**, then
used to *transform* the test set with the same statistics. The test set therefore never influences
the scaling. The model is then fit on the scaled training features.

The learned coefficients (ranked by magnitude) confirm the EDA story:

| Feature | Coefficient | Reading |
|---|---|---|
| functioning_day | **+1.85** | Working days have higher demand |
| season_Winter | **−1.65** | Winter suppresses demand |
| rainfall | **−1.28** | Rain reduces demand |
| temperature | **+0.96** | Warmer is better |
| humidity | **−0.92** | High humidity slightly hurts |
| hour | **+0.91** | Evening/commuting hours raise demand |
| solar_radiation | +0.64 | Sunlight raises demand |
| season_Spring | −0.50 | Spring is a reference-adjacent low |
| dew_point | +0.41 | Milder air raises demand |
| season_Summer | −0.37 | (relative to Winter reference) |
| snowfall | −0.35 | Snow reduces demand |
| weekend | −0.27 | Weekends slightly lower |
| holiday | −0.21 | Holidays slightly lower |
| visibility | −0.10 | Poor visibility slightly hurts |
| day_of_week | +0.09 | Weak |
| wind_speed | −0.06 | Weak |

---

## F. Model Evaluation

### F.1 Performance on the held-out test set

**Overall accuracy: 87.33%** (1,530 / 1,752 correct).

Classification report:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low demand (0) | 0.88 | 0.87 | 0.87 | 876 |
| High demand (1) | 0.87 | 0.88 | 0.87 | 876 |
| **Accuracy** | | | **0.87** | 1,752 |

### F.2 Confusion matrix and interpretation

Confusion matrix (rows = actual, columns = predicted):

|  | Predicted Low | Predicted High |
|---|---|---|
| **Actual Low** | **762** (True Neg.) | 114 (False Pos.) |
| **Actual High** | 108 (False Neg.) | **768** (True Pos.) |

**Interpretation:**

* **762 + 768 = 1,530** hours were classified correctly → **87.33% accuracy**.
* The **diagonal dominates** — the model is right on the large majority of hours. The off-diagonal
  cells (114 + 108 = 222) are the misclassified hours, mostly the *borderline, near-median* cases
  that are genuinely hard to separate.
* **Precision and recall are balanced (~0.87–0.88) for both classes.** Because the target is a
  balanced 50/50 split, a well-calibrated model should perform similarly on High and Low demand —
  and it does. The model is neither over-predicting High nor over-predicting Low.
* **High-demand recall of 0.88** means the model catches ~88% of high-demand hours — useful for
  the operational goal of pre-positioning bikes before peak demand.

### F.3 Comparison against a baseline

| Model | Test accuracy |
|---|---|
| Baseline (always predict the majority class) | 0.5000 |
| **Logistic Regression** | **0.8733** |
| **Improvement over baseline** | **+0.3733** |

Because the classes are balanced, the majority-class baseline can only achieve 50%. The model's
**87.33%** is a **+37.33 percentage-point** improvement, demonstrating that it genuinely learns
from the weather and calendar features rather than guessing.

### F.4 Limitations

* **Binary simplification.** Reducing demand to High/Low with a single median threshold loses the
  *magnitude* of demand; a regression or multi-class model would capture more detail.
* **`Visibility` is capped.** In the source data visibility is often reported at a fixed upper bound
  (≈ 2000 m, meaning "clear"), so it acts more like a poor-visibility flag than a true continuous
  measurement.
* **Single year, one city.** The model reflects one year in Seoul; seasonal and weather patterns may
  shift in other years or cities, so it would need retraining on fresh data to stay reliable.
* **Rare extremes are under-represented.** Unusual weather extremes are a small share of the data,
  so the model is less reliable on those days.

---

## G. Conclusion

Using only weather and calendar features, a simple **Logistic Regression** classifier predicts
whether an hour in Seoul will have **High** or **Low** bike demand. The full, reproducible pipeline —
data collection, cleaning, exploratory analysis, feature engineering, an 80/20 split
(`random_state = 42`), training with no data leakage, and evaluation via a **confusion matrix** and
a majority-class baseline — is contained in the companion notebook
`Seoul_Bike_ML_Project.ipynb`. The model reaches **87.33% test accuracy**, a clear, interpretable
improvement over the 50% baseline, satisfying all of the required documentation sections (A–F).

---

## H. Reproducibility

To reproduce these results:

1. Install the packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
2. Place `SeoulBikeData-1.csv` in the same folder as the notebook.
3. Open `Seoul_Bike_ML_Project.ipynb` and run all cells top to bottom
   (`Run All`). With the fixed seed (`random_state = 42`) the splits, training and metrics are
   identical on every run.

**Environment used:** Python 3.14, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2.
