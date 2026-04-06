# ExoPattern v3.1 — Implementation Blueprint

## Paper Target

**"Conformal Anomaly Ranking for Transit Candidate Prioritization in Kepler Light Curves"**

Core claim: We present the first calibrated anomaly ranking system for transit candidate prioritization, with event-level evaluation, injection-recovery validation, and conformal false alarm control.

---

## Current State (what exists)

- **5 stars** with labeled lightcurves locally (`data/labeled/`); full 150-star dataset on Kaggle
- **38 window features** extracted with `window_size=50`, `stride=12` (75% overlap)
- **8 models**: IF, LOF, OCSVM, DBSCAN, Ensemble, BLS, CNN-AE, LSTM-AE
- **Pre-trained artifacts** in `artifacts/models/` (trained on 150 stars via Kaggle notebooks)
- **7 experiments** in `experiments/run_experiments.py`

### Critical bugs that must be fixed

1. **Label-index drift**: sigma-clipping changes array length but labels are indexed from pre-clipped array
2. **No star-level CV**: windows from same star leak between train/test
3. **BLS broken in CV**: `fit(X)` without time/flux -> no ephemeris learned
4. **Experiment 3 evaluates on training data**: train = test
5. **Feature importance uses wrong method**: `tree.feature_importances_` is invalid for IF

---

## Architecture Overview

```
Raw lightcurve CSV (per star)
    |
    v
Preprocessing + Feature Extraction (existing, with fixes)
    | Output: feature matrix + metadata (star_id, window_start_idx, window_time_range)
    v
Per-Window Anomaly Scoring (3 classical models; deep models excluded from core CV)
    | Output: per-window scores from each model, all in HIGHER = MORE ANOMALOUS convention
    v
Candidate Event Generation (NEW)
    | Output: candidate events with aggregated multi-view scores
    v
Conformal Calibration (NEW)
    | Output: calibrated p-values per candidate
    v
Ranked Candidate List
    |
    v
Event-Level Evaluation (NEW) against ground truth transit windows
```

---

## Design Decisions

### D0. Score direction convention

**Universal rule: higher score = more anomalous, everywhere in the pipeline.**

All raw model scores are transformed to this convention immediately upon extraction:

| Model | Raw convention | Transformation |
|-------|---------------|----------------|
| IF | `score_samples()` returns negative (sklearn: more negative = more anomalous) | `score = -raw_score` |
| LOF | `score_samples()` returns negative (sklearn: more negative = more anomalous) | `score = -raw_score` |
| OCSVM | `score_samples()` returns negative (wrapper uses sklearn decision_function internally) | `score = -raw_score` |

Note: all three classical models expose `score_samples()` via the wrapper classes in `baselines.py`. The OCSVM wrapper's `score_samples()` calls sklearn's `decision_function()` internally (baselines.py:124-133) but presents the same interface. The negation is applied uniformly to all `.score_samples()` outputs.

**Deep models (CNN-AE, LSTM-AE) are excluded from the core CV comparison** because the pre-trained artifacts were trained on the same 150-star corpus used for evaluation. Using them inside star-level CV would leak test-star information. They may be added later as an auxiliary pretrained view with an explicit data-leakage caveat, or retrained per fold if compute allows.

This transformation happens once, at the point of score extraction in `MultiViewScorer.score_windows()`. All downstream code (thresholding, normalization, aggregation, conformal calibration) assumes higher = more anomalous. No further negations anywhere.

### D1. What is a "candidate event"?

A candidate event is a contiguous group of anomalous windows that represents a single astrophysical event (transit, flare, artifact).

**Definition:**
1. Score each window with each model (higher = more anomalous, per D0)
2. Compute composite score per window: min-max normalize each model's scores (fitted on proper-training stars), then take the mean across models
3. Compute the composite threshold from the proper-training fold: the **97.5th percentile** of composite scores on proper-training windows. This is recall-oriented: ~2.5% of training windows are flagged.
4. A window is "flagged" if its composite score exceeds this threshold
5. Merge consecutive flagged windows separated by at most `gap_tolerance` unflagged windows into a single candidate event
6. An event spans from the start time of its first flagged window to the end time of its last flagged window

Note: per-model thresholds are NOT used. Thresholding operates solely on the composite score. Individual model scores are only used as inputs to the composite.

**Parameters:**
- `flag_quantile = 0.975` (top 2.5% of training-fold composite scores)
- `gap_tolerance = 2` windows (up to 2 unflagged windows between flagged ones still merge)
- Minimum event duration: 1 window (no minimum)
- Maximum event duration: `max_event_windows = 20` (prevents runaway merges; 20 windows at stride 12 with window_size 50 spans ~290 cadences ~ 6 days)

**Event score aggregation (default + ablation):**
- **Default: top-3 mean** of window composite scores within the event (or all scores if fewer than 3 windows). More robust than pure max against single noisy windows.
- **Ablation variants to test:**
  - `max`: peak window score
  - `top3_mean`: mean of top 3 window scores (default)
  - `length_penalized_max`: `max_score * min(1, 3 / n_windows)` — penalizes long sloppy events

### D2. Multi-view scoring

Each candidate event gets a score vector from 3 classical views (core CV) with deep views available as an auxiliary comparison:

| View | Source | Score type |
|------|--------|------------|
| IF score | `isolation_forest.score_samples(X)` negated | density-based |
| LOF score | `lof.score_samples(X)` negated | k-distance |
| OCSVM score | `ocsvm.score_samples(X)` negated | decision function |

**Not included in core CV:**
- DBSCAN (returns zeros when all points are noise)
- BLS (score_samples returns zeros)
- Ensemble (redundant with IF+LOF)
- CNN-AE, LSTM-AE (pre-trained on same 150 stars — data leakage in CV; see D0)

**Composite score:**
- Fit min-max normalizer on **proper-training stars** (not calibration stars, see D3)
- Normalize each view's scores to [0, 1] via the fitted min-max
- `composite_score = mean(normalized_view_scores)`
- This is the score used for candidate generation, ranking, and conformal calibration

**Why mean, not max or learned weights:**
- Max is dominated by the noisiest model
- Learned weights require labeled anomalies (defeats unsupervised framing)
- Mean is robust and interpretable; the conformal layer handles calibration

### D3. Star-level cross-validation split protocol

**Split unit:** star (not window, not point)

**Three-way split within each outer fold:**

```
150 stars
  |-- Outer fold k: 30 test stars (evaluation only, never seen during training or calibration)
  |-- Remaining 120 stars:
       |-- 90 proper-training stars (model fitting + normalizer fitting)
       |-- 30 calibration stars (conformal null distribution)
```

**Protocol (5-fold outer CV):**
1. `GroupKFold(n_splits=5)` on star_id -> 30 test stars per fold
2. Within the 120 non-test stars: `GroupShuffleSplit(n_splits=1, test_size=0.25)` -> 90 proper-train, 30 calibration
3. **Proper-training stars (90):** fit models, fit min-max normalizer
4. **Calibration stars (30):** score with fitted models, generate candidates, collect false-positive candidate scores as the conformal null set
5. **Test stars (30):** score, generate candidates, compute conformal p-values, evaluate event-level metrics

This ensures no star serves both as model-fit data and calibration null. The exchangeability assumption for conformal prediction is clean: calibration and test candidates come from disjoint stars processed by the same pipeline.

**Retraining policy:**
- **Classical models (IF, LOF, OCSVM):** retrained per fold on proper-training windows. Fast (~seconds).
- **Deep models (CNN-AE, LSTM-AE):** excluded from core CV. The pre-trained artifacts were trained on the same 150 stars, so using them in CV leaks test-star data. Options for a future extension: (a) retrain per fold (~30 min each, feasible on Kaggle), (b) pretrain on an external disjoint corpus, (c) present as auxiliary comparison with explicit caveat. For the core paper results, only the 3 classical models participate in CV.

**With 5 local stars:** leave-one-star-out with 3 train / 1 calibration / 1 test. Illustrative only.

### D4. Event-level evaluation metrics

**Ground truth events:**
- For each star, compute transit windows from NASA ephemeris: `phase = ((time - epoch) % period) / period`
- A ground truth event is a contiguous block of points where `|phase| < (duration_hours / 24) / (2 * period)` (wrapping around phase 0/1)
- Buffer: extend each ground truth event by +/-2 cadences (+/-1 hour at Kepler long cadence) for timing uncertainty

**Matching (tightened):**
- A predicted candidate event **detects** a ground truth event if:
  - The **candidate center time** `(start_time + end_time) / 2` falls within the buffered ground truth interval, **OR**
  - The overlap between candidate and ground truth time ranges exceeds **25% of the ground truth event duration**
- This prevents large sloppy candidates from claiming credit via marginal overlap
- A ground truth event is **detected** if at least one candidate event matches it
- A candidate event is a **true positive** if it matches at least one ground truth event; otherwise **false positive**
- One-to-one matching: each ground truth event can be claimed by at most one candidate (the highest-scoring one). This prevents multiple fragmented candidates from inflating recall.

**Metrics (all event level):**

| Metric | Definition |
|--------|------------|
| Event Recall | (# detected ground truth events) / (# total ground truth events) |
| Event Precision | (# TP candidate events) / (# total candidate events) |
| Event F1 | harmonic mean of Event Recall and Event Precision |
| Recall@K | Event recall when only top K candidates (by score) are inspected |
| Precision@K | Event precision in the top K candidates |
| AU-PR (event) | Area under the event-level precision-recall curve (sweep threshold) |

**Primary metric:** Recall@K where K = 2 * (number of ground truth events in test set).

### D5. Conformal calibration

**Method:** Split conformal prediction (Vovk et al., 2005), inductive conformal anomaly detection variant.

**Setup:**
- **Calibration set:** false-positive candidate events from the 30 calibration stars (separate from both proper-training and test stars, per D3)
- **Null distribution:** composite scores of calibration candidates that do NOT overlap any ground truth transit on the calibration stars
- **Exchangeability:** calibration and test candidates are generated by the same pipeline from disjoint Kepler stars with comparable noise properties

**Algorithm:**
1. On calibration stars: score windows, generate candidates, identify false positives (no ground truth overlap)
2. Collect their composite scores: `{s_1, s_2, ..., s_n}`
3. For a new test candidate with composite score `s_test`:
   - `p_value = (|{i : s_i >= s_test}| + 1) / (n + 1)`
4. Reject null (declare anomalous) at significance level alpha if `p_value <= alpha`

**Guarantee:** FAR <= alpha + 1/(n+1) for any alpha, regardless of score distribution.

**Cross-conformal extension:** For the paper, aggregate p-values across the 5 outer folds. Each fold has its own calibration set. Report per-fold and aggregated metrics.

**Paper claim scope:** "Calibrated candidate prioritization among transit-host light curves." We do NOT claim general survey-level false alarm control (that would require a control sample of non-planet-host stars).

### D6. TLS feature extraction

**Package:** `transitleastsquares` (Hippke & Heller 2019)

**Per-star global features (6, computed once per star, constant across all windows):**
- `tls_sde`: Signal Detection Efficiency (primary statistic)
- `tls_period`: best-fit orbital period (days)
- `tls_depth`: best-fit transit depth (fractional)
- `tls_duration`: best-fit transit duration (days)
- `tls_odd_even`: ratio of odd-transit depth to even-transit depth (EB discriminator)
- `tls_snr`: signal-to-noise ratio

**Per-event consistency features (3, computed per candidate event):**
- `tls_epoch_distance`: time from candidate event center to nearest TLS-predicted transit epoch (days). Small values = candidate coincides with a known periodic signal.
- `tls_phase_agreement`: `1 - |candidate_phase - nearest_transit_phase|` where phase is computed from TLS best-fit period. High values = candidate aligns with TLS periodicity.
- `tls_depth_ratio`: ratio of candidate event's flux dip depth to TLS-inferred transit depth. Values near 1.0 = consistent with TLS model; outlier values = unusual depth.

**Integration:**
- 6 global TLS features appended to window feature vector (44 = 38 + 6). These enter the anomaly models and thus affect window-level scoring.
- 3 event-consistency features are computed per candidate AFTER candidate generation. They do NOT affect candidate generation (which uses window composite scores only). They are appended to the candidate's final ranking score as additional dimensions:
  - **Candidate ranking score** = `alpha * composite_score + (1 - alpha) * consistency_score` where `consistency_score` is a normalized combination of the 3 TLS event features, and `alpha = 0.7` (ablated in experiments).
  - Normalization of the 3 consistency features is fit on **proper-training candidates only** (min-max per feature), then frozen and applied identically to calibration and test candidates. This ensures the relative scale of composite_score and consistency_score is stable across splits.
  - This means: candidates are generated purely from window anomaly scores, but ranked using both anomaly scores and TLS consistency. The separation is clean: generation is anomaly-driven, ranking is anomaly + physics.

### D7. Injection-recovery test design

**Purpose:** Characterize pipeline completeness as f(planet_radius, orbital_period).

**Injection method:**
- `batman` package (Kreidberg 2015) for synthetic transit light curves
- Inject by multiplication: `flux_injected = flux_real * transit_model`
- One planet per star per trial

**Parameter grid:**
- Planet radius: [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0] R_Earth (8 values)
- Orbital period: [1, 2, 5, 10, 20, 50, 100, 200] days (8 values)
- Impact parameter b: uniform [0, 0.8]
- Transit epoch t0: uniform within observation window
- Stellar parameters: default 1 R_sun (or from metadata if available)
- Limb darkening: quadratic, solar-type coefficients

**Grid:** 64 bins * 25 injections = 1,600 trials

**Recovery criterion:**
- Run full pipeline on injected light curve
- An injection is **recovered** if any candidate event's center falls within +/-duration of any injected transit mid-time
- Star-level recovery: at least one transit from the injected planet is recovered

**Output:** 8x8 completeness heatmap in (log P, R_p) space with 50%, 80%, 90% contours.

**Compute budget:** 1,600 injections * ~3 seconds per pipeline run (feature extraction + scoring, no model retraining) ~ 80 min. Models are pre-trained; injection-recovery is inference-only.

### D8. SHAP explainability

- `shap.TreeExplainer` on the underlying sklearn IsolationForest (correct method)
- Compute for top-10 ranked candidate events
- Show which features drive anomaly scores
- Beeswarm plot (paper figure) + per-candidate waterfall (supplement)

---

## File Plan

### New files

| File | Purpose |
|------|---------|
| `backend/ml/events.py` | CandidateEvent dataclass + candidate generation |
| `backend/ml/event_evaluation.py` | Ground truth events, matching, event-level metrics |
| `backend/ml/conformal.py` | Split conformal + cross-conformal calibration |
| `backend/ml/tls_features.py` | TLS global + event-consistency feature extraction |
| `backend/ml/injection.py` | Injection-recovery test framework (batman) |
| `backend/ml/multi_view.py` | Score normalization + multi-view aggregation |
| `backend/ml/pipeline.py` | End-to-end RankingPipeline |
| `experiments/run_paper_experiments.py` | New experiment runner for paper |
| `experiments/paper_config.py` | Config for paper experiments |

### Modified files

| File | Change |
|------|--------|
| `backend/ml/preprocessing.py` | Add `extract_features_with_metadata()` (new function, keep old one intact) |
| `backend/ml/evaluation.py` | Add `cross_validate_star_level()` (new function) |
| `backend/ml/feature_names.py` | Add `'tls'` feature group (extend, don't rewrite) |

### NOT modified

- `frontend/app.py` — update after paper pipeline works
- `backend/ml/deep_models.py` — models are fine
- `backend/ml/model_registry.py` — factory is fine
- `backend/ml/baselines.py` — keep as-is; score negation happens in multi_view.py

---

## Implementation Sequence

### Phase 1: Foundation (metadata + events + star-level CV + event metrics)

1. Fix label-index drift in preprocessing (reindex labels after sigma-clipping)
2. Add `extract_features_with_metadata()` to preprocessing.py (returns features + metadata list)
3. Implement `CandidateEvent` dataclass and `generate_candidates()` in events.py
4. Implement ground truth event computation and event matching in event_evaluation.py
5. Implement event-level metrics (recall, precision, F1, recall@K, AU-PR)
6. Add `cross_validate_star_level()` to evaluation.py with 3-way split (train/calibration/test)
7. Standardize score direction in multi_view.py (negate all raw scores)

### Phase 2: TLS features

1. Implement `extract_tls_features()` in tls_features.py
2. Implement event-consistency features (epoch distance, phase agreement, depth ratio)
3. Register TLS feature group in feature_names.py

### Phase 3: Multi-view ranking

1. Implement `MultiViewScorer` with min-max normalization and mean aggregation
2. Implement `RankingPipeline` end-to-end
3. Wire up candidate generation with composite scores

### Phase 4: Conformal calibration

1. Implement `ConformalCalibrator` (fit on null scores, predict p-values)
2. Implement cross-conformal aggregation across folds
3. Integrate into RankingPipeline

### Phase 5: Injection-recovery

1. Implement transit injection via batman
2. Implement `InjectionRecoveryTest` (grid, recovery criterion, completeness map)

### Phase 6: SHAP

1. TreeExplainer on IF for top-10 candidates
2. Generate figures

---

## Experiment Design

### Experiment 1: Star-Level CV with Event-Level Metrics

5-fold star-level GroupKFold, 90/30/30 split per fold.
Models: IF, LOF, OCSVM (retrained per fold), multi-view composite of the 3.
Deep models (CNN-AE, LSTM-AE) excluded from core CV due to data leakage; may appear in auxiliary comparison with caveat.
Metrics: Event Recall, Event Precision, Event F1, Recall@2K, AU-PR.
Ablation: event score aggregation (max vs top-3-mean vs length-penalized).

### Experiment 2: Conformal Calibration Analysis

Cross-conformal on composite scores.
Show: calibration plot (empirical FAR vs nominal alpha), candidates retained at alpha = {0.01, 0.05, 0.10}, event recall at each alpha.

### Experiment 3: Injection-Recovery Completeness

1,600 injections. Completeness heatmap in (R_p, P). Report: "ExoPattern detects X% of planets with R > Y for P < Z."

### Experiment 4: Ablation

Incremental: 38 features -> +6 TLS -> +event-consistency -> multi-view -> conformal.
All evaluated with star-level CV and event-level metrics.

### Experiment 5: SHAP Feature Importance

Top-10 candidates. Beeswarm + waterfall.

---

## Paper Outline

1. **Introduction** — transit detection, limitations of supervised pipelines, gap in calibrated unsupervised ranking
2. **Related Work** — Astronomaly, conformal prediction in astronomy (GW calibration), TLS, injection-recovery
3. **Method**
   - 3.1 Feature extraction (38 window + 6 TLS + 3 event-consistency)
   - 3.2 Multi-view anomaly scoring (3 classical models, score normalization)
   - 3.3 Candidate event generation
   - 3.4 Conformal calibration with proper train/calibration/test separation
4. **Experiments**
   - 4.1 Dataset: 150 Kepler planet hosts, star-level CV
   - 4.2 Event-level evaluation framework
   - 4.3 Model comparison (Exp 1)
   - 4.4 Calibration analysis (Exp 2)
   - 4.5 Injection-recovery (Exp 3)
   - 4.6 Ablation (Exp 4)
5. **Results & Discussion**
   - Ranking quality, calibration fidelity, completeness curves
   - Comparison to TLS-only detection
   - Limitations: 150 planet-host stars only, Kepler-only, no TESS transfer, no non-host control sample
6. **Conclusion**

Target: 10-12 pages, MNRAS or RASTI format.

---

## Dependencies to Add

```
transitleastsquares>=2.0    # TLS features
batman-package>=2.4         # transit injection
shap>=0.42                  # explainability
```

---

## Compute Budget (revised, explicit about retraining)

| Task | Retraining? | Time (Kaggle) |
|------|-------------|---------------|
| TLS feature extraction (150 stars) | N/A | ~30 min |
| 5-fold CV: classical models (IF, LOF, OCSVM) | Retrained per fold | ~5 min total |
| Multi-view scoring + candidate generation | N/A | ~5 min |
| Conformal calibration | N/A | < 1 min |
| Injection-recovery (1,600 trials, inference only) | No retraining | ~80 min |
| SHAP (top-10 candidates) | N/A | ~5 min |
| **Total** | | **~2.5 hours** |

Fits within a single Kaggle session (12-hour limit).
