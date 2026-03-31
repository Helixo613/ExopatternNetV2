# ExoPattern v3.1 — Concerns

## Highest-Risk Technical Debt
- `experiments/run_experiments.py` currently evaluates on the same data it trains on in at least one path, so published metrics can be overstated if the pipeline is used as-is.
- `backend/ml/preprocessing.py` removes rows during sigma clipping, while downstream label handling can still assume the original row count, which creates label-index drift and misaligned supervision.
- `backend/ml/baselines.py` includes models that do not behave like the rest of the stack: `DBSCAN` can degenerate into all-noise behavior, and `BLS` does not learn a real ephemeris in the generic `fit(X)` path.
- `backend/ml/deep_models.py` is operationally expensive and the shipped pretrained weights are tied to the same 150-star corpus used elsewhere, which makes core CV leakage a real risk.

## Data And Reproducibility
- `backend/data/acquisition.py` depends on live NASA/MAST services, so data collection is not deterministic unless the downloaded dataset is cached and versioned.
- `data/labeled/` is only a small local smoke-test set, while the research story depends on a larger Kaggle-generated corpus stored outside the repo.
- `artifacts/models/` is also external to git, so local results depend on large binary artifacts that may be missing or version-mismatched on a fresh checkout.
- `notebooks/kaggle/` appear to be the true training pipeline, which means the repo is not fully reproducible from the tracked source alone.

## Security And Operational Risk
- `backend/api/routes.py` saves uploaded files using user-supplied names, so filename validation and collision handling should be treated as a security concern.
- `backend/api/routes.py` writes temporary files and only cleans them up on the success path in some routes, so failed requests can leave debris in `backend/uploads/`.
- `frontend/app.py` falls back to training a synthetic baseline model at runtime when no artifact is found, which can hide deployment problems and produce misleading "working" states.
- `backend/api/routes.py` and `frontend/app.py` accept arbitrary user uploads, so file size limits, content validation, and cleanup need to stay strict.

## Performance And Scale
- `frontend/app.py` is doing a lot of work in a single request cycle: file loading, preprocessing, scoring, visualization, and optional fallback training.
- `backend/ml/preprocessing.py` uses sliding windows with heavy feature extraction; this is acceptable for the 5-star smoke test but can become slow on the 150-star corpus.
- `experiments/run_experiments.py` repeatedly recomputes features across folds and feature groups, which can explode runtime if caching is not carefully controlled.
- `backend/ml/deep_models.py` is expensive enough that fold-wise retraining is likely impractical without a tighter compute plan.

## Modeling Risks
- `backend/ml/models.py` maps window predictions back to points with heuristic voting, which is fragile for sparse transit signals and can blur event boundaries.
- `backend/ml/evaluation.py` and `experiments/run_experiments.py` still reflect point-level anomaly framing in places, which is weaker than event-level evaluation for astronomy.
- `backend/ml/model_registry.py` mixes classical and deep model loading in one registry, but the pretrained artifact assumptions differ enough that misuse is easy.
- `backend/ml/feature_names.py` documents 38 core features, but the broader project already relies on additional TLS and global features, so feature-set drift is likely unless versioned carefully.

## Architectural Fragility
- `frontend/app.py` is tightly coupled to specific artifact names and a specific training story, which makes UI behavior sensitive to backend changes.
- `backend/api/routes.py` trains, predicts, and exports from one Flask app, so the service boundary is thin and hard to test in isolation.
- `experiments/config.py` encodes the research design in constants, so small changes in the study protocol can require coordinated edits across multiple modules.
- `results/` is reproducible output, but because it is generated rather than source-controlled, stale figures and tables can easily survive after code changes.

## Summary
- The codebase is real and substantial, but the main risks are leakage, reproducibility, and scientific validity rather than lack of implementation.
- The safest next work is to harden the evaluation pipeline, isolate training from calibration/test, and make artifact provenance explicit.
