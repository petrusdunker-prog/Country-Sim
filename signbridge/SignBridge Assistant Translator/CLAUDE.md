# SignBridge — Claude Code Guide

ASL sign language recognition system. Python backend: MediaPipe feature extraction feeding an ensemble of TF/Keras classifiers.

## Environment

- **Python**: 3.12 only — `py -3.12` (mediapipe has no wheel for 3.13/3.14)
- **Working directory**: `signbridge/python/`
- **Dataset cache**: `D:\SignBridgeCache\` (on external drive — plug in external HDD before processing)

## Current State

- **Dataset processing**: complete — 2,731 / 2,731 real signs processed into `data/*.npz`.
- **Training**: complete — 251 main batches (10 signs each, `conversation_b*`) + 414 remedial batches (3–5 signs each, `conversation_r*`) covering ~2,303 signs, plus a sign-level router (`sign_router_*`) that gates which batches run per frame at inference time.
- **Fingerspelling**: complete, 99.71% accuracy (`models/ASL/fingerspelling_*`).
- **Full-ensemble accuracy**: `stress_test.py`'s sentence-level simulation (real frame data, temporal smoothing) currently scores **~33% sign-level / <1% sentence-level** — well below the 50% target.
- **Known bottleneck**: the ensemble's cross-batch confidence scores are degenerate — correct and incorrect predictions are statistically indistinguishable (~99.8% either way). Post-hoc decision-rule fixes (confidence margin, two Platt-scaling calibration variants, smoother threshold sweeps) have all been tried and don't help — there's no signal in the scores to exploit. The real next lever is upstream of the decision rule: changing how individual batch models are trained (e.g. label smoothing, so they stop saturating near 100% on every non-rejected input), or replacing confidence-based arbitration between batches with a different combination mechanism entirely.

To retrain or extend the ensemble:
```powershell
cd signbridge/python
py -3.12 auto_train_conversation.py --batch N --target 90 --augment
py -3.12 auto_train_remedial.py --batch rN --target 85 --augment
py -3.12 train_sign_router.py
py -3.12 stress_test.py
```

`run_completion_chain.py` automates the eval → remedial-generation → router-retrain → stress-test loop end to end.

## Key Files

| File | Purpose |
|------|---------|
| `pipeline/capture.py` | MediaPipe Tasks API wrapper (Hand + Pose + Face landmarkers) |
| `pipeline/features.py` | 143-element feature vector per frame |
| `pipeline/classifier.py` | `ASLClassifier` (rule + ML, basic signs — ML half currently untrained, falls back to rules only), `FingerspellingClassifier` (complete), **`ConversationClassifier`** (production ensemble — router-gated batches, chance-normalized confidence), `MotionSignClassifier` (temporal motion signs: STOP/HELP/MORE/FINISHED) |
| `pipeline/smoother.py` | `TemporalSmoother` (confirms a sign after sustained frame agreement) + `MotionTracker` |
| `pipeline/tf_wrapper.py` | `TFClassifierWrapper` — pickled wrapper around a Keras model, lazy-loads on first `predict_proba()` call |
| `server.py` | FastAPI WebSocket server — `ws://localhost:8000/ws/classify` |
| `collect.py` | Webcam data collection (SPACE=record, N=next, R=redo, Q=quit) |
| `train.py` | Trains the legacy/basic classifier → `models/asl_classifier.joblib`. **Not the production path** — currently untrained/unused; superseded by the batch ensemble below |
| `process_asl_citizen.py` | Process ASL Citizen dataset → `data/*.npz` |
| `process_wlasl.py` | Process WLASL dataset (optional second source) |
| `train_conversation.py` | Defines the `BATCHES` dict; trains one main batch (10 signs, 60% threshold) |
| `train_remedial.py` | Defines the `REMEDIAL_BATCHES` dict; trains one remedial batch (3–5 signs, 55% threshold) |
| `auto_train_conversation.py` / `auto_train_remedial.py` | Warm-start training loop for one batch until target accuracy or plateau |
| `train_sign_router.py` | Trains the sign-level router (top-K candidate signs per frame → batch keys) |
| `eval_per_sign.py` | Evaluates every batch model in isolation; finds weak signs to remediate |
| `run_completion_chain.py` | Orchestrates eval → remedial generation → router retrain → stress test |
| `stress_test.py` | Full sentence-level ensemble simulation with temporal smoothing — the real accuracy metric (vs. `eval_per_sign.py`'s per-batch isolated numbers, which don't catch cross-batch issues) |
| `calibrate_confidence.py` | Per-batch Platt-scaling calibration — tried twice, both regressed accuracy; disabled by default (`models/ASL/batch_calibration_map.joblib.rejected_*`) |

## Dataset — ASL Citizen

- 83,399 MP4 videos recorded by 52 Deaf signers, but only **2,731 are real distinct signs** (80,671 videos) — the rest (~2,664 videos) are "seed" reference/exemplar clips, not real signing data (see Known Gotchas)
- Zip already downloaded: `D:\SignBridgeCache\ASL_Citizen.zip` (42.77 GB)
- Already extracted to: `D:\SignBridgeCache\ASL_Citizen\`
- Video filename format: `{numeric_id}-{GLOSS}.mp4` (or `{numeric_id}-seed{GLOSS}.mp4` for excluded seed clips)
- Processed features saved to: `python/data/*.npz` (one file per sign)
- MediaPipe model files auto-downloaded to: `python/_models/`

## Running the Server

```powershell
cd signbridge/python
py -3.12 server.py
```

After training, reload the model without restarting:
```powershell
curl -X POST http://localhost:8000/train/reload
```

## Running Data Collection (webcam)

For a Deaf signer to record personal fine-tuning data:
```powershell
cd signbridge/python
py -3.12 collect.py
```

Controls: `SPACE` record sample · `N` next sign · `R` redo · `Q` quit

## Architecture

```
Browser webcam frame (base64 JPEG)
  → WebSocket (server.py)
    → HolisticCapture.process() — MediaPipe Hand + Pose + Face
      → extract_static_features() — 143-element float32 vector
        → ASLClassifier            (rule + ML, basic signs)        ─┐
        → FingerspellingClassifier (99.71% acc)                     ├─ merged
        → ConversationClassifier   (router-gated batch ensemble)    │
        → MotionSignClassifier     (temporal motion signs)         ─┘
          → TemporalSmoother — confirm after sustained frame agreement
            → { label, confidence, source } back to browser
```

`ConversationClassifier` internals: the sign-level router predicts the top-K candidate signs for each frame → maps each to its batch key → only those main + remedial batches actually run → the highest chance-normalized confidence among their non-`__OTHER__` predictions wins.

## Known Gotchas

- **Always use `PYTHONIOENCODING=utf-8`** when running any script on Windows, not just when redirecting to a file — several scripts crash on Unicode characters in print statements otherwise
- **Always use `py -3.12 -u`** (`-u` = unbuffered) if you want live output when piping
- The `W0000` and `E0000` lines from MediaPipe are harmless — telemetry noise, ignore them
- CSV manifest column detection doesn't match ASL Citizen's column names — the script correctly falls back to parsing the gloss from the video filename (`{id}-{GLOSS}.mp4`)
- ASL Citizen "seed" videos (`{id}-seedGLOSS.mp4`) are reference/exemplar clips shown to a signer as a prompt before they perform the sign — not real signing data. `process_asl_citizen.py` excludes them at the source (checks for a lowercase `seed` prefix before uppercasing the gloss); don't remove that check or they'll reappear as bogus pseudo-signs like `SEEDABOUT`
- Renaming/renumbering a `conversation_b{N}_*`/`conversation_r{N}_*` model file does **not** update the path baked inside its pickled `TFClassifierWrapper` — it'll still point at the old filename and crash `eval_per_sign.py` / production `pipeline/classifier.py` (silently mis-loads in some cases). Re-save the wrapper with a corrected `_model_path` after any bulk rename.
- This project lives under a OneDrive-synced folder. Heavy concurrent file churn (many scripts writing/deleting at once) can cause silent file corruption — verify file *content*, not just existence, after any bulk file-surgery operation.
- When using a `CosineDecay` LR schedule, don't also add a `ReduceLROnPlateau` callback — they fight each other. Use `EarlyStopping` + `ModelCheckpoint` only.
- Training data from the user's own webcam recordings was discarded — user doesn't know ASL well enough. All training data comes from ASL Citizen (Deaf signers). A contact may record personal fine-tuning data later.

## Next Steps

Dataset processing and batch training are both complete — there's no backlog to resume. The open question is how to push the ensemble's stress-test accuracy past ~33%, given the confidence-rescaling/decision-rule angle is exhausted (see Current State above). Candidates, roughly in order of how invasive they are:

1. **Upstream batch-model calibration** — retrain individual batches with label smoothing or different regularization so their softmax outputs express genuine uncertainty instead of saturating near 100%
2. **Non-confidence ensemble combination** — replace "highest confidence wins" with a structurally different arbitration mechanism (e.g. a learned meta-classifier over candidate batches, rather than comparing their raw scores)
3. **Smoother redesign** — the temporal smoother's strict vote-confirmation path never actually fires in practice (confirmed empirically); its permissive single-frame passthrough dominates instead. Worth a deeper look once (1) or (2) gives the confidence signal something real to act on.

None of these are quick experiments — discuss the approach before diving in, given several quicker ideas (margin rule, calibration, threshold tuning) have already been tried this way and didn't pan out.
