# Smart Expense Classifier handoff — 24 November 2026

## Resume here

The next agreed milestone is **offline Arabic prediction in Flutter**. The Arabic scoring model already runs in TensorFlow Lite on the computer, and the Dart text processor already matches Python. Connect those two pieces in Flutter and verify the result on a phone or emulator. The user wants to learn by doing one stage at a time. Explain what each step does, why it is needed, and how it was checked. Work on Arabic mobile first. The user explicitly deferred improvements to the mixed model.

Use the existing Windows checkout at `C:\Users\user\New folder\smart-expense-classifier`. At handoff, branch `main` contains modified and untracked project files. The original pre-repair project is available at commit `ad34a55`. A new clone of that commit or the earlier `expense-classifier-updates.zip` will omit the later bonuses. Preserve the working tree.

## Mobile starting point

| Existing piece | Role and evidence |
|---|---|
| `models/arabic.joblib` | Reference Python classifier. Use the current model when comparing outputs. |
| `mobile/export_arabic.py` | Exports the fitted linear classifier, folds in its sigmoid calibration parameters, normalizes the seven scores and checks them with the LiteRT interpreter. Requires `ai-edge-litert==2.4.0` in addition to project dependencies. |
| `mobile/assets/arabic.tflite` | 434,652 bytes. Input is a float32 feature vector shaped **[1, 15488]**. Output is **[1, 7]** normalized class probabilities. |
| `mobile/assets/arabic_features.json` | Vocabulary order, IDF values, feature block weights, and class order. Ship this and the TFLite file from the same export. |
| `mobile/lib/arabic_features.dart` | `ArabicFeatures.fromJson(assetText)` and `encode(description)`. NFC normalization, cleanup, word unigrams/bigrams, character fragments, per-block L2 normalization, and block weights. |
| `mobile/verify_dart.py`, `mobile/tool/check_vectors.dart` | Compares Dart’s nonzero feature indices and values with the fitted Python pipeline. The Dart package uses `unorm_dart`. |

**Last verified:** LiteRT chose the same categories as Python on 2,129 inputs with maximum absolute probability difference 0.000000348. Dart matched all feature positions on 2,138 inputs with maximum absolute value difference about 0.000000015. These are conversion checks, not accuracy estimates. Neither check exercised Flutter’s native runtime on a device.

**Next implementation and acceptance check**

1. Inspect the installed Flutter SDK and available devices. The last session found Flutter at `C:\src\flutter` with bundled Dart 3.3.4. Choose a compatible Flutter interpreter package and preserve the existing files when adding app scaffolding.
2. Register the two assets in the Flutter pubspec. Load the vocabulary and interpreter once. Build a small Arabic input screen with a classify button and a result.
3. Encode the sentence, reject blank or all-zero vectors as the Python demo does, and pass a single float32 row to the interpreter. Display the category at the largest output index and its probability. Read label order from the JSON.
4. Test **دفعت فاتورة الإنترنت**. The current Python/mobile scoring reference is **Bills, about 99.2%**. An earlier Arabic model produced 98.6%. Also test diacritics, punctuation-only input, repeated classifications, and an offline run on a device or emulator. Record the target, build command and observed results. An APK, installed app and device-level parity are still pending.

From the project root in PowerShell:

```powershell
# Recheck Dart features. Run dart pub get inside mobile first on a fresh checkout.
& '.venv/Scripts/python.exe' -X utf8 mobile/verify_dart.py --dart 'C:/src/flutter/bin/cache/dart-sdk/bin/dart.exe'
# Rebuild and verify the two mobile assets if the Arabic model changes.
& '.venv/Scripts/python.exe' -X utf8 mobile/export_arabic.py
# Run the desktop/web demo.
& '.venv/Scripts/python.exe' -m streamlit run app/main.py --server.address=127.0.0.1 --server.port=8517
```

## Project history and model decisions

- **Brief.** The assignment PDF is at `C:\Users\user\Downloads\AI & machineLearning-internTask.pdf`. Seven labels: Food, Transport, Bills, Shopping, Health, Entertainment, Other. Core work includes EDA, preprocessing, TF-IDF/logistic regression baseline, two additional models, tuning, accuracy/precision/recall/F1, confusion matrix, saved model, prediction function and Streamlit demo. No numerical passing accuracy is specified.
- **Original diagnosis.** The shipped project did well on synthetic statements but got only 12/28 newly authored everyday probes correct. All 47 airport-parking expenses were labeled Other. The original English cleaner removed Arabic letters, and the original classifier could assign Food to punctuation alone. The original 3-page diagnostic is `reports/classifier_audit.pdf`.
- **English data repair.** Fixed the Travel labeling rule in `01_data_inspection.ipynb` and regenerated the processed CSV. The raw bank file is preserved. There are 38,215 usable bank records after removing label conflicts. Added 343 everyday notes (252 train, 35 validation, 56 test). `src/dataset.py` groups bank splits by normalized text and keeps the authored partitions fixed. Final notebook is literally named `notebooks/02_final_project.ipynb.ipynb`. It compares logistic regression, Naive Bayes and Linear SVM, tunes and calibrates the SVM, and saves the classical English model. Notebook 03 is for loading/recovery.
- **English transformer bonus.** A frozen multilingual MiniLM encoder with a classifier trained on everyday training notes performed better for notes than one trained on bank-dominated data. The live English notes prediction averages the semantic and classical probabilities. Marked debit/credit statements use the classical path. Word-removal effects explain the combined prediction. English remarks with no recognized classical features still trigger the insufficient-detail guard. See `src/predict.py`, `src/semantic_encoder.py`, and `experiments/semantic_comparison.py`. On a fresh checkout, run `models/semantic_encoder/restore.ps1` for the two git-ignored encoder assets. The fitted everyday head is in that folder; the comparison script writes candidate heads into its supplied encoder directory.
- **Arabic.** The cleaner now retains Unicode letters, normalizes NFC, and strips optional Arabic vowel marks and elongation. `data/arabic_expenses.csv` has 2,128 notes: 2,016 train, 56 validation, 56 test. Training includes 315 authored scenario families with variants. `data/arabic_expense_sources.csv` tracks translations, source-informed terminology and paraphrases. Word plus character TF-IDF beat words alone in the initial comparison. The deployed model uses a calibrated Linear SVM with word weight 0.5, character weight 1 and C=2. The tested Arabic transformer candidate performed worse on validation.
- **Mixed.** A separate Arabic–English code-switching dataset contains 126 notes (84/21/21). Its saved model uses word and character features with logistic regression. This was selected after an earlier test inspection, so reserve a fresh independent set before claiming a final mixed benchmark. `experiments/train_language_models.py` presently rebuilds and overwrites **both** Arabic and mixed artifacts. Mobile integration requires neither retraining nor running this script.
- **Demo.** `app/main.py` provides sidebar choices English, العربية and Arabic + English, each with separate examples, confidence and a “Why this category?” section. `src/language_predict.py` routes requests. All 26 suggestions were exercised after the model update. Stale Streamlit imports previously caused an import error and an Arabic no-features warning. When code or artifacts change, restart the actual local server and verify the user’s phrase through the live app.

## Latest recorded scores

| Active prediction path | Validation accuracy | Test accuracy |
|---|---:|---:|
| English bank statements | 96.91% (7,932 rows) | 99.13% (7,595 rows) |
| English everyday notes, via serving code | 32/35 · 91.4% | 53/56 · 94.6% |
| Arabic | 50/56 · 89.3% | 50/56 · 89.3% |
| Mixed | 13/21 · 61.9% | 12/21 · 57.1% |

The user’s aspiration is 90–95% on everyday expenses. These are small, authored benchmarks explored during development. Keep bank and everyday scores separate. Fresh independent English, Arabic and mixed samples would support stronger accuracy claims. Use `tests/audit_classifier.py`, `tests/test_text_processing.py`, and `experiments/arabic_features.py` for their respective checks; the historical audit alone does not measure every current serving path.

## After mobile

The explanation, multilingual interface and transformer-comparison bonuses have been implemented. The outstanding submission work is a setup README, a current 2–4 page report with the deployed-model results and limitations, a final Git commit/push, and the 5–10 minute presentation. The historical PDF and ZIP predate the latest work. Check the final contents before delivery. No remote submission or presentation has been verified.
