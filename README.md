# Fake Review Detector

A command-line machine learning tool that classifies product or service reviews as **fake** (deceptive/spam) or **genuine**. It combines text features (word and character TF-IDF) with hand-crafted stylometric features, and uses a stacked ensemble of classifiers with a tuned decision threshold.

## Features

- **Stacked ensemble model**: Logistic Regression + calibrated Linear SVM (both on TF-IDF and style features) + Gradient Boosting (on style features), combined by a Logistic Regression meta-classifier.
- **Rich features**:
  - Word n-grams (1-3) and character n-grams (2-5) with TF-IDF
  - Style features: text length, average word length, sentence length, exclamation/question mark rate, uppercase ratio, digit ratio, lexical diversity, repeated phrases, first-person pronoun use, strong/superlative wording, promotional wording, and URL count
- **Automatic threshold tuning** on a validation split to maximize F1.
- **Interactive mode**: type a review and get a verdict, fake probability, and the signals that triggered suspicion.
- **Batch mode**: score every review in a CSV file and write predictions to a new CSV.
- **Flexible column detection**: automatically finds the text and label columns, or you can specify them.
- **Built-in demo data**: runs without a dataset (for testing only).

## Requirements

- Python 3.8+
- Packages: `numpy`, `pandas`, `scikit-learn`, `joblib`

Install with:

```bash
pip install numpy pandas scikit-learn joblib
```

## Project Setup

1. Create a project folder (for example `fake_review_detector`).
2. Inside that folder, add the script file `fake_review.py`.
3. Inside the same folder, create a folder named `data`.
4. Download the **Kaggle** fake reviews dataset (the one used to build this project) and put it inside the `data` folder, named `reviews.csv`.

Your folder should look like this:

```
fake_review_detector/
├── fake_review.py
├── README.md
└── data/
    └── reviews.csv      <- Kaggle dataset
```

## Quick Start

After the setup above, open a terminal in the project folder and run:

```bash
python fake_review.py
```

The script automatically finds `data/reviews.csv`. On the first run the model trains and is saved to `fake_review.joblib`. Subsequent runs load the saved model and start the interactive prompt.

## Retraining With a Different Dataset

If you want to retrain the model on another dataset instead of the Kaggle one, pass the new file with `--data` and add `--retrain`:

```bash
python fake_review.py --data path/to/your_new_dataset.csv --retrain
```

If your column names are not detected automatically, specify them:

```bash
python fake_review.py --data path/to/your_new_dataset.csv --text-col review_text --label-col label --retrain
```

`--retrain` is needed because otherwise the script reuses the already saved `fake_review.joblib` model. The new dataset must follow the format described in the next section.

> **Note:** If no dataset is found, the script trains on a tiny synthetic demo dataset. The accuracy shown in that case is **not realistic** and the model should not be used for real decisions.

## Dataset Format

A CSV file with one column of review text and one column of labels.

**Recognized text column names** (case-insensitive): `text`, `text_`, `review`, `review_text`, `reviewtext`, `content`, `comment`, `body`

**Recognized label column names**: `label`, `class`, `category`, `fake`, `is_fake`, `deceptive`, `target`, `spam`

**Recognized label values** (case-insensitive):

| Meaning | Values |
|---------|--------|
| Fake    | `cg`, `fake`, `deceptive`, `spam`, `1`, `true`, `yes` |
| Genuine | `or`, `genuine`, `truthful`, `real`, `0`, `false`, `original`, `no`, `ham` |

Rows with unrecognized labels, text shorter than 10 characters, or duplicate text are dropped. The dataset must contain both classes.

Example:

```csv
text_,label
"Absolutely amazing! Best purchase ever!!!",CG
"Used it for three weeks, battery is okay but the charger feels flimsy.",OR
```

This format matches the popular Kaggle "Fake Reviews Dataset" (`CG` = computer-generated, `OR` = original).

## Usage

### Train / run interactively

```bash
python fake_review.py
python fake_review.py --data path/to/reviews.csv
python fake_review.py --data reviews.csv --text-col review_text --label-col label
```

### Force retraining

```bash
python fake_review.py --retrain
```

The model is also retrained automatically if no model file exists, or if a real dataset is now available but the saved model was trained on demo data.

### Batch prediction

```bash
python fake_review.py --batch new_reviews.csv --output predictions.csv
```

Adds two columns to your CSV:

- `fake_score`: probability that the review is fake
- `prediction`: `fake` or `genuine` (based on the tuned threshold)

### Interactive session example

```
Fake Review Detector | model accuracy: 95.12% | type q to quit

Enter a review: Best product ever!!! Must buy, you will not regret it!
Verdict          : FAKE
Fake probability : 97.40%
Model accuracy   : 95.12%
Signals          : heavy exclamation use, extreme or superlative wording, promotional wording
```

Type `q`, `quit`, or `exit` (or press Ctrl+C) to leave.

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data FILE` | Path to the labeled training CSV | `reviews.csv` or `data/reviews.csv` |
| `--text-col NAME` | Name of the review text column | auto-detected |
| `--label-col NAME` | Name of the label column | auto-detected |
| `--batch FILE` | CSV of reviews to score in batch mode | none (interactive mode) |
| `--output FILE` | Where to write batch predictions | `predictions.csv` |
| `--model FILE` | Path to save/load the trained model | `fake_review.joblib` |
| `--retrain` | Force retraining even if a model exists | off |

## How It Works

1. **Load and clean**: HTML tags and entities are stripped, whitespace is normalized, labels are mapped to 0/1.
2. **Split**: 80% train / 20% test (stratified). A further slice of the training data is used as validation.
3. **Threshold tuning**: a model is fit on the reduced training set, and the probability threshold (0.20-0.80) that maximizes F1 on validation data is chosen.
4. **Final training**: a fresh model is trained on the full training set.
5. **Evaluation**: prints accuracy, F1, ROC AUC, a confusion matrix, and a classification report on the held-out test set.
6. **Saving**: the model, threshold, accuracy, and a flag indicating real vs. demo data are saved with `joblib`.

### Model architecture

```
                    +--> Logistic Regression (word + char TF-IDF + style) --+
Review text ------->+--> Calibrated LinearSVC (word + char TF-IDF + style) -+--> Logistic Regression --> P(fake)
                    +--> Gradient Boosting (style features only) -----------+      (meta-classifier)
```

## Output Files

| File | Description |
|------|-------------|
| `fake_review.joblib` | Saved model bundle (model, threshold, accuracy, data flag) |
| `predictions.csv` | Batch-mode results |

## Limitations

- The detector learns patterns from its training data; it may not generalize to other domains (e.g., hotel reviews vs. product reviews) or to carefully written fake reviews.
- Reported accuracy is measured on a random split of the same dataset and may be optimistic for real-world data.
- Results are probabilistic signals, not proof. Use them to flag reviews for human review rather than as sole evidence.
- Training the stacked ensemble can be slow on large datasets (hundreds of thousands of reviews).
- Models saved with `joblib` should only be loaded from trusted sources.

## Troubleshooting

- **"Column not found"**: pass `--text-col` and/or `--label-col` with the exact column names.
- **"Dataset needs both fake and genuine reviews"**: check that your label values are among the recognized ones listed above.
- **Unrealistically high accuracy (~100%)**: you are likely running on the synthetic demo data. Supply a real dataset with `--data`.
- **Slow training**: use a smaller sample of your dataset, or reduce `max_features` in `features()`.
