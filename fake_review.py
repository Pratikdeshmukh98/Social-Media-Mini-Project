import argparse
import os
import random
import re
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
TEXT = ["text", "text_", "review", "review_text", "reviewtext", "content", "comment", "body"]
LABEL = ["label", "class", "category", "fake", "is_fake", "deceptive", "target", "spam"]
FAKE = {"cg", "fake", "deceptive", "spam", "1", "true", "yes"}
REAL = {"or", "genuine", "truthful", "real", "0", "false", "original", "no", "ham"}
WORD = re.compile(r"[A-Za-z']+")
STRONG = {"amazing", "awesome", "perfect", "best", "excellent", "fantastic", "incredible", "love", "loved",
          "wonderful", "outstanding", "superb", "great", "terrible", "worst", "horrible", "awful",
          "unbelievable", "flawless", "must", "highly", "definitely", "absolutely", "totally"}
FIRST = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
PROMO = {"buy", "order", "recommend", "recommended", "deal", "discount", "shop", "store", "seller", "purchase"}
SIGNALS = [(4, 1.0, "heavy exclamation use"), (13, 0.12, "extreme or superlative wording"),
           (14, 0.06, "promotional wording"), (11, 0.15, "repeated phrases")]


class Style(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        return np.array([self.row(str(t)) for t in X], dtype=float)

    @staticmethod
    def row(text):
        n, words = max(len(text), 1), WORD.findall(text)
        low = [w.lower() for w in words]
        k = max(len(words), 1)
        sents = max(len([s for s in re.split(r"[.!?]+", text) if s.strip()]), 1)
        top = pd.Series(low).value_counts().iloc[0] if low else 0
        bi = list(zip(low, low[1:]))
        frac = lambda g: sum(w in g for w in low) / k
        return [
            np.log1p(n), np.log1p(len(words)), np.mean([len(w) for w in words]) if words else 0.0,
            len(words) / sents, text.count("!") / n * 100, text.count("?") / n * 100,
            sum(c.isupper() for c in text) / n, sum(len(w) > 1 and w.isupper() for w in words) / k,
            sum(c.isdigit() for c in text) / n, len(set(low)) / k, top / k,
            (len(bi) - len(set(bi))) / max(len(bi), 1),
            frac(FIRST), frac(STRONG), frac(PROMO), len(re.findall(r"https?://|www\.", text)),
        ]


def features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.95, sublinear_tf=True, strip_accents="unicode", max_features=150000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)),
        ("style", Pipeline([("s", Style()), ("z", StandardScaler())])),
    ])


def build_model():
    lr = Pipeline([("f", features()), ("c", LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced"))])
    svm = Pipeline([("f", features()), ("c", CalibratedClassifierCV(LinearSVC(C=0.4, class_weight="balanced"), cv=3))])
    gb = Pipeline([("s", Style()), ("c", HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6, l2_regularization=1.0, random_state=42))])
    return StackingClassifier(
        [("lr", lr), ("svm", svm), ("gb", gb)], final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba", cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=-1)


def clean(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>|&\w+;", " ", str(text))).strip()


def find(cols, names, override):
    lowered = {c.lower(): c for c in cols}
    hit = override or next((lowered[n] for n in names if n in lowered), None)
    if hit not in cols:
        sys.exit(f"Column not found. Available: {list(cols)}. Use --text-col / --label-col.")
    return hit


def load_data(path, text_col, label_col):
    df = pd.read_csv(path)
    t, lab = find(df.columns, TEXT, text_col), find(df.columns, LABEL, label_col)
    y = df[lab].astype(str).str.strip().str.lower().map(lambda v: 1 if v in FAKE else 0 if v in REAL else None)
    out = pd.DataFrame({"text": df[t].map(clean), "label": y}).dropna()
    out["label"] = out["label"].astype(int)
    out = out[out["text"].str.len() >= 10].drop_duplicates("text").reset_index(drop=True)
    if out["label"].nunique() < 2:
        sys.exit("Dataset needs both fake and genuine reviews.")
    return out


def demo_data(n=500, seed=7):
    rng = random.Random(seed)
    f = [["Absolutely amazing", "Best purchase ever", "Perfect in every way", "Highly recommended", "Must buy"],
         ["I love it so much", "everyone should buy this", "totally worth every penny", "you will not regret it"],
         ["Buy now!!!", "Order today!", "Best deal ever!!!", "Trust me, just buy it!"]]
    r = [["I used this for about three weeks", "Bought it last month", "Delivery took six days", "Mixed feelings so far"],
         ["the battery lasts a day but the charger feels flimsy", "setup was easy yet the manual is confusing",
          "it is heavier than I expected", "the staff were polite although the wait was long"],
         ["Would consider buying again if the price drops.", "Overall okay, not outstanding.", "Returned the first unit because of a scratch."]]
    rows = []
    for _ in range(n):
        fake = f"{rng.choice(f[0])}! {rng.choice(f[1]).capitalize()}. {rng.choice(f[2])}"
        real = f"{rng.choice(r[0])}, {rng.choice(r[1])}. {rng.choice(r[2])}"
        rows += [(fake, 1), (fake if rng.random() < 0.08 else real, 0)]
    return pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates("text").reset_index(drop=True)


def tune_threshold(model, X, y):
    p, grid = model.predict_proba(X)[:, 1], np.arange(0.2, 0.81, 0.02)
    return float(grid[int(np.argmax([f1_score(y, (p >= t).astype(int)) for t in grid]))])


def data_path(args):
    return args.data or next((p for p in ("reviews.csv", "data/reviews.csv") if os.path.exists(p)), None)


def train(args):
    path = data_path(args)
    if path:
        df = load_data(path, args.text_col, args.label_col)
    else:
        print("No dataset found (use --data FILE or save reviews.csv next to this script).")
        print("Training on tiny synthetic demo data instead, so the accuracy below is NOT realistic.")
        df = demo_data()
    print(f"Training on {len(df)} reviews, please wait...")
    X_tr, X_te, y_tr, y_te = train_test_split(df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=1)
    threshold = tune_threshold(build_model().fit(X_fit, y_fit), X_val, y_val)
    model = build_model().fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    pred = (p >= threshold).astype(int)
    acc = accuracy_score(y_te, pred)
    print(f"Test accuracy: {acc:.2%} | F1: {f1_score(y_te, pred):.4f} | ROC AUC: {roc_auc_score(y_te, p):.4f}")
    print(confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred, target_names=["genuine", "fake"], digits=4))
    joblib.dump({"model": model, "threshold": threshold, "accuracy": acc, "real": path is not None}, args.model)


def interactive(bundle):
    print(f"\nFake Review Detector | model accuracy: {bundle['accuracy']:.2%} | type q to quit")
    while True:
        try:
            text = clean(input("\nEnter a review: "))
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"q", "quit", "exit"}:
            break
        if len(text) < 5:
            print("Please enter a longer review.")
            continue
        p, feats = bundle["model"].predict_proba([text])[0, 1], Style.row(text)
        print(f"Verdict          : {'FAKE' if p >= bundle['threshold'] else 'GENUINE'}")
        print(f"Fake probability : {p:.2%}")
        print(f"Model accuracy   : {bundle['accuracy']:.2%}")
        print("Signals          : " + (", ".join(m for i, t, m in SIGNALS if feats[i] > t) or "none notable"))


def batch(bundle, args):
    df = pd.read_csv(args.batch)
    df["fake_score"] = bundle["model"].predict_proba(df[find(df.columns, TEXT, args.text_col)].map(clean))[:, 1]
    df["prediction"] = np.where(df["fake_score"] >= bundle["threshold"], "fake", "genuine")
    df.to_csv(args.output, index=False)
    print(f"Predictions written to {args.output}")


def main():
    ap = argparse.ArgumentParser(description="Fake review detection system")
    for flag in ["--data", "--text-col", "--label-col", "--batch"]:
        ap.add_argument(flag)
    ap.add_argument("--model", default="fake_review.joblib")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    if args.retrain or not os.path.exists(args.model) or (data_path(args) and not joblib.load(args.model).get("real")):
        train(args)
    bundle = joblib.load(args.model)
    batch(bundle, args) if args.batch else interactive(bundle)


if __name__ == "__main__":
    main()
