"""
Train a Random Forest classifier on the combined ASL letter + custom
gesture landmark CSVs.

Usage:
    python train_model.py \
        --inputs ../data/landmarks/asl_landmarks.csv ../data/landmarks/custom_space.csv \
        --output ../models/asl_classifier.pkl

Why Random Forest:
  - Works very well on small-to-medium tabular feature sets like 63
    landmark coordinates.
  - Trains in seconds on a laptop CPU, no GPU needed.
  - Handles the nonlinear decision boundaries between hand poses without
    much tuning.
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True,
                         help='One or more landmark CSV files to combine')
    parser.add_argument('--output', required=True, help='Where to save the trained model (.pkl)')
    args = parser.parse_args()

    dfs = [pd.read_csv(p) for p in args.inputs]
    df = pd.concat(dfs, ignore_index=True)

    print(f'Loaded {len(df)} samples across {df["label"].nunique()} classes:')
    print(df['label'].value_counts())

    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('\n' + classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(clf, args.output)
    print(f'\nModel saved to {args.output}')


if __name__ == '__main__':
    main()
