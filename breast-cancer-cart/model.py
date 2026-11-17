"""
model.py - Machine-learning logic for the CART Breast Cancer classifier.

CART (Classification And Regression Trees) is implemented by scikit-learn's
DecisionTreeClassifier. With criterion="gini" it picks, at every node, the
binary split that gives the largest drop in Gini impurity.

Note: in sklearn's dataset, target 0 = malignant and target 1 = benign.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# Features the user types in on the Prediction page. The other 22 features
# are estimated from these (see CARTModel.train).
INPUT_FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points",
]

MAX_DEPTH = 4          # limits tree depth to reduce overfitting
RANDOM_STATE = 42      # makes results reproducible


class CARTModel:
    def __init__(self):
        self.data = None            # sklearn Bunch
        self.df = None              # features + target as a DataFrame
        self.class_names = None     # ['malignant', 'benign']
        self.feature_names = None
        self.clf = None             # trained DecisionTreeClassifier
        self.is_trained = False
        self.metrics = {}
        self.cm = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.load_data()

    # ------------------------------------------------------------------ data
    def load_data(self):
        """Load the Breast Cancer Wisconsin dataset into a DataFrame."""
        self.data = load_breast_cancer()
        self.feature_names = [str(f) for f in self.data.feature_names]
        self.class_names = [str(c) for c in self.data.target_names]
        self.df = pd.DataFrame(self.data.data, columns=self.feature_names)
        self.df["target"] = self.data.target

    def dataset_info(self):
        """Basic facts shown on the Dashboard."""
        counts = self.df["target"].value_counts()
        return {
            "records": len(self.df),
            "features": len(self.feature_names),
            "classes": len(self.class_names),
            "malignant": int(counts.get(0, 0)),
            "benign": int(counts.get(1, 0)),
            "missing": int(self.df.isnull().sum().sum()),
        }

    def sample_records(self, n=5):
        """First n rows (a few columns) for display."""
        cols = INPUT_FEATURES[:5] + ["target"]
        return self.df[cols].head(n)

    def preprocess(self):
        """
        Basic preprocessing: check for missing values and duplicates.
        Feature scaling is NOT needed because decision trees split on
        thresholds and are not affected by feature scale.
        """
        if self.df.isnull().sum().sum() > 0:
            self.df = self.df.dropna()
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        X = self.df[self.feature_names]
        y = self.df["target"]
        return X, y

    # -------------------------------------------------------------- training
    def train(self):
        """Preprocess, split 80:20, train CART with Gini, evaluate."""
        X, y = self.preprocess()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

        self.clf = DecisionTreeClassifier(
            criterion="gini", max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
        self.clf.fit(self.X_train, self.y_train)

        # The GUI only asks for 8 inputs but the tree was trained on 30
        # features. A small linear regression estimates the other 22 from
        # the 8 entered ones (they are strongly correlated, e.g. "worst
        # radius" vs "mean radius"). Fitted on training data only.
        self.other_features = [f for f in self.feature_names
                               if f not in INPUT_FEATURES]
        self.estimator = LinearRegression().fit(
            self.X_train[INPUT_FEATURES], self.X_train[self.other_features])

        y_pred = self.clf.predict(self.X_test)
        # Malignant (0) is treated as the "positive" class: missing a cancer
        # (false negative) is the costly error, so recall matters most.
        self.metrics = {
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "Precision": precision_score(self.y_test, y_pred, pos_label=0),
            "Recall": recall_score(self.y_test, y_pred, pos_label=0),
            "F1-score": f1_score(self.y_test, y_pred, pos_label=0),
            "Train accuracy": self.clf.score(self.X_train, self.y_train),
        }
        self.cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
        self.is_trained = True
        return self.metrics

    # ------------------------------------------------------------ prediction
    def feature_ranges(self):
        """(min, max, mean) of each input feature, for hints and validation."""
        out = {}
        for f in INPUT_FEATURES:
            out[f] = (self.df[f].min(), self.df[f].max(), self.df[f].mean())
        return out

    def class_example(self, label):
        """Average values of the input features for class 0 or 1."""
        return self.df[self.df["target"] == label][INPUT_FEATURES].mean()

    def predict(self, values):
        """
        values: dict {feature name: float} for INPUT_FEATURES.
        Returns (class name, probability of that class).
        """
        if not self.is_trained:
            raise RuntimeError("Train the model first.")
        # Estimate the 22 features the user did not enter, then add the
        # user's 8 values and put the columns in the original order.
        given = pd.DataFrame([values], columns=INPUT_FEATURES)
        est = pd.DataFrame(self.estimator.predict(given),
                           columns=self.other_features)
        sample = pd.concat([given, est], axis=1)[self.feature_names]
        sample.columns = [str(c) for c in sample.columns]
        pred = int(self.clf.predict(sample)[0])
        proba = float(self.clf.predict_proba(sample)[0][pred])
        return self.class_names[pred], proba
