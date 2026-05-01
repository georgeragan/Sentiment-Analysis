import os
import mlflow
import dagshub
import pandas as pd
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings 
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output   

# ---------------------------
# MLflow setup
# ---------------------------
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

dagshub.init(
    repo_owner='georgeragan',
    repo_name='Sentiment-Analysis',
    mlflow=True
)

mlflow.set_experiment("Logistic GridSearch Final")


# ---------------------------
# Data
# ---------------------------
def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return (X_train, X_test, y_train, y_test), vectorizer


# ---------------------------
# YOUR FUNCTION (only grid updated)
# ---------------------------
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):

    # ✅ ONLY change → added class_weight
    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "class_weight": [None, "balanced"]
    }

    with mlflow.start_run(run_name="logistic_gridsearch_main"):

        grid_search = GridSearchCV(
            LogisticRegression(max_iter=1000),
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        # ----------------------------
        # LOG ALL COMBINATIONS
        # ----------------------------
        for params, mean_score, std_score in zip(
            grid_search.cv_results_["params"],
            grid_search.cv_results_["mean_test_score"],
            grid_search.cv_results_["std_test_score"]
        ):

            with mlflow.start_run(run_name=f"LR_{params}", nested=True):

                try:
                    model = LogisticRegression(
                        max_iter=1000,
                        **params
                    )
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    mlflow.log_params(params)

                    mlflow.log_metrics({
                        "accuracy": accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred, zero_division=0),
                        "recall": recall_score(y_test, y_pred, zero_division=0),
                        "f1_score": f1_score(y_test, y_pred, zero_division=0),
                        "mean_cv_score": mean_score,
                        "std_cv_score": std_score
                    })

                    print(f"Params: {params} | F1: {mean_score:.4f}")

                except Exception as e:
                    print(f"Failed params {params}: {e}")

        # ----------------------------
        # BEST MODEL
        # ----------------------------
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)

        mlflow.sklearn.log_model(best_model, "model")

        print(f"\nBest Params: {best_params} | Best F1: {best_f1:.4f}")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data("notebooks/data.csv")
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)