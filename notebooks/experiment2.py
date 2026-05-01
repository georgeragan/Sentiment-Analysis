import os
import logging
import mlflow
import pandas as pd
import dagshub
from dotenv import load_dotenv

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


# ---------------------------
# 1. Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------
# 2. Load ENV + MLflow setup
# ---------------------------
load_dotenv()

try:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI missing")

    mlflow.set_tracking_uri(tracking_uri)

    dagshub.init(
        repo_owner='georgeragan',
        repo_name='Sentiment-Analysis',
        mlflow=True
    )

    mlflow.set_experiment("Sentiment Analysis All Models")

    logging.info("MLflow initialized")

except Exception as e:
    logging.error(f"Setup failed: {e}")
    raise


# ---------------------------
# 3. Load data
# ---------------------------
def load_data(path):
    try:
        df = pd.read_csv(path)
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        logging.info("Data loaded")
        return df
    except Exception as e:
        logging.error(f"Data load failed: {e}")
        raise


# ---------------------------
# 4. Models
# ---------------------------
VECTORIZERS = {
    'bow': CountVectorizer(),
    'tfidf': TfidfVectorizer()
}

CLASSIFIERS = {
    'logistic_regression': LogisticRegression(max_iter=1000),
    'naive_bayes': MultinomialNB(),
    'random_forest': RandomForestClassifier(n_estimators=100),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100),
    'xgboost': XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss'
    )
}


# ---------------------------
# 5. Training
# ---------------------------
def train_and_evaluate(df):
    try:
        X = df['review']
        y = df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        for vec_name, vectorizer in VECTORIZERS.items():

            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            for clf_name, clf in CLASSIFIERS.items():

                try:
                    # ✅ IMPORTANT: new run per combo
                    with mlflow.start_run(run_name=f"{vec_name}_{clf_name}"):

                        clf.fit(X_train_vec, y_train)
                        y_pred = clf.predict(X_test_vec)

                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred)
                        rec = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)

                        # params
                        mlflow.log_param("vectorizer", vec_name)
                        mlflow.log_param("classifier", clf_name)

                        # metrics
                        mlflow.log_metric("accuracy", acc)
                        mlflow.log_metric("precision", prec)
                        mlflow.log_metric("recall", rec)
                        mlflow.log_metric("f1_score", f1)

                        logging.info(
                            f"{vec_name} + {clf_name} → F1: {f1:.4f}"
                        )

                except Exception as e:
                    logging.error(
                        f"Failed model ({vec_name}, {clf_name}): {e}"
                    )

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


# ---------------------------
# 6. Main
# ---------------------------
if __name__ == "__main__":
    try:
        df = load_data("notebooks/data.csv")
        train_and_evaluate(df)
        logging.info("All experiments completed")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")