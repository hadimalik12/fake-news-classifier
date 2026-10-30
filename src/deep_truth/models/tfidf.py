"""Construct the TF-IDF and logistic regression baseline."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from deep_truth.config import Config


def build_tfidf_model(config: Config, parameters: dict) -> Pipeline:
    settings = config.data["tfidf"]
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=settings["lowercase"],
            analyzer=settings["analyzer"],
            norm=settings["norm"],
            use_idf=settings["use_idf"],
            ngram_range=tuple(parameters["ngram_range"]),
            min_df=parameters["min_df"],
            max_features=parameters["max_features"],
        )),
        ("classifier", LogisticRegression(
            solver=settings["solver"],
            max_iter=settings["max_iter"],
            random_state=config.seed,
            C=parameters["C"],
        )),
    ])
