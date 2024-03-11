from typing import Optional

from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from config import AppConfigSettings

ADDITIONAL_STOPWORDS = frozenset(
    ["dress", "model", "wears", "fit", "true", "to", "size", "uk", "us", "tall", "cm"]
)
CLASSIFIER_STEP_NAME = "classifier"
VECTORIZER_STEP_NAME = "vectorizer"
SCALER_STEP_NAME = "scaler"


class PipelineFactory:

    @staticmethod
    def fetch_classifier(config: AppConfigSettings) -> tuple[str, ClassifierMixin]:
        match classifier_name := config.classifier.classifier_name:
            case "SGDClassifier":
                classifier = SGDClassifier(
                    loss=config.classifier.classifier_loss,
                    penalty=config.classifier.classifier_penalty,
                    alpha=config.classifier.classifier_alpha,
                    random_state=config.train.seed,
                )
            case _:
                raise ValueError(f"Undefined model: {classifier_name}")
        return CLASSIFIER_STEP_NAME, classifier

    @staticmethod
    def fetch_vectorizer(
        config: AppConfigSettings, **kwargs
    ) -> tuple[str, TransformerMixin]:
        match vectorizer_name := config.vectorizer.vectorizer_name:
            case "tfid_vectorizer":
                stops = kwargs.get("stops", None)
                vectorizer = TfidfVectorizer(stop_words=stops)
            case _:
                raise ValueError(f"Undefined vectorizer: {vectorizer_name}")
        return VECTORIZER_STEP_NAME, vectorizer

    @staticmethod
    def fetch_scaler(
        config: AppConfigSettings, **kwargs
    ) -> tuple[str, TransformerMixin]:
        match scaler_name := config.scaler.scaler_name:
            case "standard_scaler":
                with_mean = kwargs.get("with_mean", False)
                scaler = StandardScaler(with_mean=with_mean)
            case "max_abs_scaler":
                scaler = MaxAbsScaler()
            case None:
                scaler = None
            case _:
                raise ValueError(f"Undefined scaler: {scaler_name}")
        return SCALER_STEP_NAME, scaler

    @staticmethod
    def build_stopword_vocabulary(
        custom_stopwords: Optional[frozenset[str]],
    ) -> frozenset[str]:
        return (
            text.ENGLISH_STOP_WORDS
            | ADDITIONAL_STOPWORDS
            | (custom_stopwords or frozenset())
        )

    def build_classifier_pipeline(
        self,
        config: AppConfigSettings,
        custom_stopwords: Optional[frozenset[str]],
    ) -> Pipeline:
        stopwords = self.build_stopword_vocabulary(custom_stopwords=custom_stopwords)

        steps = []
        steps.append(
            PipelineFactory.fetch_vectorizer(config=config, stopwords=stopwords)
        )
        steps.append(PipelineFactory.fetch_scaler(config=config))
        steps.append(PipelineFactory.fetch_classifier(config=config))

        # if transformers are computationally expensive we can cache (eg. for grid search)
        # from tempfile import mkdtemp
        # cachedir = mkdtemp()
        # return Pipeline(steps=steps, memory=cachedir)
        return Pipeline(steps=steps, memory=None)
