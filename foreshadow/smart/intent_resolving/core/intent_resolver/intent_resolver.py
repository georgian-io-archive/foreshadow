"""Class definition for the IntentResolver class."""

import logging
from pathlib import Path
from typing import Union

import pandas as pd

from .. import io
from ..data_set_parsers import DataFrameDataSetParser
from ..secondary_featurizers import (
    FeaturizerCurator,
    RawDataSetFeaturizerViaLambda,
)


class IntentResolver:
    """
    Identify the intent of a feature column.

    Supported intents are listed in the LABEL_MAPPING class attribute.

    Attributes:
        raw {pd.DataFrame} -- Raw data set to analyze
        components_path {Path} -- Path to saved and trained components
        model -- Trained intent resolver model, saved in components file
        scaler -- Trained feature normalization scaler, saved in components file

    Class attributes:
        LABEL_MAPPING -- Supported intents.
    """

    LABEL_MAPPING = {
        -1: "Neither",
        0: "Numerical",
        # 1: 'Needs Extraction',
        2: "Categorical",
        # 3: 'Not Generalizable',
        # 4: 'Context Specific'
    }

    def __init__(
        self,
        raw: pd.DataFrame,
        components_path: Path = Path(__file__).with_name(
            "resolver_components.pkl"
        ),
    ):
        """
        Init function.

        Arguments:
            raw {pd.DataFrame} -- Raw data set to analyze

        Keyword Arguments:
            components_path {Path} -- Path to saved and trained components.
                   (default: {Path(__file__).with_name('resolver_components.pkl')})

        Raises:
            FileNotFoundError -- If `components_path` file does not exist.
            TypeError -- If `raw` is not a pd.DataFrame.
        """
        if not Path(components_path).is_file():
            raise FileNotFoundError
        if not isinstance(raw, pd.DataFrame):
            raise TypeError("`raw` must be a pd.DataFrame.")

        self.parser = DataFrameDataSetParser(raw)
        self.components_path = Path(components_path)

        # Load trained assets
        components = io.from_pickle(self.components_path)
        self.model = components["model"]
        self.scaler = components["scaler"]
        self.parser.featurizers = FeaturizerCurator.from_config(
            func_config=components["function_featurizers_config"],
            text_config=components["text_featurizer_config"],
        )

        # Reset `RawDataSetFeaturizerViaLambda` attributes so that
        # featurization triggers properly for new data sets
        for f in self.parser.featurizers:
            if isinstance(f, RawDataSetFeaturizerViaLambda):
                if f.fast_load != False or f.save_dir is not None:
                    raise ValueError(
                        "RawDataSetFeaturizerViaLambda has incorrectly set "
                        "attrbutes. This means was not saved properly during "
                        "the training phase."
                    )

        self.__initialise_parser()

    def __initialise_parser(self) -> None:
        self.parser.load_data_set()
        self.parser.featurize_base()
        self.parser.featurize_secondary()

    def predict(
        self, threshold: float = 0.7, return_conf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Predict the intents of raw feature columns.

        Keyword Arguments:
            threshold {float}
                -- The minimum confidence before classifying a feature column as
                   numerical or categorical. Default value is "Neither".
            return_conf {bool}
                -- Return confidence of predictions. (default: {False})

        Returns:
            Union[pd.Series, pd.DataFrame]
            -- A pd.Series of predictions is returned if `return_bool` is False.
               Otherwise, a dataframe of prediction and confidence is returned.
        """
        X = self.parser.normalize_features(self.scaler)

        # Get predictions as pd.Categorical type
        predictions = pd.Categorical(
            map(IntentResolver.LABEL_MAPPING.get, self.model.predict(X)),
            categories=IntentResolver.LABEL_MAPPING.values(),
        )
        if predictions.isnull().any():
            logging.warning("Foreign categories detected in prediction.")

        # Get prediction confidences
        proba = self.model.predict_proba(X).max(axis=1)

        # Modify prediction and confidence for feature columns with unsure predictions
        for i, prob in enumerate(proba):
            if prob < threshold:
                predictions[i] = IntentResolver.LABEL_MAPPING[-1]
                proba[i] = 0

        if return_conf:
            return pd.DataFrame(
                {"intents": predictions, "confidence": proba},
                index=self.parser.raw.columns,
            )

        return pd.Series(
            pd.Categorical(
                predictions, categories=IntentResolver.LABEL_MAPPING.values()
            ),
            index=self.parser.raw.columns,
            name="intents",
        )
