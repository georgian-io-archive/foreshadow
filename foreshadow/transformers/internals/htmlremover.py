import re

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import lxml.html

from foreshadow.utils import check_df


HTML_REGEX = r"<[^<]+?>"


class HTMLRemover(BaseEstimator, TransformerMixin):
    """Removes html tags from text data."""

    def is_html(input_str):
        return lxml.html.fromstring(input_str).find(".//*") is not None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_df(X, single_column=True).iloc[:, 0]
        X = X.str.replace(HTML_REGEX, "")

        return X
