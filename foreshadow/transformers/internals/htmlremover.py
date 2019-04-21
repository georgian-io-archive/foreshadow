import re

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from foreshadow.utils import check_df


HTML_REGEX = re.compile(r"<[^<]+?>")
IS_HTML_REGEX = re.compile(
    r"<(br|basefont|hr|input|source|frame|param|area|meta|!--|col|link|option|base|img|wbr|!DOCTYPE).*?>|<(a|abbr|acronym|address|applet|article|aside|audio|b|bdi|bdo|big|blockquote|body|button|canvas|caption|center|cite|code|colgroup|command|datalist|dd|del|details|dfn|dialog|dir|div|dl|dt|em|embed|fieldset|figcaption|figure|font|footer|form|frameset|head|header|hgroup|h1|h2|h3|h4|h5|h6|html|i|iframe|ins|kbd|keygen|label|legend|li|map|mark|menu|meter|nav|noframes|noscript|object|ol|optgroup|output|p|pre|progress|q|rp|rt|ruby|s|samp|script|section|select|small|span|strike|strong|style|sub|summary|sup|table|tbody|td|textarea|tfoot|th|thead|time|title|tr|track|tt|u|ul|var|video).*?<\/\2>"
)


class HTMLRemover(BaseEstimator, TransformerMixin):
    """Removes html tags from text data."""

    def is_html(input_str):
        return IS_HTML_REGEX.match(input_str) is not None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_df(X, single_column=True).iloc[:, 0]
        X = X.str.replace(HTML_REGEX, "")

        return X
