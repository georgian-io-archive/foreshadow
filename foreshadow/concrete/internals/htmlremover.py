"""HTML tag remover and helpers."""
import re

from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.utils import check_df
from foreshadow.wrapper import pandas_wrap


HTML_REGEX = re.compile(r"<[^<]+?>")
IS_HTML_REGEX = re.compile(
    r"<(br|basefont|hr|input|source|frame|param|area|meta|!--|col|link|option"
    r"|base|img|wbr|!DOCTYPE).*?>|<(a|abbr|acronym|address|applet|article"
    r"|aside|audio|b|bdi|bdo|big|blockquote|body|button|canvas|caption|center"
    r"|cite|code|colgroup|command|datalist|dd|del|details|dfn|dialog|dir|div"
    r"|dl|dt|em|embed|fieldset|figcaption|figure|font|footer|form|frameset"
    r"|head|header|hgroup|h1|h2|h3|h4|h5|h6|html|i|iframe|ins|kbd|keygen|label"
    r"|legend|li|map|mark|menu|meter|nav|noframes|noscript|object|ol|optgroup"
    r"|output|p|pre|progress|q|rp|rt|ruby|s|samp|script|section|select|small"
    r"|span|strike|strong|style|sub|summary|sup|table|tbody|td|textarea|tfoot"
    r"|th|thead|time|title|tr|track|tt|u|ul|var|video).*?<\/\2>"
)


@pandas_wrap
class HTMLRemover(BaseEstimator, TransformerMixin):
    """Removes html tags from text data."""

    @staticmethod
    def is_html(input_str):
        """Determine whether an input string contains HTML tags.

        Args:
            input_str (str): A string, potentially containing HTML tags

        Returns:
            bool

        """
        return IS_HTML_REGEX.match(input_str) is not None

    def fit(self, X, y=None):
        """Empty fit.

        Args:
            X: input observations
            y: input labels

        Returns:
            self

        """
        return self

    def transform(self, X, y=None):
        """Remove HTML tags from passed in strings.

        Args:
            X: input observations
            y: input labels

        Returns:
            transformed X

        """
        X = check_df(X, single_column=True).iloc[:, 0]
        X = X.str.replace(HTML_REGEX, "")

        return X
