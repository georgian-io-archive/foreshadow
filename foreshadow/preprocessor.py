"""
Preprocesser
"""

from sklearn.base import TransformerMixin, BaseEstimator
from .intents.general_intents import GenericIntent

class Preprocessor(TransformerMixin, BaseEstimator):

    def __init__(self):
        self.pipeline = None
        self.intent_rslts = {} # [col_name][(priority, intent_class), ...]
        self.intent_mapping = {} # [col_name][selected_intent_class]

    def map_intents(self, X_df):
        columns = X_df.columns
        if len(x) > len(set(x)):
            raise ValueError('Input dataframe columns must not have the same name.')
        for c in columns:
            col_data = X_df.loc[:, [c]]
            valid_cols = [(i, k) for i, k in enumerate(GenericIntent.level_order_traverse()) if k.is_intent(col_data)]
            if len(valid_cols) == 0:
                self.intent_rslts[c] = None
                self.intent_mapping[c] = None
            else:
               self.intent_rslts[c] = valid_cols
               self.intent_mapping[c] = valid_cols[-1]
    