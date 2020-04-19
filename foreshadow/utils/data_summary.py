# noqa
from collections import OrderedDict

import numpy as np


# TODO fix the noqa


def mode_freq(s, count=10):  # noqa
    vc = s.value_counts()
    unique = len(vc)
    vc = vc.nlargest(count).reset_index()
    vc["PCT"] = vc.iloc[:, -1] / s.size
    return unique, vc.to_dict(orient="split")["data"]


def get_outliers(s, count=10):  # noqa
    out_ser = s[np.abs(s - s.mean()) > (3 * s.std())]
    out_df = out_ser.to_frame()
    out_df["selector"] = out_ser.abs()

    return out_df.loc[out_df["selector"].nlargest(count).index].iloc[:, 0]


def standard_col_summary(df):  # noqa
    data = df.iloc[:, 0]
    count = len(data)
    nan_pct = data.isnull().sum() * 100.0 / count
    unique, top10 = mode_freq(data)
    result = OrderedDict(
        [
            ("count", count),
            ("nan_percent", nan_pct),
            ("unique", unique),
            ("top10", top10),
        ]
    )
    return result
