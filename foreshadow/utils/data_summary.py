# noqa
from collections import OrderedDict

import numpy as np


# TODO fix the noqa


def mode_freq(s, count=10):  # noqa
    mode = s.mode().values.tolist()
    vc = s.value_counts().nlargest(count).reset_index()
    vc["PCT"] = vc.iloc[:, -1] / s.size
    return mode, vc.values.tolist()


def get_outliers(s, count=10):  # noqa
    out_ser = s[np.abs(s - s.mean()) > (3 * s.std())]
    out_df = out_ser.to_frame()
    out_df["selector"] = out_ser.abs()

    return out_df.loc[out_df["selector"].nlargest(count).index].iloc[:, 0]


def standard_col_summary(df):  # noqa
    data = df.iloc[:, 0]
    nan_num = int(data.isnull().sum())
    mode, top10 = mode_freq(data)

    return OrderedDict([("nan", nan_num), ("mode", mode), ("top10", top10)])
