def test_smart_scaler_normal():
    import numpy as np
    import scipy.stats as ss

    from foreshadow.transformers import SmartScaler
    from foreshadow.transformers import StandardScaler

    np.random.seed(0)
    normal_data = ss.norm.rvs(size=100)
    smart_scaler = SmartScaler()
    assert isinstance(smart_scaler.fit(normal_data), StandardScaler)


def test_smart_scaler_unifrom():
    import numpy as np
    import scipy.stats as ss

    from foreshadow.transformers import SmartScaler
    from foreshadow.transformers import MinMaxScaler

    np.random.seed(0)
    uniform_data = ss.uniform.rvs(size=100)
    smart_scaler = SmartScaler()
    assert isinstance(smart_scaler.fit(uniform_data), MinMaxScaler)


def test_smart_scaler_neither():
    import numpy as np
    import scipy.stats as ss

    from foreshadow.transformers import SmartScaler
    from sklearn.pipeline import Pipeline

    np.random.seed(0)
    lognorm_data = ss.lognorm.rvs(size=100, s=0.954)  # one example
    smart_scaler = SmartScaler()
    assert isinstance(smart_scaler.fit(lognorm_data), Pipeline)


def test_smart_encoder_less_than_30_levels():
    import numpy as np
    import scipy.stats as ss

    from foreshadow.transformers import SmartCoder
    from foreshadow.transformers import OneHotEncoder

    np.random.seed(0)
    leq_30_random_data = np.random.choice(30, size=500)
    smart_coder = SmartCoder()
    assert isinstance(smart_coder.fit(leq_30_random_data), OneHotEncoder)


def test_smart_encoder_more_than_30_levels():
    import numpy as np
    import scipy.stats as ss

    from foreshadow.transformers import SmartCoder
    from foreshadow.transformers import HashingEncoder

    np.random.seed(0)
    gt_30_random_data = np.random.choice(31, size=500)
    smart_coder = SmartCoder()
    assert isinstance(smart_coder.fit(gt_30_random_data), HashingEncoder)


def test_smart_impute_simple_none():
    import pandas as pd
    from foreshadow.transformers import SmartSimpleImputer

    impute = SmartSimpleImputer(threshold=0.05)
    df = pd.read_csv("./foreshadow/tests/data/heart-h.csv")

    data = df[["chol"]]

    impute.fit(data)
    out = impute.transform(data)

    assert data.equals(out)


def test_smart_impute_simple_mean():
    import pandas as pd
    from foreshadow.transformers import SmartSimpleImputer

    impute = SmartSimpleImputer()
    df = pd.read_csv("./foreshadow/tests/data/heart-h.csv")

    data = df[["chol"]]

    impute.fit(data)
    out = impute.transform(data)
    truth = pd.read_csv("./foreshadow/tests/data/heart-h_impute_mean.csv", index_col=0)

    assert out.equals(truth)


def test_smart_impute_simple_median():
    import pandas as pd
    import numpy as np
    from foreshadow.transformers import SmartSimpleImputer

    impute = SmartSimpleImputer()
    df = pd.read_csv("./foreshadow/tests/data/heart-h.csv")

    data = df["chol"].values
    data = np.append(data, [2 ** 10] * 100)

    impute.fit(data)
    out = impute.transform(data)
    truth = pd.read_csv(
        "./foreshadow/tests/data/heart-h_impute_median.csv", index_col=0
    )

    assert out.equals(truth)


def test_smart_impute_multiple():
    import numpy as np
    import pandas as pd
    from foreshadow.transformers import SmartMultiImputer

    impute = SmartMultiImputer()
    df = pd.read_csv("./foreshadow/tests/data/heart-h.csv")

    data = df[["thalach", "chol", "trestbps", "age"]]

    impute.fit(data)
    out = impute.transform(data)
    truth = pd.read_csv("./foreshadow/tests/data/heart-h_impute_multi.csv", index_col=0)

    assert np.allclose(truth.values, out.values)
