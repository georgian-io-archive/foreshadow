def test_scaler_normal():
    import numpy as np
    import scipy.stats as ss

    from ...transformers.smart import SmartScaler
    from ...transformers import StandardScaler

    np.random.seed(0)
    normal_data = ss.norm.rvs(size=100)
    smart = SmartScaler()
    assert isinstance(smart.fit(normal_data), StandardScaler)


def test_scaler_unifrom():
    import numpy as np
    import scipy.stats as ss

    from ...transformers.smart import SmartScaler
    from ...transformers import MinMaxScaler

    np.random.seed(0)
    uniform_data = ss.uniform.rvs(size=100)
    smart = SmartScaler()
    assert isinstance(smart.fit(uniform_data), MinMaxScaler)


def test_scaler_neither():
    import numpy as np
    import scipy.stats as ss

    from ...transformers.smart import SmartScaler
    from sklearn.pipeline import Pipeline

    np.random.seed(0)
    lognorm_data = ss.lognorm.rvs(size=100, s=0.954)  # one example
    print(lognorm_data)
    smart = SmartScaler()
    assert isinstance(smart.fit(lognorm_data), Pipeline)
