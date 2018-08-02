def test_scaler_normal():
    import numpy as np
    import scipy.stats as ss

    from foreshadow.transformers import SmartScaler
    from foreshadow.transformers import StandardScaler

    np.random.seed(0)
    normal_data = ss.norm.rvs(size=100)
    smart_scaler = SmartScaler()
    assert isinstance(smart_scaler.fit(normal_data), StandardScaler)


def test_scaler_unifrom():
    import numpy as np
    import scipy.stats as ss

    from foreshadow.transformers import SmartScaler
    from foreshadow.transformers import MinMaxScaler

    np.random.seed(0)
    uniform_data = ss.uniform.rvs(size=100)
    smart_scaler = SmartScaler()
    assert isinstance(smart_scaler.fit(uniform_data), MinMaxScaler)


def test_scaler_neither():
    import numpy as np
    import scipy.stats as ss

    from foreshadow.transformers import SmartScaler
    from sklearn.pipeline import Pipeline

    np.random.seed(0)
    lognorm_data = ss.lognorm.rvs(size=100, s=0.954)  # one example
    smart_scaler = SmartScaler()
    assert isinstance(smart_scaler.fit(lognorm_data), Pipeline)


def test_encoder_less_than_30_levels():
    import numpy as np
    import scipy.stats as ss

    from foreshadow.transformers import SmartCoder
    from foreshadow.transformers import OneHotEncoder

    np.random.seed(0)
    leq_30_random_data = np.random.choice(30, size=500)
    smart_coder = SmartCoder()
    assert isinstance(smart_coder.fit(leq_30_random_data), OneHotEncoder)


def test_encoder_more_than_30_levels():
    import numpy as np
    import scipy.stats as ss

    from foreshadow.transformers import SmartCoder
    from foreshadow.transformers import HashingEncoder

    np.random.seed(0)
    gt_30_random_data = np.random.choice(31, size=500)
    smart_coder = SmartCoder()
    assert isinstance(smart_coder.fit(gt_30_random_data), HashingEncoder)
