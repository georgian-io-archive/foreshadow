def test_box_cox():
    import numpy as np
    import pandas as pd
    import scipy.stats as ss

    from foreshadow.transformers import BoxCoxTransformer

    np.random.seed(0)
    data = pd.DataFrame(ss.lognorm.rvs(size=100, s=0.954))
    bc = BoxCoxTransformer()
    bc_data = bc.fit_transform(data)
    assert ss.shapiro(bc_data)[1] > 0.05
    assert np.allclose(
        data.values.ravel(), bc.inverse_transform(bc_data).values.ravel()
    )
