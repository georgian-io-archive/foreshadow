import pytest


def test_check_df_passthrough():
    import pandas as pd
    from foreshadow.utils import check_df

    input_df = pd.DataFrame([1, 2, 3, 4])
    assert input_df.equals(check_df(input_df))


def test_check_df_rename_cols():
    import pandas as pd
    from foreshadow.utils import check_df

    input_df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "A"])
    input_df = check_df(input_df)
    assert input_df.columns.tolist() == ["A", "A.1"]


def test_check_df_convert_to_df():
    import numpy as np
    import pandas as pd
    from foreshadow.utils import check_df

    input_arr = np.array([1, 2, 3, 4])
    input_df = check_df(input_arr)
    assert isinstance(input_df, pd.DataFrame)


def test_check_df_convert_series_to_df():
    import pandas as pd
    from foreshadow.utils import check_df

    input_ser = pd.Series([1, 2, 3, 4])
    input_df = check_df(input_ser)
    assert isinstance(input_df, pd.DataFrame)


def test_check_df_raises_on_invalid():
    from foreshadow.utils import check_df

    input_df = None
    with pytest.raises(ValueError) as e:
        input_df = check_df(input_df)
    assert str(e.value) == (
        "Invalid input type, neither pd.DataFrame, pd.Series, np.ndarray, nor"
        " list"
    )


def test_check_df_passthrough_none():
    from foreshadow.utils import check_df

    input_df = None
    assert check_df(input_df, ignore_none=True) is None


def test_check_df_single_column():
    import numpy as np
    from foreshadow.utils import check_df

    input_arr = np.arange(8).reshape((4, 2))

    with pytest.raises(ValueError) as e:
        _ = check_df(input_arr, single_column=True)

    assert str(e.value) == ("Input Dataframe must have only one column")


def test_module_not_installed():
    from foreshadow.utils import check_module_installed

    assert check_module_installed("not_installed") is False


def test_module_installed():
    from foreshadow.utils import check_module_installed

    assert check_module_installed("sys") is True


def test_check_transformer_imports(capsys):
    from foreshadow.utils import check_transformer_imports

    inter_trans, exter_trans = check_transformer_imports()
    out, err = capsys.readouterr()

    assert out.startswith("Loaded")
    assert len(inter_trans) > 0
    assert len(exter_trans) > 0


def test_check_transformer_imports_no_output(capsys):
    from foreshadow.utils import check_transformer_imports

    check_transformer_imports(printout=False)
    out, err = capsys.readouterr()

    assert len(out) == 0
