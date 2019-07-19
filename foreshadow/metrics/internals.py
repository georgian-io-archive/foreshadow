"""Useful metrics used internal and provided as part of source code."""
from foreshadow.metrics.metrics import metric


# ------------------------------------------------
@metric
def unique_count(feature):
    """Count number of unique values in feature.

    Args:
        feature: feature/column of pandas dataset

    Returns:
        number of unique values in feature as int.

    """
    return len(feature.value_counts())


@metric
def unique_count_bias(feature):
    """Difference of count of unique values relative to the length of feature.

    Args:
        feature: feature/column of pandas dataset

    Returns:
        Number of unique values relative to the length of dataset.

    """
    return len(feature) - len(feature.value_counts())


@metric
def unique_count_weight(feature):
    """Normalize count number of unique values relative to length of feature.

    Args:
        feature: feature/column of pandas dataset

    Returns:
        Normalized Number of unique values relative to length of feature.

    """
    return len(feature.value_counts()) / len(feature)


@metric
def regex_rows(feature, encoder):
    search_results = [encoder(row)[1] for row in feature]
    return sum([1 for sr in search_results if sr[-1] is not None]) / len(
        feature
    )


@metric
def avg_col_regex(feature, encoder, mode=min):
    search_results = [(encoder(row)[1], len(row)) for row in feature]
    return sum([mode(sr) / row_len for sr, row_len in search_results]) / len(
        feature
    )
