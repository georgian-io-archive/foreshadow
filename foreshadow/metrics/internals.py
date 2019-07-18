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
def regex_rows(feature, transformations):
    import re

    def perform_regexes(row, transformations):
        row_regex = str(row)
        for transformation in transformations:
            search_result, row_regex = transformation(row_regex)
            if search_result is None:
                return False
        return True

    return sum(
        [1 for row in feature if perform_regexes(row, transformations)]
    ) / len(feature)


@metric
def avg_col_regex(feature, transformations, mode=min):
    def perform_regexes(row, search_repl):
        row_regex = str(row)
        search_results = []
        for transformation in search_repl:
            search_result, row_regex = transformation(row_regex)
            if search_result is None:
                return 0
            search_results.append(search_result.endpos - search_result.pos)
        mode_result = mode(search_results)
        return mode_result / len(str(row))

    return sum(
        [perform_regexes(row, transformations) for row in feature]
    ) / len(feature)
