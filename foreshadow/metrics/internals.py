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
def regex_rows(feature, cleaner):
    """Return percentage of rows matched by regex transformations.

    Cleaner(row) will return a CleanerReturn namedtupled, which will have the
    transformed text after all transformations (or the original text if it
    failed) and then a list of all the number of characters matched by each
    individual regex transform. Ergo, if any one of them is 0, then the
    transformations failed.

    Args:
        feature: a column of the dataset
        cleaner: callable that will perform all transformations to row of
            feature

    Returns:
        Return percentage of rows matched by regex transformations.

    """
    matched_lens = [
        cleaner(row[0]).match_lens for _, row in feature.iterrows()
    ]
    return sum([min(list_lens) for list_lens in matched_lens]) / len(feature)


@metric
def avg_col_regex(feature, cleaner, mode=min):
    """Return average percentage of each row's text transformed by cleaner.

    Cleaner will be a list of transformations, that will take the original
    text and transform it to new text. The percentage of the original text
    that is kept is averaged across all rows using 'mode'. For instance,
    if the original text was 'hello' and transformation 1 mapped 'll' to 'l'
    and transformation2 mapped 'o' to o, the amount of the column changed
    across all steps is: [2, 1], for that particular row. If mode is min,
    we take 1 row that row and the average text transformed for the row is
    20%. This is averaged across all rows.

    Args:
        feature: feature of dataset
        cleaner: callable that will perform all transformations to row of
            feature
        mode: callable operation to apply to list of characters matched for
            each transformation. Defines how we want to average for each row.

    Returns:
        Average amount of each column transformed, where 'mode' of all
        transformations is used to determine how much of the string was
        transformed.

    """
    matched_lens = [
        (cleaner(row[0]).match_lens, len(row)) for _, row in feature.iterrows()
    ]
    return sum(
        [mode(list_lens) / row_len for list_lens, row_len in matched_lens]
    ) / len(feature)
