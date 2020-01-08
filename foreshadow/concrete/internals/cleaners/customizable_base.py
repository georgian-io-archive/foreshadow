"""Customizable BaseCleaner for cleaner transformers that user can extend."""

from abc import abstractmethod

from .base import BaseCleaner, CleanerReturn


class CustomizableBaseCleaner(BaseCleaner):
    """Base class for any user customizable cleaner transformer.

    This experimental class is to hide the match_len field from the user as I
    feel this is rather internal and tedious to explain if user ask about it.

    With this class, the user provided transformation function only needs to
    consider the transformation logic without wondering what that match length
    means.

    I want to refactor this metric calculation but don't want to do anything
    hasty at this moment just for the demo.
    """

    def __init__(self, transformation):
        """Construct a user supplied cleaner/flattener. # noqa S001

        Args:
            transformation: a callable that takes a string and returns # noqa DAR003
            transformed string.

        """
        super().__init__([transformation])

    @abstractmethod
    def metric_score(self, X):
        """Calculate the matching metric score of the cleaner on this col.

        In this method, you specify the condition on when to apply the
        cleaner and calculate a confidence score between 0 and 1 where 1
        means 100% certainty to apply the transformation.

        Args:
            X: a column as a dataframe.

        """
        pass

    def transform_row(self, row_of_feature, return_tuple=True):
        """Perform clean operations on text, that is a row of feature.

        Uses self.transformations determined at init time by the child class
        and performs the transformations sequentially.

        Args:
            row_of_feature: one row of one column
            return_tuple: return named_tuple object instead of just the row.
                This will often be set to False when passing this method to an
                external function (non source code) that will expect the
                output to only be the transformed row, such as DataFrame.apply.

        Returns:
            NamedTuple object with:
            .text
            the text in row_of_feature transformed by transformations. If
            not possible, it will be None.
            .match_lens
            the number of characters from original text at each step that
            was transformed.

        """
        matched_lengths = []  # this does not play nice with creating new
        # columns
        transformed_row = row_of_feature
        for transform in self.transformations:
            transformed_row = transform(row_of_feature)

        match_len = 0 if transformed_row == row_of_feature else 1
        if match_len == 0:
            matched_lengths.append(0)
            transformed_row = self.default(row_of_feature)
        matched_lengths.append(match_len)

        if return_tuple:
            return CleanerReturn(transformed_row, matched_lengths)
        else:
            return transformed_row
