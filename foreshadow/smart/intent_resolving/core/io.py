"""Module containing IO functionalities."""
import dill


def to_pickle(items, path: str):
    """Serialize components to a pickle file."""
    with open(path, "wb") as file:
        dill.dump(items, file)


def from_pickle(path: str):
    """Deserialize components to a pickle file."""
    with open(path, "rb") as file:
        return dill.load(file)
