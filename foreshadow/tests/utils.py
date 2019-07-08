import os


def _get_test_folder():
    """Get path of the main test folder.

    Path is assumed to be somewhere above this file.

    Returns:
        path to test folder (root for tests)

    Raises:
        FileNotFoundError: if tests folder could not be found

    """
    path = os.path.dirname(__file__)
    while len(path) > 0:
        folder = os.path.basename(path)
        if folder == "tests":
            return path
        else:
            path = os.path.dirname(path)
    raise FileNotFoundError("Could not find tests directory in path")


def get_file_path(file_type, file_name):
    """Gets the path to a file inside of tests.

    Useful for paths to static files, such as data or configs.

    Args:
        file_type: identifies where the file is stored
        file_name: name of file.

    Returns:
        path to file

    """
    test_path = _get_test_folder()
    return os.path.join(test_path, file_type, file_name)
