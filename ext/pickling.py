"""Convenient interface for loading and saving pickles.

My usual strategy for saving pickles is to have a main subfolder in which they
live, and potentially further subfolders within that.

The code here follows that convention. All functions take three arguments:
  pkl_dir: defines the main subfolder
  pkl_name: the name of the file (with no need to add a .pkl or any extension)
  sub_dirs: an ordered list of desired subdirectories within that main subfolder

The subdirs may be None, and is therefore the last argument.
"""
import pickle
import os


def exists(pkl_dir, pkl_name, sub_dirs=None):
    """Check if a pickle exists.

    Args:
      pkl_dir: String, the directory in which to save the pickle.
      pkl_name: String, the file_name for the pickle. No need to have .pkl at
        the end.
      sub_dirs: List of String subdirectories to append after the pkl_dir and
        before the name. Defaults to None.

    Returns:
      Boolean.
    """
    path = pkl_path(pkl_dir, pkl_name, sub_dirs)
    return os.path.exists(path)


def load(pkl_dir, pkl_name, sub_dirs=None):
    """Load a pickle.

    Args:
      pkl_dir: String, the directory in which to save the pickle.
      pkl_name: String, the file_name for the pickle. No need to have .pkl at
        the end.
      sub_dirs: List of String subdirectories to append after the pkl_dir and
        before the name. Defaults to None.

    Returns:
      Object.

    Raises:
      Exception if pickle not found.
    """
    path = pkl_path(pkl_dir, pkl_name, sub_dirs)
    try:
        with open(path, 'rb') as file:
            obj = pickle.load(file)
            return obj
    except FileNotFoundError:
        raise Exception('Pickle not found: %s' % path)


def pkl_path(pkl_dir, pkl_name, sub_dirs=None):
    """Get the file path for a pickle.

    Will add .pkl to the end of the pkl_name.

    Args:
      pkl_dir: String, the base directory.
      pkl_name: String, file name without .pkl at the end.
      sub_dirs: List of String subdirectories to append after the pkl_dir and
        before the name. Defaults to None.

    Returns:
      String.
    """
    if sub_dirs:
        return os.path.join(pkl_dir, '/'.join(sub_dirs), pkl_name + '.pkl')
    else:
        return os.path.join(pkl_dir, pkl_name + '.pkl')


def save(obj, pkl_dir, pkl_name, sub_dirs=None):
    """Save a pickle.

    Will create subfolders as necessary.

    Args:
      obj: Object, the object to pickle.
      pkl_dir: String, the directory in which to save the pickle.
      pkl_name: String, the file_name for the pickle. No need to have .pkl at
        the end.
      sub_dirs: List of String subdirectories to append after the pkl_dir and
        before the name. Defaults to None.
    """
    path = pkl_path(pkl_dir, pkl_name, sub_dirs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


class Pickler:
    """Wraps pickling functions and stores root directory."""

    def __init__(self, pkl_dir):
        """Create a new Pickler.

        Args:
          pkl_dir: String. The main subfolder for pickles.
        """
        self.pkl_dir = pkl_dir

    def exists(self, pkl_name, sub_dirs=None):
        """Check if a pickle exists.

        Args:
          pkl_name: String. No need to have .pkl at the end.
          sub_dirs: List of String subdirectories to append after the pkl_dir
            and before the name. Defaults to None.

        Returns:
          Bool.
        """
        return exists(self.pkl_dir, pkl_name, sub_dirs)

    def file_path(self, pkl_name, sub_dirs=None):
        """Get the file path to a pickle given by name.

        Args:
          pkl_name: String. No need to have .pkl at the end.
          sub_dirs: List of String subdirectories to append after the pkl_dir
            and before the name. Defaults to None.

        Returns:
          String.
        """
        return pkl_path(self.pkl_dir, pkl_name, sub_dirs)

    def load(self, pkl_name, sub_dirs=None):
        """Load a pickle.

        Args:
          pkl_name: String. No need to have .pkl at the end.
          sub_dirs: List of String subdirectories to append after the pkl_dir
            and before the name. Defaults to None.

        Returns:
          Object.

        Raises:
          Exception if pickle not found.
        """
        return load(self.pkl_dir, pkl_name, sub_dirs)

    def save(self, obj, pkl_name, sub_dirs=None):
        """Save a pickle.

        Args:
          obj: Object.
          pkl_name: String. No need to have .pkl at the end.
          sub_dirs: List of String subdirectories to append after the pkl_dir
            and before the name. Defaults to None.
        """
        save(obj, self.pkl_dir, pkl_name, sub_dirs)
