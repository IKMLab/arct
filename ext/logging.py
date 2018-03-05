"""For logging training information to files."""
import os


def delete_log(file_path):
    """Delete a log file.

    Args:
      file_path: String, the full path to the log file.

    Raises:
      ValueError: if file not found.
    """
    if os.path.exists(file_path):
        print('Deleting log %s...' % file_path)
        os.remove(file_path)
    else:
        raise ValueError("File  %r doesn't exists - cannot delete." % file_path)


class Logger:
    """For logging information to file."""

    def __init__(self, file_path, print_too=True, override=False):
        """Create a new Logger.

        Args:
          file_path: String, the full path to the target file.
          print_too: Bool, whether or not to also print logger info to terminal.
          override: Bool, whether or not to delete any old files.
        """
        self.file_path = file_path
        self.print_too = print_too
        if override:
            if os.path.exists(file_path):
                print('Overriding - deleting previous log...')
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def log(self, info):
        with open(self.file_path, 'a') as file:
            file.write('\n' + info)
        if self.print_too:
            print(info)
