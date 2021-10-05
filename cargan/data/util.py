import contextlib
import os


@contextlib.contextmanager
def chdir(directory):
    curr_dir = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(curr_dir)
