import os


def get_filename_from_path(path):
    return strip_extension(os.path.basename(path))


def strip_extension(file_name):
    return os.path.splitext(file_name)[0]


def get_files_in_directory(directory):
    dir = os.fsencode(directory)
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        # TODO check that file is an actual image '.jpg' or '.png'
        yield directory + filename
