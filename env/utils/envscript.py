import os
import shutil


def write_env_file(source_path, header, path_write):
    ENV_SCRIPT_CONTENT = header + "\n"

    # Get current file path
    cur_file = os.path.realpath(os.path.expanduser(__file__))

    with open(source_path) as f:
        contents = f.read()
        ENV_SCRIPT_CONTENT += contents

    with open(path_write, "w") as env_script:
        env_script.write(ENV_SCRIPT_CONTENT)


def copy_env_file(source_path, path_write):
    shutil.copy2(source_path, path_write)


def file_to_string(path):
    with open(path) as f:
        contents = f.read()
        return contents
