import os

import utils.envscript
from utils.oscmd import *


class SetupArg:
    """argument that will be passed to the machine setups"""

    def __init__(self, argv, builddir, shamrockdir, buildtype, pylib, lib_mode):
        self.argv = argv
        self.builddir = builddir
        self.shamrockdir = shamrockdir
        self.buildtype = buildtype
        self.pylib = pylib
        self.lib_mode = lib_mode


class EnvGen:
    def __init__(self):
        self.ENV_SCRIPT_HEADER = ""
        self.export_list = {}
        self.ext_script_list = []

    def gen_env_file(self, source_path, builddir):

        for k in self.export_list.keys():
            self.ENV_SCRIPT_HEADER += "export " + k + "=" + self.export_list[k] + "\n"

        spacer = "\n####################################################################################################"

        for f in self.ext_script_list:
            self.ENV_SCRIPT_HEADER += f"{spacer}\n# Imported script " + f + f"{spacer}\n"
            self.ENV_SCRIPT_HEADER += utils.envscript.file_to_string(f)
            self.ENV_SCRIPT_HEADER += f"{spacer}{spacer}{spacer}\n"

        run_cmd("mkdir -p " + builddir)

        ENV_SCRIPT_PATH = builddir + "/activate"

        print("-- Generating env file " + ENV_SCRIPT_PATH)

        utils.envscript.write_env_file(
            source_path=source_path, header=self.ENV_SCRIPT_HEADER, path_write=ENV_SCRIPT_PATH
        )
