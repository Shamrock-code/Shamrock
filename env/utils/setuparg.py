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
    def __init__(self, machinefolder, builddir):
        self.ENV_SCRIPT_HEADER = ""
        self.export_list = {}
        self.ext_script_list = []
        self.machinefolder = machinefolder
        self.builddir = builddir

    def gen_env_file(self, source_file):

        for k in self.export_list.keys():
            self.ENV_SCRIPT_HEADER += "export " + k + "=" + self.export_list[k] + "\n"

        spacer = "\n####################################################################################################"

        for f in self.ext_script_list:
            self.ENV_SCRIPT_HEADER += f"{spacer}\n# Imported script " + f + f"{spacer}\n"
            self.ENV_SCRIPT_HEADER += utils.envscript.file_to_string(f)
            self.ENV_SCRIPT_HEADER += f"{spacer}{spacer}{spacer}\n"

        run_cmd("mkdir -p " + self.builddir)

        ENV_SCRIPT_PATH = self.builddir + "/activate"
        source_path = os.path.join(self.machinefolder, source_file)

        print("-- Generating env file " + ENV_SCRIPT_PATH)
        print("     -> From Base file " + source_path)

        utils.envscript.write_env_file(
            source_path=source_path, header=self.ENV_SCRIPT_HEADER, path_write=ENV_SCRIPT_PATH
        )

    def copy_env_file(self, source_file, dest_file):

        source_path = os.path.join(self.machinefolder, source_file)
        ENV_SCRIPT_PATH = self.builddir + "/" + dest_file

        print("-- Copying env file " + ENV_SCRIPT_PATH)
        print("     -> From Base file " + source_path)

        utils.envscript.copy_env_file(source_path=source_path, path_write=ENV_SCRIPT_PATH)
