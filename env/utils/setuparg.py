import os

import utils.envscript
from utils.oscmd import *


class SetupArg:
    """argument that will be passed to the machine setups"""

    def __init__(self, argv, builddir, shamrockdir, buildtype, lib_mode):
        self.argv = argv
        self.builddir = builddir
        self.shamrockdir = shamrockdir
        self.buildtype = buildtype
        self.lib_mode = lib_mode


class EnvGen:
    def __init__(self, machinefolder, builddir, env_setup_cmd):
        self.ENV_SCRIPT_HEADER = ""
        self.export_list = {}
        self.ext_script_list = []
        self.machinefolder = machinefolder
        self.builddir = builddir
        self.env_setup_cmd = env_setup_cmd

    def mod_setup_command(self):
        cmd = self.env_setup_cmd

        for idx, arg in enumerate(cmd):
            if arg == "--":
                wrapper_program_args = cmd[1:idx]
                underlying_tool_args = cmd[idx + 1 :]
                break
            else:
                wrapper_program_args = cmd[1:]
                underlying_tool_args = []

        builddir = wrapper_program_args.index("--builddir") + 1
        wrapper_program_args[builddir] = "$BUILD_DIR"

        cmd = ["$SHAMROCK_DIR/env/new-env"] + wrapper_program_args + ["--"] + underlying_tool_args
        return cmd

    def gen_env_file(self, source_file, destname="activate"):

        ENV_SCRIPT_HEADER = self.ENV_SCRIPT_HEADER

        for k in self.export_list.keys():
            ENV_SCRIPT_HEADER += "export " + k + "=" + self.export_list[k] + "\n"

        spacer = "\n####################################################################################################"

        for f in self.ext_script_list:
            ENV_SCRIPT_HEADER += f"{spacer}\n# Imported script " + f + f"{spacer}\n"
            ENV_SCRIPT_HEADER += utils.envscript.file_to_string(f)
            ENV_SCRIPT_HEADER += f"{spacer}{spacer}{spacer}\n"

        ENV_SCRIPT_HEADER += f"{spacer}\n# Env setup util" + f"{spacer}\n"
        ENV_SCRIPT_HEADER += "# Command used to setup: " + " ".join(self.env_setup_cmd)
        ENV_SCRIPT_HEADER += "\n\n"
        ENV_SCRIPT_HEADER += "function reset_env {\n"
        ENV_SCRIPT_HEADER += "    rm -rf $BUILD_DIR/.env\n"
        ENV_SCRIPT_HEADER += "    " + " ".join(self.mod_setup_command()) + "\n"
        ENV_SCRIPT_HEADER += '    echo "You can rerun source ./activate"\n'
        ENV_SCRIPT_HEADER += "}\n"
        ENV_SCRIPT_HEADER += f"{spacer}{spacer}{spacer}\n"

        ENV_SCRIPT_PATH = self.builddir + "/" + destname
        source_path = os.path.join(self.machinefolder, source_file)

        run_cmd(f"mkdir -p {os.path.dirname(ENV_SCRIPT_PATH)}")

        print("-- Generating env file " + ENV_SCRIPT_PATH)
        print("     -> From Base file " + source_path)

        utils.envscript.write_env_file(
            source_path=source_path, header=ENV_SCRIPT_HEADER, path_write=ENV_SCRIPT_PATH
        )

    def copy_file(self, source_path, dest_file):

        ENV_SCRIPT_PATH = self.builddir + "/" + dest_file

        run_cmd(f"mkdir -p {os.path.dirname(ENV_SCRIPT_PATH)}")

        print("-- Copying env file " + ENV_SCRIPT_PATH)
        print("     -> From Base file " + source_path)

        utils.envscript.copy_env_file(source_path=source_path, path_write=ENV_SCRIPT_PATH)

    def copy_env_file(self, source_file, dest_file):

        source_path = os.path.join(self.machinefolder, source_file)
        self.copy_file(source_path, dest_file)
