# This file must be copied to the build directory in order for `pip install .` to work

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class ShamEnvExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class ShamEnvBuild(build_ext):

    def is_editable_mode(self) -> bool:
        # Detect editable mode
        editable_mode = False

        # Method 1: Check self.inplace (most reliable for build_ext)
        if hasattr(self, "inplace") and self.inplace:
            editable_mode = True

        # Method 2: Check for editable_mode attribute (newer setuptools)
        if hasattr(self, "editable_mode") and self.editable_mode:
            editable_mode = True
        return editable_mode

    

    def get_extdir(self, ext: ShamEnvExtension):

        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)  # type: ignore[no-untyped-call]
        extdir = ext_fullpath.parent.resolve()

        return extdir.parent.resolve()

    def build_extension(self, ext: ShamEnvExtension) -> None:

        if self.is_editable_mode():
            raise Exception(
                "Editable mode not supported for this config:\n"
                "  -> both the executable and the pylib are called shamrock"
                "  -> so there is a name conflict in editable mode since"
                "  -> the pylib will be copied to a shamrock folder which is the executable ..."
                )

        extdir = self.get_extdir(ext)

        print("-- Installing shamrock lib")

        print("-- mkdir output dir")
        print(f" -> mkdir -p {extdir}")
        subprocess.run(["bash", "-c", f"mkdir -p {extdir}"], check=True)

        install_steps = [
            "source ./activate",
            "shamconfigure",
            f"cmake . -DCMAKE_INSTALL_PREFIX={sys.prefix} -DCMAKE_INSTALL_PYTHONDIR={extdir} -DSHAMROCK_PATCH_LIB_RPATH=On",
            "shammake install",
        ]

        cmd = " && ".join(install_steps)
        print(f"-- Run install: {cmd}")
        subprocess.run(["bash", "-c", cmd], check=True)


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="shamrock",
    version="2025.10.0",
    author="Timothée David--Cléris",
    author_email="tim.shamrock@proton.me",
    description="SHAMROCK Code for astrophysics",
    long_description="",
    ext_modules=[ShamEnvExtension("shamrock.shamrock")],
    data_files=[("bin", ["shamrock"])],
    cmdclass={"build_ext": ShamEnvBuild},
    zip_safe=False,
    extras_require={},
    python_requires=">=3.7",
)
