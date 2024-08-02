
# This file must be copied to the build directory in order for `pip install -e .` to work

import os
import re
import subprocess
import sys
from pathlib import Path
import time

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
    def build_extension(self, ext: ShamEnvExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)  # type: ignore[no-untyped-call]
        extdir = ext_fullpath.parent.resolve()

        cmake_lib_out = f"{extdir}{os.sep}"

        print(ext_fullpath,extdir,cmake_lib_out)

        subprocess.run(
            ["bash", "-c", "source ./activate && shamconfigure && shammake shamrock"], check=True
        )
        
        subprocess.run(
            ["bash", "-c", f"cp -v *.so {extdir}"], check=True
        )


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="shamrock",
    version="0.0.0.0.0",
    author="Timothée David--Cléris",
    author_email="timothee.david--cleris@ens-lyon.fr",
    description="SHAMROCK Code for astrophysics",
    long_description="",
    ext_modules=[ShamEnvExtension("shamrock")],
    cmdclass={"build_ext": ShamEnvBuild},
    zip_safe=False,
    extras_require={},
    python_requires=">=3.7",
)