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
    def build_extension(self, ext: ShamEnvExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)  # type: ignore[no-untyped-call]
        extdir = ext_fullpath.parent.resolve()

        cmake_lib_out = f"{extdir}{os.sep}"

        print("-- Installing shamrock lib")
        print(f"### {ext_fullpath=}\n### {extdir=}\n### {cmake_lib_out=}")

        print("-- Modify builddir in local env")

        activate_build_dir = None
        with open(Path.cwd() / "activate", "r") as f:
            for line in f:
                if line.startswith("export BUILD_DIR="):
                    activate_build_dir = line.split("=")[1].strip()
                    break

        if activate_build_dir is None:
            raise Exception("BUILD_DIR not found in local env")

        cwd = os.getcwd()
        cwd_is_build = cwd == activate_build_dir

        print(f"### {cwd=}")
        print(f"### {activate_build_dir=}")
        print(f"### {cwd_is_build=}")

        print("-- Activating env")
        subprocess.run(
            [
                "bash",
                "-c",
                "source ./activate",
            ],
            check=True,
        )

        print("-- Configure")
        subprocess.run(
            [
                "bash",
                "-c",
                "source ./activate && shamconfigure",
            ],
            check=True,
        )

        print("-- Compile")
        subprocess.run(
            [
                "bash",
                "-c",
                "source ./activate && shammake shamrock shamrock_pylib",
            ],
            check=True,
        )

        print("-- mkdir output dir")
        print(f" -> mkdir -p {extdir}")
        subprocess.run(["bash", "-c", f"mkdir -p {extdir}"], check=True)

        print("-- Copy lib&exe to output dir")
        subprocess.run(["bash", "-c", f"ls {activate_build_dir}"], check=True)

        if not cwd_is_build:
            subprocess.run(
                ["bash", "-c", f" cp -v {activate_build_dir}/*.so {activate_build_dir}/shamrock ."],
                check=True,
            )

        subprocess.run(["bash", "-c", f"ls {activate_build_dir}"], check=True)
        subprocess.run(["bash", "-c", f"cp -v {activate_build_dir}/*.so {extdir}"], check=True)


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="shamrock",
    version="2025.05.0",
    author="Timothée David--Cléris",
    author_email="tim.shamrock@proton.me",
    description="SHAMROCK Code for astrophysics",
    long_description="",
    ext_modules=[ShamEnvExtension("shamrock")],
    data_files=[("bin", ["shamrock"])],
    cmdclass={"build_ext": ShamEnvBuild},
    zip_safe=False,
    extras_require={},
    python_requires=">=3.7",
)
