{ pkgs ? (import <nixpkgs> {
    config.allowUnfree = true;
    config.cudaSupport = true;
}), ... }:

let
  llvm = pkgs.llvmPackages_18;
  gccForLibs = pkgs.stdenv.cc.cc;

in
pkgs.mkShell {

  nativeBuildInputs = [
      pkgs.autoAddDriverRunpath
      pkgs.cudaPackages.cuda_nvcc
      pkgs.cudaPackages.cuda_cudart
      (pkgs.lib.getOutput "stubs" pkgs.cudaPackages.cuda_cudart)
  ];

  buildInputs = [

    llvm.clang-tools
    llvm.clang
    llvm.llvm
    llvm.openmp
    llvm.libclang
    pkgs.lldb_18

    pkgs.boost
    pkgs.cmake
    pkgs.zsh

    pkgs.python312Packages.matplotlib
    pkgs.python312Packages.numpy
    pkgs.python312Packages.scipy
    pkgs.python312Packages.ipython
    #pkgs.python312Packages.pybind11
    #pkgs.adaptivecpp

    #pkgs.pocl

    pkgs.mpi

    pkgs.doxygen
    pkgs.graphviz

    pkgs.pre-commit

    pkgs.texliveFull

    pkgs.cudaPackages.cudatoolkit

    pkgs.cudaPackages.nsight_systems.bin
    #pkgs.cudaPackages.nsight_compute
  ];

  CMAKE_CLANG_INCLUDE_PATH = "${llvm.libclang.dev}/include";

  # Set environment variables directly
  #ACPP_INSTALL_DIR = "${AdaptiveCpp}";

  ACPP_DEBUG_LEVEL = "0";

  # disable all hardening flags
  NIX_HARDENING_ENABLE = "";

  shellHook = ''
    # Optional: Add custom message for debugging or confirmation
    echo "Entering Shamrock dev shell shell with LLVM 18 + AdaptiveCpp"
    echo "acpp install path : $ACPP_INSTALL_DIR"
  '';
}
