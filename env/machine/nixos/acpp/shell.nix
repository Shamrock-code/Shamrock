{ pkgs ? (import <nixpkgs> { 
    config.allowUnfree = true;
    config.segger-jlink.acceptLicense = true; 
    config.cudaSupport = true;
}), ... }:

let
  llvm = pkgs.llvmPackages_18;
  gccForLibs = pkgs.stdenv.cc.cc;

  # Derivation to combine clang and clang-tools
  AdaptiveCpp = pkgs.stdenv.mkDerivation rec {

    pname = "acpp";
    version = "24.06.0";  # Adjust the version as needed

    # Fetch the tarball from GitHub releases
    src = fetchTarball {
      url = "https://github.com/AdaptiveCpp/AdaptiveCpp/archive/refs/tags/v24.06.0.tar.gz";
      sha256 = "sha256:1d7ld2azk45sv7124zkrkj1nfkmq0dani5zlalyn8v5s7q6vdxjc";
    };

    nativeBuildInputs = [
      pkgs.autoAddDriverRunpath
      pkgs.cudaPackages.cuda_nvcc
      pkgs.cmake
      pkgs.ninja
    ];

    buildInputs = [
      pkgs.libxml2
      pkgs.libffi
      pkgs.boost
      
      llvm.clang-tools
      llvm.clang
      llvm.llvm
      llvm.libclang

      pkgs.cudaPackages.cuda_cudart
      (pkgs.lib.getOutput "stubs" pkgs.cudaPackages.cuda_cudart)
    ];

    # this hardening option breaks rocm builds
    hardeningDisable = [ "zerocallusedregs" ];

    configurePhase = ''
      cmake -S . -GNinja -DCMAKE_INSTALL_PREFIX=$out -DCLANG_INCLUDE_PATH=${llvm.libclang.dev}/include
    '';

    buildPhase = "ninja";

    installPhase = "ninja install";

  };

in
pkgs.mkShell {


  nativeBuildInputs = [
    pkgs.autoAddDriverRunpath
    pkgs.cudaPackages.cuda_nvcc
  ];

  buildInputs = [
    AdaptiveCpp

    llvm.clang-tools
    llvm.clang
    llvm.llvm
    llvm.openmp
    llvm.libclang

    pkgs.boost
    pkgs.cmake
    pkgs.zsh

    pkgs.python312Packages.matplotlib
    pkgs.python312Packages.numpy
    pkgs.python312Packages.scipy
    pkgs.python312Packages.ipython

    pkgs.mpi

    pkgs.pre-commit

    pkgs.texliveFull
  
    pkgs.cudaPackages.cudatoolkit
    pkgs.cudaPackages.cuda_cudart
    (pkgs.lib.getOutput "stubs" pkgs.cudaPackages.cuda_cudart)
  ];

  # Set environment variables directly
  ACPP_INSTALL_DIR = "${AdaptiveCpp}";

  ACPP_DEBUG_LEVEL = "0";

  # disable all hardening flags
  NIX_HARDENING_ENABLE = "";

  shellHook = ''
    # Optional: Add custom message for debugging or confirmation
    echo "Entering Shamrock dev shell shell with LLVM 18 + AdaptiveCpp"
    echo "acpp install path : $ACPP_INSTALL_DIR"
  '';
}
