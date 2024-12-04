{ pkgs ? import <nixpkgs> {} }:

let
  llvm18 = pkgs.llvmPackages_18;
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

    buildInputs = [
      llvm18.clang-tools
      llvm18.clang
      llvm18.llvm
      llvm18.libclang
      pkgs.boost186
      pkgs.cmake
      pkgs.ninja
    ];

    configurePhase = ''
      ls
      cmake -S . -GNinja -DCMAKE_INSTALL_PREFIX=$out -DCLANG_INCLUDE_PATH=${llvm18.libclang.dev}/include
    '';

    buildPhase = "ninja";

    installPhase = "ninja install";

  };

in
pkgs.mkShell {

  buildInputs = [
    AdaptiveCpp

    llvm18.clang-tools
    llvm18.clang
    llvm18.llvm
    llvm18.openmp
    llvm18.libclang

    pkgs.boost186
    pkgs.cmake
    pkgs.zsh

    pkgs.mpi
  ];

  # Set environment variables directly
  ACPP_INSTALL_DIR = "${AdaptiveCpp}";

  shellHook = ''
    # Optional: Add custom message for debugging or confirmation
    echo "Entering zsh shell with LLVM 18..."
    echo $ACPP_INSTALL_DIR
  '';
}
