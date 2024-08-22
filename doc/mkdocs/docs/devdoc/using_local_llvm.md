# LLVM local install on cluster

Make a shallow clone of llvm :
```bash
git clone --depth 1 https://github.com/llvm/llvm-project.git -b release/17.x
cd llvm-project
```

```bash
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.1/llvm-project-17.0.1.src.tar.xz
tar -xvf llvm-project-17.0.1.src.tar.xz
cd llvm-project-17.0.1.src
```

configure it :
```bash
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS=all -DCMAKE_INSTALL_PREFIX=...instal loc.../llvm-17.x-local -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../llvm
```

compile it :
```bash
make -j install
```
