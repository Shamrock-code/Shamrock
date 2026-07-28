# Linux (Debian/Ubuntu) Configuration

## Recommended Setup

**From source**: Using AdaptiveCpp OpenMP backend

```bash
# Clone the repo
git clone --recurse-submodules git@github.com:Shamrock-code/Shamrock.git
# cd into it
cd Shamrock

# Required packages
wget --progress=bar:force https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 18
sudo apt install -y libclang-18-dev clang-tools-18 libomp-18-dev
sudo apt install cmake libboost-all-dev python3-ipython

# Select the env to build from source
./env/new-env --builddir build --machine debian-generic.acpp -- --backend omp

# Now move in the build directory
cd build
# Activate the workspace, which will define some utility functions
source ./activate
# Configure Shamrock
shamconfigure
# Build Shamrock
shammake
```
