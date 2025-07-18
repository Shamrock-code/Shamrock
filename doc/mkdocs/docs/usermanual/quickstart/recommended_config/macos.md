# macOS Configuration

## Recommended Setup

**From source**: Using AdaptiveCpp OpenMP backend

```bash
# Clone the repo
git clone --recurse-submodules git@github.com:Shamrock-code/Shamrock.git
# cd into it
cd Shamrock

# Required packages
brew install cmake libomp boost open-mpi adaptivecpp

# Select the env to build from source
./env/new-env --builddir build --machine macos-generic.acpp -- --backend omp

# Now move in the build directory
cd build
# Activate the workspace, which will define some utility functions
source ./activate
# Configure Shamrock
shamconfigure
# Build Shamrock
shammake
```
