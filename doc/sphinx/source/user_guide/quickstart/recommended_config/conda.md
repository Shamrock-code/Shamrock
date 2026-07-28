# Conda Environment Configuration

## Recommended Setup

**From source**: Using AdaptiveCpp OpenMP backend with conda environment

```bash
# Clone the repo
git clone --recurse-submodules git@github.com:Shamrock-code/Shamrock.git
# cd into it
cd Shamrock

# Select the env to build from source
./env/new-env --machine conda.acpp --builddir build -- --backend omp

# Now move in the build directory
cd build
# Activate the workspace, which will define some utility functions
# Note that in conda mode this will create a conda env & install the correct packages
source ./activate
# Configure Shamrock
shamconfigure
# Build Shamrock
shammake
```
