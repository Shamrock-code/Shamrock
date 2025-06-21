Example batch script
There is an additional so that Turpan can find libomp correctly

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p shared
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00

export LD_LIBRARY_PATH=your_Shamrock_directory/build/.env/llvm-install/lib/aarch64-unknown-linux-gnu/:$LD_LIBRARY_PATH # Pour libomp
export ACPP_DEBUG_LEVEL=0

module purge
module load openmpi/gnu/4.1.6
module load boost/gnu/1.81.0

module list

mpirun ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ../exemples/sph_weak_scale_test.py
```
