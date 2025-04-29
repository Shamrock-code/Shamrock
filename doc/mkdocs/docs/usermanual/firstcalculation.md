# Running your first calculation
First you need to set up a SYCL configuration:

```bash
./shamrock --sycl-cfg 0:0 --smi
mpirun -n 4 ./shamrock --sycl-cfg 0:0 --smi
```
## Using an ipython console within the terminal

First make sure you have ipython installed:
```bash
sudo apt install python3-ipython
```
Then open the console

```bash
./shamrock --sycl-cfg 0:0 --smi --ipython
```
## run a pre-written script
Setting and running a simulation can all be done through a python script. You don't need to go through the arcanes of SHAMROCK! First, activate the Shamrock virtual environment:

```python
source Shamrock-venv/bin/activate
```

SHAMROCK provides an array of pre-cooked scripts you can run as is. When running a simulation of your own, you can just follow the same steps as for the pre-cooked scripts.
For the sake of clarity, let's take the example of the spherical_wave.py script. To run it, type this in your terminal:

```bash
./shamrock --sycl-cfg 0:0 --smi --loglevel 10 --rscript ../exemples/spherical_wave.py
```
the --loglevel argument specifies the degree of verbosity of SHAMROCK.
- 0 silent
- 1 tells basic steps
- 10 good for basic use
- 100 specifies instructions in the kernel

If you want to use modules that need to me installed (eg sarracen), you need to sudo pip install them.

Shamrock files can be outputed in two formats: the vtk/native format (suitable for number of SPH particles > 10⁸), of the fortran format (phantom dump format, which allows for the use of various utilities developed for Phantom, see below).

## importing a setup from phantom

It is possible to create a set-up with Phantom and use it in Shamrock. Doing this can proof useful if you are already familiar with Phantom set-ups, or if you want to evolve a silumation started in Phantom on large timescales. To do so:
```python
import shamrock

ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_SPH(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

dump = shamrock.load_phantom_dump("reference-files/blast_00010")

cfg = model.gen_config_from_phantom_dump(dump)
cfg.set_boundary_periodic()
cfg.print_status()

model.set_solver_config(cfg)
model.init_scheduler(int(1e5),1)

model.init_from_phantom_dump(dump)
```

## Analysing dumps via the phantomanalysis tool

After running your simulation in Shamrock and outputing it in fortran format, you can use the Phantom utility phantomanalysis. For it to work, you NEED to copy in the directory where your shamrock dumps are the .in file and the .params file. Careful with underscores.
