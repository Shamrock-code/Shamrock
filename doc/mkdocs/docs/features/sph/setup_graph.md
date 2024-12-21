# Setup graph for SPH


After being way to annoyed with the overwhelming diversity of SPH setup a solution had to be found to tackle it !

This solution is the generalized setups in a form of a graph where link pass PatchData from a to node to another one.

## Node types

Node can be generators (e.g. lattice hcp, monte carlo, ...) modifiers (strechmapping, disc warp, ...) or combiners (Just combining the result of two other nodes.

## Exemple

Imagine a even number generator, a modifier (multiplier) that multiplies the result by 10, and a combiner.

![Screenshot_2024-08-29_22-18-29](https://github.com/user-attachments/assets/194d81a1-94d9-4fed-a2ee-36fe65c96ed4)

The following should generate the list of even numbers multiplied by 10, twice.

## Python wish

![Screenshot_2024-08-29_22-19-20](https://github.com/user-attachments/assets/3f68adc0-b8d5-4d5f-a632-c245ed1568f1)

We want something that look like this to generate the graph above

## When the wish come true

Well the answer is this PR.

It implements currently a way to generate a node and apply the resulting setup, but also:
- LatticeHCPGenerator

With the current PR the sedov taylor setup for SPH is now
```py
setup = model.get_setup()
gen = setup.make_generator_lattice_hcp(dr, bmin,bmax)
setup.apply_setup(gen)
```
Or for the sod tube:
```py
setup = model.get_setup()
gen1 = setup.make_generator_lattice_hcp(dr, (-xs,-ys/2,-zs/2),(0,ys/2,zs/2))
gen2 = setup.make_generator_lattice_hcp(dr*fact, (0,-ys/2,-zs/2),(xs,ys/2,zs/2))
comb = setup.make_combiner_add(gen1,gen2)
setup.apply_setup(comb)
```
