# Setup graph for SPH

SPH simulations come with an overwhelming diversity of possible setups. To tackle this, Shamrock provides generalized setups in the form of a graph, where links pass `PatchData` from one node to another.

## Node types

Nodes can be:

- **Generators**: create particle distributions (e.g. HCP lattice, cubic lattice, Monte Carlo disc, ...)
- **Modifiers**: transform the output of another node (e.g. offset, disc warp, filter, ...)
- **Combiners**: combine the results of two other nodes.

## Example

Imagine an even number generator, a modifier (multiplier) that multiplies the result by 10, and a combiner.

![Setup graph example](https://github.com/user-attachments/assets/194d81a1-94d9-4fed-a2ee-36fe65c96ed4)

Such a graph generates the list of even numbers multiplied by 10, twice.

## Usage

In a runscript the setup graph is manipulated through the `model.get_setup()` object. Nodes are created using the `make_*` factory functions, and the resulting graph is applied with `apply_setup`.

The available factories are currently:

- Generators:
    - `make_generator_lattice_hcp`
    - `make_generator_lattice_cubic`
    - `make_generator_disc_mc`
    - `make_generator_from_context`
- Modifiers:
    - `make_modifier_warp_disc`
    - `make_modifier_custom_warp`
    - `make_modifier_offset`
    - `make_modifier_filter`
    - `make_modifier_split_part`
- Combiners:
    - `make_combiner_add`

See the [Python API documentation](../../api.rst) for the complete and up-to-date list.

For example, the Sedov-Taylor setup for SPH is:

```python
setup = model.get_setup()
gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
setup.apply_setup(gen)
```

Or for the Sod tube:

```python
setup = model.get_setup()
gen1 = setup.make_generator_lattice_hcp(dr, (-xs,-ys/2,-zs/2),(0,ys/2,zs/2))
gen2 = setup.make_generator_lattice_hcp(dr*fact, (0,-ys/2,-zs/2),(xs,ys/2,zs/2))
comb = setup.make_combiner_add(gen1,gen2)
setup.apply_setup(comb)
```
