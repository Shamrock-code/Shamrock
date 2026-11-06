# Particle injection

Particles can be injected during a run, for instance to feed the fallback stream of a tidal
disruption event (TDE) into a simulation of the black hole surroundings. Injection is available in
both the SPH and the GSPH solvers, with the same config methods.

## Fallback stream

The stream follows the fallback rate

$$
\dot M(t) = \dot M_{\rm peak} \left(\frac{t - t_{\rm disrupt}}{t_{\rm mb}}\right)^{n}
\quad \text{for } t_{\rm disrupt} + t_{\rm mb} \le t < t_{\rm end},
$$

and zero otherwise, with $n = -5/3$ for a flat distribution of specific energy of the debris
($n = 0$ gives a constant rate).

```python
shamrock.enable_experimental_features()

cfg = model.gen_default_config()
# ...
cfg.add_injection_fallback_stream(
    inject_pos=(-2.0, 0.2, 0.0),  # centre of the injection disc
    inject_vel=(1.0, 0.0, 0.0),  # velocity of the injected gas
    stream_radius=0.02,  # radius of the stream cross-section
    u_inject=1e-4,  # specific internal energy of the injected gas
    mdot_peak=1e-3,  # rate at t_disrupt + t_mb
    t_mb=1.0,  # return time of the most bound debris
    t_disrupt=-1.0,  # optional, default 0
    mdot_index=-5.0 / 3.0,  # optional, default -5/3
)
model.set_solver_config(cfg)
```

The injected particles have the particle mass of the simulation. Several streams can be added.

## How it works

The gas crosses the disc of radius `stream_radius` centred on `inject_pos` and orthogonal to
`inject_vel` as walls of particles: a square lattice cropped to the disc. A new wall is born each
time the rate has supplied its mass, so the mean injected rate follows $\dot M(t)$ whatever the
timestep. The wall birth times do not depend on the timestep either: a wall born during a step is
placed where it would be at the end of the step. The lattice spacing follows the density of the
stream $\rho = \dot M / (\pi R^2 v)$, so that the particle distribution stays isotropic as the rate
decreases: the walls get fewer particles, and stay spaced like the particles within a wall.

The next walls to be born are kept upstream of the injection point as boundary particles. They are
removed and re-created at their analytic position every step, so that the youngest walls of the
stream see a correct pressure instead of a free surface (the "conveyor belt" of Phantom's injection
modules). By default there are enough of them to cover the kernel support of the youngest wall.

The injected particles are tagged in the `part_id` field, which is added to the layout when an
injection is configured:

- bit 62 is set for the injected particles,
- bit 61 is set for the boundary particles, which should be filtered out of the analysis:

```python
part_id = np.asarray(data["part_id"], dtype=np.uint64)
is_boundary = ((part_id >> np.uint64(61)) & np.uint64(1)) == 1
```

The other particles keep the ids of the setup (0 unless particle tracking is enabled).

## Parallelism, determinism and restarts

The walls are a deterministic function of time: every MPI rank generates the same particles and
only inserts the ones falling in its own patches, so injecting needs no communication. The injected
particles do not depend on the number of ranks, and two identical runs are bitwise identical.

Nothing is stored besides the config: after a restart from a dump, the sequence of walls is
replayed from the start of the fallback, and the walls born before the time of the dump are assumed
to be in the dump. This also means that walls born before the time at which a simulation starts are
not injected.

## Caveats

- The injection region must be inside the simulation box.
- The region around the injection point should be free of other gas. The boundary walls are
  re-created there every step whatever is already there, and a particle buried in them sees an
  unresolved dense clump, where the default smoothing length iteration can diverge. In SPH, the
  bisection safeguarded iteration (`cfg.set_smoothing_length_density_based_bisecting_nr()`) copes
  with it.
- The timestep is limited so that a step does not jump over the start of an injection. After that
  only the usual CFL conditions apply, even if many walls are born in one step.
