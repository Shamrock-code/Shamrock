# Determinism

Shamrock is **deterministic**, in the sense that the same inputs always give the same outputs
under the same conditions. More precisely, it has bitwise run-to-run reproducibility: running the
same binary, on the same hardware, with the same inputs, gives the same outputs, bit for bit. Two
runs in a row must produce identical dumps, logs of physical quantities, and checkpoints.

This is a project invariant, not only for the sake of it. It is what makes it possible to:

- debug a simulation by rerunning it and getting the same trajectory (and the same crash);
- keep your sanity while debugging because you can now isolate a faulty particle and track its
  change and that particle will stay the same;
- bisect a regression by comparing outputs of two commits on the same machine;
- restart from a checkpoint and check that nothing changed, e.g. detect hardware issues (like the
  issue we had on Aurora with register spilling induced data corruption which we only managed to
  isolate thanks to that property);
- tell a real change in the physics apart from run-to-run noise.

A change that makes the code faster but breaks this property is a regression, even if the result
is still "physically correct" or within a tolerance. Such a change must not be merged as is.

## What "same inputs" covers

The guarantee holds when all of the following are identical between the two runs:

- the binary (same commit, same compiler, same flags, same SYCL backend);
- the hardware (same devices, same device selection with `--sycl-cfg`);
- the run configuration (same number of MPI ranks, same command line and environment);
- the input files and Python script.

It is **not** a guarantee across different compilers, compiler flags (e.g. `-ffp-contract`,
`-march`), SYCL backends, or hardware. Floating-point results are allowed to differ there (see the
FMA note in {doc}`sph_kernel_optimisations`). Changes to the code itself (a different summation
order, a different kernel) are also allowed to change the last bits, as long as each version is
deterministic on its own.

## What breaks determinism

On GPUs and with MPI, most sources of non-determinism come from an order that depends on thread
scheduling, message arrival, or memory addresses. The common ones:

- **Floating-point atomic additions** (`atomic_ref<f32/f64>::fetch_add`/`fetch_sub`):
  floating-point addition rounds and is not associative, so the result depends on the order in
  which threads apply their updates. Use a reduction with a fixed tree (`shamalgs` primitives)
  instead. Atomic min/max are not affected: they are exact, so the final value does not depend on
  the order. The same goes for integer atomic add/min/max when only the final value is used.
- **Integer atomics used to choose a position** (e.g. `fetch_add` to append to an output buffer):
  the value set is the same, but its order depends on thread scheduling. This is fine only if the
  order is fixed afterwards (see below).
- **Reductions whose order depends on the work-group size, the number of threads or the
  scheduling**: e.g. a parallel reduction whose partition changes from run to run, or OpenMP
  `reduction` clauses on floats with dynamic scheduling.
- **Iterating over a container whose order is not fixed** and doing order-dependent work on it
  (floating-point sums, concatenation, assigning ids): hash maps keyed on pointers, or
  `DistributedData::for_each` when the order matters.
- **MPI messages processed in arrival order** (`MPI_ANY_SOURCE`, `MPI_Waitany`, ...) where the
  processing is order-dependent.
- **Unseeded randomness** (`std::random_device`, time-based seeds) or decisions based on the
  wall clock or measured timings (e.g. choosing an algorithm or a load-balancing split based on
  how long a kernel took).
- **Uninitialized memory** that ends up in the results.

## Patterns to keep it deterministic

- Prefer the `shamalgs` primitives (reductions, scans, sorts) over hand-rolled atomics: they are
  deterministic for a given device and configuration.
- If atomics are the fastest way to scatter data, restore a fixed order afterwards. Example:
  `src/shamtree/include/shamtree/details/reorder_scan_dtt_result.hpp` appends pairs with an
  atomic `fetch_add`, then calls `shamalgs::primitives::segmented_sort_in_place` on each slot so
  that the final order no longer depends on thread scheduling.
- Use `DistributedData::for_each_ordered` (increasing patch id) instead of `for_each` whenever the
  iteration order affects the result, e.g. concatenating per-patch data or summing floats.
- Seed every random number generator explicitly from the inputs (see
  `shamalgs::primitives::mock_vector`).
- Never feed a timing measurement back into the computation.

## Checking it

If a change uses one of the patterns listed above, or you are unsure whether it does, run the
same setup twice on the same machine and compare the outputs bitwise (e.g. the dumps or a hash of
the fields).
