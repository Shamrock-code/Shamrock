"""
CI test: every sibling group merges in a uniform RAMSES grid (new AMR)
=====================================================================

All the blocks of a uniform grid are below the density based merge criterion
(``cfg.set_amr_mode_density_based``), so one AMR pass of the new AMR
(``cfg.set_amr_mode_old(False)``) must merge every group of 8 siblings: a base
grid of ``base^3`` blocks becomes ``base^3 / 8`` blocks twice as large.

This is a regression test for a race in the kernel validating the derefinement
flags, which read the flags of the blocks ``id .. id+7`` while writing the
result for the block ``id`` into the same buffer. Depending on the execution
order of the work-items, whole sibling groups were refused (with several
OpenMP threads on the AdaptiveCpp OpenMP backend, most of them were).
"""

import shamrock

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# the new AMR is experimental
shamrock.enable_experimental_features()

# int coords size of a base block
sz = 2

# number of base blocks per side, chosen so that the sibling groups are not aligned on the
# chunks of the work-items split between threads
bases = [6, 10]
n_repeat = 3


def run_one_amr_pass(base):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    cfg.set_scale_factor(1.0 / (sz * base))
    cfg.set_eos_gamma(5.0 / 3.0)
    cfg.set_amr_mode_old(False)
    # the mass of a block is at most 1 in the unit box, so every block wants to merge
    cfg.set_amr_mode_density_based(crit_mass=10.0)
    model.set_solver_config(cfg)

    # a single patch holding all the blocks
    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base, base, base))

    model.set_field_value_lambda_f64("rho", lambda rmin, rmax: 1.0)
    model.set_field_value_lambda_f64("rhoetot", lambda rmin, rmax: 1.0)
    model.set_field_value_lambda_f64_3("rhovel", lambda rmin, rmax: (0.0, 0.0, 0.0))

    # the AMR pass is done at the end of a step with dt > 0
    model.evolve_once_override_time(0.0, 1e-6)

    data = ctx.collect_data()
    return data["cell_min"], data["cell_max"]


errors = []
for base in bases:
    expected_count = base**3 // 8
    for i in range(n_repeat):
        cell_min, cell_max = run_one_amr_pass(base)
        count = len(cell_min)
        refused = (count - expected_count) // 7
        print(f"base={base} run={i}: {count} blocks (expected {expected_count}), {refused} refused")

        if count != expected_count:
            errors.append(f"base={base} run={i}: {refused} sibling groups were not merged")
            continue

        for bmin, bmax in zip(cell_min, cell_max):
            if any(hi - lo != 2 * sz for lo, hi in zip(bmin, bmax)):
                errors.append(f"base={base} run={i}: unexpected block {bmin} {bmax}")
                break

if errors:
    raise RuntimeError("derefinement of a uniform grid failed:\n" + "\n".join(errors))
