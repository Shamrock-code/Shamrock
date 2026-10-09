# Implementation selection

Several performance-critical algorithms in Shamrock ship with more than one concrete
implementation (e.g. a portable fallback vs. a backend-specific accelerated kernel, or several
kernel strategies tuned for different problem sizes). Implementation selection is the mechanism
that lets you list the implementations available for such an algorithm, query the one currently
active, and switch it at runtime, from Python, without recompiling.

This page documents both sides of it: how to use it as a user (e.g. for benchmarking, or to work
around a bad implementation on a given platform), and how to wire a new algorithm into it as a
developer.

## Current status

Implementation selection is built on `shamalgs::ImplVariantGlobal`
(`shamalgs/include/shamalgs/ImplVariant.hpp`), a `std::variant`-based selector. This is the
pattern to use for any algorithm that needs implementation selection, whether new or being added
to an existing one.

## User side (Python)

Every algorithm that supports implementation selection is registered under a name (e.g.
`"reduction"`, `"scan_exclusive_sum_in_place"`, `"clbvh_dual_tree_traversal"`). The following
six functions are exposed under `shamrock.algs`, and all except `get_registered_algs` take that
name as their first argument `alg`:

- `get_registered_algs()`: the sorted list of registered algorithm names.
- `get_default_impl_list(alg)`: the list of available implementations.
- `get_current_impl(alg)`: the implementation currently selected.
- `set_impl(alg, impl)`: select an implementation.
- `is_impl_set(alg)`: whether an implementation has been selected yet.
- `autoselect_impl(alg)`: select the algorithm's default implementation.

An unknown `alg` raises an exception whose message lists the registered names. The same
functions cover every algorithm, including `shamtree`'s dual tree traversal.

```python
import shamrock

print(shamrock.algs.get_registered_algs())
# ['clbvh_dual_tree_traversal', 'compute_histogram', 'is_all_true', 'reduction', ...]
```

Implementations are plain JSON config strings of the form
`{"implementation": "<name>", "parameters": {...}}`. `set_impl` takes that whole string back.

`is_impl_set` and `autoselect_impl` matter because an algorithm starts with no implementation
selected: it only picks its default the first time it actually runs, so `get_current_impl(alg)`
returns `"null"` until then, unless you call `autoselect_impl(alg)` yourself first. From Python,
the default is picked for the compute device (`shamsys::instance::get_compute_scheduler_ptr()`),
so `autoselect_impl` requires the devices to be initialized (`shamrock.sys.init(...)` in library
mode). The other functions also work before that.

```python
import shamrock

current = shamrock.algs.get_current_impl("scan_exclusive_sum_in_place")
print(current)
# null (nothing selected yet, and the algorithm hasn't run)

# two ways of selecting an implementation manually:

# 1. pick a specific one
shamrock.algs.set_impl(
    "scan_exclusive_sum_in_place", '{"implementation":"std_scan","parameters":{}}'
)

# 2. or fall back to the algorithm's own default
if not shamrock.algs.is_impl_set("scan_exclusive_sum_in_place"):
    shamrock.algs.autoselect_impl("scan_exclusive_sum_in_place")
```

If you want to test something against every available implementation, do:

```python
import json
import shamrock

for impl in shamrock.algs.get_default_impl_list("scan_exclusive_sum_in_place"):
    shamrock.algs.set_impl("scan_exclusive_sum_in_place", impl)
    name = json.loads(impl)["implementation"]
    print(f"running with {name}")
    # ...
```

### Where this is used in practice

The benchmark scripts under `examples/benchmarks/` sweep over every available implementation of
an algorithm this way to compare their performance: `run_segmented_sort_in_place_performance.py`,
`run_exclusive_scan_in_place.py`.

The C++ unit tests for these algorithms follow the same loop, to run every implementation against
the same reference data (see e.g. `tests/shamalgs/primitives/scan_exclusive_sum_in_placeTests.cpp`).

Selection is process-wide and does not persist: it resets to the algorithm's default every time
the process restarts.

## Developer side (C++)

### Adding implementation selection to a new algorithm

Use `shamalgs::ImplVariantGlobal`, documented in detail in
`shamalgs/include/shamalgs/ImplVariant.hpp`. Each implementation is a small tag struct exposing a
`variant_type_name`; the selector is a `std::variant` of those, and dispatch is a plain
`std::visit`. Skeleton, following `scan_exclusive_sum_in_place.cpp` as a reference:

```cpp
#include "shambase/overloaded.hpp"
#include "shamalgs/ImplVariant.hpp"
#include "shamalgs/impl_registry.hpp"
#include "shambackends/DeviceScheduler.hpp"

namespace shamalgs::primitives {

    namespace impl {

        /// One-line doc per alternative: what it does / when it's a good fit
        struct AltA {
            static constexpr std::string_view variant_type_name = "alt_a";
        };
        struct AltB {
            static constexpr std::string_view variant_type_name = "alt_b";
        };

        /// Registry name, shared by the registration and the dispatch site(s)
        constexpr std::string_view my_algo_impl_name = "my_algo";

        /// The lambda picks the default implementation, it may inspect the device behind
        /// the scheduler or ignore it
        shamalgs::ImplVariantGlobal<AltA, AltB> my_algo_impl{
            [](const sham::DeviceScheduler_ptr &, auto &self) {
                self.set(AltA{});
            }};

        // Must come after the global it registers: same TU, so it is initialized after it
        SHAMALGS_REGISTER_IMPL(my_algo_impl_name, my_algo_impl);

    } // namespace impl

    void my_algo(const sham::DeviceScheduler_ptr &dev_sched, ...) {
        // Lazy default on first use, if no implementation was selected yet
        if (!impl::my_algo_impl.is_set()) {
            shamalgs::impl_registry::autoselect_impl(impl::my_algo_impl_name, dev_sched);
        }

        std::visit(
            shambase::overloaded{
                [&](impl::AltA) { /* ... */ },
                [&](impl::AltB) { /* ... */ },
            },
            impl::my_algo_impl.get());
    }

} // namespace shamalgs::primitives
```

`ImplVariantGlobal` starts unset — `is_set()` is `false` until something selects an
implementation — but the rule picking its default is given at construction, as a callable of
signature `void(const sham::DeviceScheduler_ptr &, ImplVariantGlobal &)` that calls `set()` on the
selector it is handed. `autoselect(dev_sched)` runs it. Both `is_set()` and `autoselect()` are part
of the non-template `shamalgs::IImplVariant` interface, so code holding a selector type-erased can
also check and fill it in. The lazy-default pattern above (check `is_set()`, autoselect right
before dispatching) is what every algorithm currently does.

`SHAMALGS_REGISTER_IMPL` (`shamalgs/include/shamalgs/impl_registry.hpp`) registers the selector
in `shamalgs::impl_registry` under its name, at static initialization. Put it at namespace scope
in the `.cpp` file, right after the selector's definition: objects of one translation unit are
initialized in definition order, so the selector is already constructed when it registers.
Registering the same name twice throws. The selector must be a non-`inline` global defined in a
single `.cpp` file (declare it `extern` in the header if a header-only dispatch needs it, as
`compute_histogram.hpp` does), and `ImplVariantGlobal` is neither copyable nor movable since the
registry stores its address. The registry then reads and changes the selection of every
algorithm by name, purely through `IImplVariant`; it is also what the Python bindings and the
unit tests use, so an algorithm needs no selection function of its own.

Dispatch sites keep the direct `is_set()` / `get()` access on the typed global, which
`std::visit` needs, but autoselect through `shamalgs::impl_registry::autoselect_impl` rather than
calling `my_algo_impl.autoselect(...)` directly, so that every default selection goes through
the registry (which also logs it). The registry lookup only happens on first use.

`autoselect_impl` always takes the `sham::DeviceScheduler_ptr` the algorithm runs on, and
forwards it to the selector's `autoselect` (it throws if the scheduler is null). Most lambdas
ignore it, because the default only depends on compile-time information (a `#ifdef`
backend/platform check, e.g. `scan_exclusive_sum_in_place`'s). When the default depends on the
device, the lambda looks at it. `compute_histogram` does this: a GPU device picks a different
default than a CPU one (`compute_histogram.cpp`).

```cpp
ComputeHistogramImpl compute_histogram_impl{
    [](const sham::DeviceScheduler_ptr &dev_sched, auto &self) {
        if (dev_sched->ctx->device->prop.type == sham::DeviceType::GPU) {
            self.set(GpuOversubscribe{});
        } else {
            self.set(NaiveGpu{});
        }
    }};

SHAMALGS_REGISTER_IMPL(compute_histogram_impl_name, compute_histogram_impl);
```

The dispatching function passes its own scheduler (or `buf.get_dev_scheduler_ptr()` when it only
gets buffers). The Python binding `shamrock.algs.autoselect_impl(alg)` and the unit tests supply
the compute scheduler explicitly:

```cpp
shamalgs::impl_registry::autoselect_impl(
    "compute_histogram", shamsys::instance::get_compute_scheduler_ptr());
```

An alternative with tunable fields specializes `shamalgs::ImplVariantParams<Alt>` to control how
those fields serialize to/from the `"parameters"` JSON — see the doc comment at the top of
`ImplVariant.hpp` for a worked example (a `group_size` field), or
`shamalgs::primitives::impl::AtomicEarlyExit` in `is_all_true.cpp` for a real one.

### Exposing more than one default per alternative

By default, `get_default_impl_list(alg)` lists exactly one instance per alternative type
(the default-constructed one). An alternative with tunable fields can opt into listing several
of its own instances instead — e.g. the same kernel at a few different group sizes — by adding a
static `variant_custom_defaults()` method returning a `std::vector<Alt>`:

```cpp
struct AtomicEarlyExit {
    static constexpr std::string_view variant_type_name = "atomic_early_exit";
    u32 group_size = 256;

    static std::vector<AtomicEarlyExit> variant_custom_defaults() {
        return {AtomicEarlyExit{128}, AtomicEarlyExit{256}};
    }
};
```

`ImplVariantGlobal`/`get_default_config_list()` detects this automatically (via the
`HasCustomDefaults` concept in `ImplVariant.hpp`) and lists one config string per returned
instance instead of just one; alternatives that don't define `variant_custom_defaults()` are
unaffected. Because all of them still share the same `variant_type_name`, only `"parameters"`
tells them apart in the resulting config strings, so any code keying results off
`json.loads(impl)["implementation"]` alone (see the benchmark script note below) needs to fold
`"parameters"` into the key too, or entries collide.

Once the selector, its registration and the dispatch are in place, wire it up end to end:

1. Header: nothing to declare for implementation selection. The selector, its name constant and
   its registration all live in the `.cpp` file (only a header-only dispatch, like
   `compute_histogram`'s, needs the selector declared `extern` and the name constant in the
   header).
2. Python bindings: nothing to add. `shamrock.algs.get_registered_algs()` lists the new name, and
   the name-keyed functions of `shamrock.algs` already cover it.
3. Unit test: autoselect if `shamalgs::impl_registry::is_impl_set("<algo>")` is `false`, save
   `get_current_impl("<algo>")`, loop over `get_default_impl_list("<algo>")` calling
   `set_impl("<algo>", impl)` before each run, then restore the saved implementation.
4. Benchmark script (`examples/benchmarks/`, if one exists for the algorithm): same loop through
   `shamrock.algs`, extracting the implementation's display name with
   `json.loads(impl)["implementation"]`; call `shamrock.algs.autoselect_impl("<algo>")` first
   when `shamrock.algs.is_impl_set("<algo>")` is `False`.

## Related files

- `shamalgs/include/shamalgs/ImplVariant.hpp` — authoritative reference for
  `ImplVariantGlobal`'s API.
- `shamalgs/include/shamalgs/impl_registry.hpp` — the name-keyed registry and
  `SHAMALGS_REGISTER_IMPL`.
