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

For every algorithm that supports it, three functions are exposed on the Python bindings, under
`shamrock.algs` for `shamalgs` primitives and `shamrock.tree` for `shamtree`'s dual tree
traversal:

- `get_default_impl_list_<algo>()` — the list of available implementations.
- `get_current_impl_<algo>()` — the implementation currently selected.
- `set_impl_<algo>(...)` — select an implementation.
- `is_impl_set_<algo>()` — whether an implementation has been selected yet.
- `autoselect_impl_<algo>()` — select the algorithm's default implementation.

Implementations are plain JSON config strings of the form
`{"implementation": "<name>", "parameters": {...}}`. `set_impl_<algo>` takes that whole string
back.

`is_impl_set_<algo>` / `autoselect_impl_<algo>` matter because an `ImplVariantGlobal` starts
unset: algorithms only pick their default the first time they actually run, so
`get_current_impl_<algo>()` returns `"null"` until then, unless you call
`autoselect_impl_<algo>()` yourself first. From Python, the default is picked for the compute
device (`shamsys::instance::get_compute_scheduler_ptr()`).

```python
import shamrock

current = shamrock.algs.get_current_impl_scan_exclusive_sum_in_place()
print(current)
# null (nothing selected yet, and the algorithm hasn't run)

# two ways of selecting an implementation manually:

# 1. pick a specific one
shamrock.algs.set_impl_scan_exclusive_sum_in_place('{"implementation":"std_scan","parameters":{}}')

# 2. or fall back to the algorithm's own default
if not shamrock.algs.is_impl_set_scan_exclusive_sum_in_place():
    shamrock.algs.autoselect_impl_scan_exclusive_sum_in_place()
```

If you want to test something against every available implementation, do:

```python
import json
import shamrock

for impl in shamrock.algs.get_default_impl_list_scan_exclusive_sum_in_place():
    shamrock.algs.set_impl_scan_exclusive_sum_in_place(impl)
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
#include "shamalgs/ImplVariant.hpp"
#include "shambase/overloaded.hpp"
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

        /// The lambda picks the default implementation, it may inspect the device behind
        /// the scheduler or ignore it
        shamalgs::ImplVariantGlobal<AltA, AltB> my_algo_impl{
            [](const sham::DeviceScheduler_ptr &, auto &self) {
                self.set(AltA{});
            }};

        std::vector<std::string> get_default_impl_list_my_algo() {
            return my_algo_impl.get_default_config_list();
        }

        std::string get_current_impl_my_algo() { return my_algo_impl.get_current_config(); }

        bool is_impl_set_my_algo() { return my_algo_impl.is_set(); }

        void set_impl_my_algo(const std::string &impl) { my_algo_impl.set(impl); }

        /// Called lazily on first use if no implementation was selected yet
        void autoselect_impl_my_algo(const sham::DeviceScheduler_ptr &dev_sched) {
            my_algo_impl.autoselect(dev_sched);
        }

    } // namespace impl

    void my_algo(const sham::DeviceScheduler_ptr &dev_sched, ...) {
        if (!impl::my_algo_impl.is_set()) {
            impl::autoselect_impl_my_algo(dev_sched);
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

`autoselect_impl_<algo>` always takes the `sham::DeviceScheduler_ptr` the algorithm runs on, and
forwards it to the selector's `autoselect`. Most lambdas ignore it, because the default only
depends on compile-time information (a `#ifdef` backend/platform check, e.g.
`scan_exclusive_sum_in_place`'s). When the default depends on the device, the lambda looks at it.
`compute_histogram` does this: a GPU device picks a different default than a CPU one.

```cpp
inline shamalgs::ImplVariantGlobal<Reference, NaiveGpu, GpuTeamFetching, GpuOversubscribe>
    compute_histogram_impl{[](const sham::DeviceScheduler_ptr &dev_sched, auto &self) {
        if (dev_sched->ctx->device->prop.type == sham::DeviceType::GPU) {
            self.set(GpuOversubscribe{});
        } else {
            self.set(NaiveGpu{});
        }
    }};
```

The dispatching function passes its own scheduler (or `buf.get_dev_scheduler_ptr()` when it only
gets buffers). The Python binding and unit tests supply the compute scheduler explicitly:

```cpp
shamalgs_module.def("autoselect_impl_compute_histogram", []() {
    shamalgs::primitives::impl::autoselect_impl_compute_histogram(
        shamsys::instance::get_compute_scheduler_ptr());
});
```

An alternative with tunable fields specializes `shamalgs::ImplVariantParams<Alt>` to control how
those fields serialize to/from the `"parameters"` JSON — see the doc comment at the top of
`ImplVariant.hpp` for a worked example (a `group_size` field), or
`shamalgs::primitives::impl::AtomicEarlyExit` in `is_all_true.cpp` for a real one.

### Exposing more than one default per alternative

By default, `get_default_impl_list_<algo>()` lists exactly one instance per alternative type
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

Once the selector and dispatch are in place, wire it up end to end:

1. Header: declare `get_default_impl_list_<algo>`, `get_current_impl_<algo>`,
   `set_impl_<algo>`, `is_impl_set_<algo>` and
   `autoselect_impl_<algo>(const sham::DeviceScheduler_ptr &)` in the algorithm's `impl`
   namespace.
2. Python bindings (`shampylib/src/pyShamalgs.cpp` or `pyShamtree.cpp`): expose all the
   user-facing functions declared in the header under the relevant submodule.
3. Unit test: loop over `get_default_impl_list_<algo>()`, calling `set_impl_<algo>` before each
   run, then restore the implementation that was active before the loop.
4. Benchmark script (`examples/benchmarks/`, if one exists for the algorithm): same loop,
   extracting the implementation's display name with `json.loads(impl)["implementation"]`; call
   `autoselect_impl_<algo>()` first when `is_impl_set_<algo>()` is `False`.

## Related files

- `shamalgs/include/shamalgs/ImplVariant.hpp` — authoritative reference for
  `ImplVariantGlobal`'s API.
