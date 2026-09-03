# Coding Conventions

The following coding conventions are followed when developing Shamrock. In practice, there may be slight deviations from these guidelines 😅. Please notify or raise an issue if these conventions are not followed somewhere in the code.

## C++ Style Guide

### General Rules
- No tabs (use spaces for indentation).
- No raw pointers without wrapper or smart pointer.
- Use `T{}` for zero initialization of template types instead of `T(0)` to ensure compatibility with vectors and other complex types.
- Use exceptions for error handling with `shambase::throw_with_loc<exception type>` to carry source location information.
- Use `// TODO:` in the code and `@todo` in Doxygen documentation.
- Use `sham::kernel_call` when possible to invoke kernels.

## Naming Conventions

Most of the rules below are checked by `readability-identifier-naming` in
`.clang-tidy`, reported as **warnings** (not build-breaking errors) by the
clang-tidy CI job. One caveat: that check's `CamelCase` style only rejects
a name that has an underscore or a lowercase first letter — it can't tell
a genuine acronym (`AMD`, `CUDA`, `CG` — fine as-is, see Enum Values below)
from a plain word typed in caps by mistake (`UNKNOWN`, `MULTIGRID` — not
fine, should be `Unknown`/`Multigrid`). It accepts both silently, so the
second kind has to be caught by hand until renamed.

### Primitive Types

Primitive types are basic types representable by the actual hardware, typically integers, floats, and SYCL vectors.

Since Shamrock uses binary manipulation extensively, all types are named with a prefix (`u` for unsigned, `i` for signed integers, `f` for floats) followed by the number of bits. This can optionally be followed by `_x` where `x` is the number of elements in a vector.

**Primitive types:** `i64`, `i32`, `i16`, `i8`, `u64`, `u32`, `u16`, `u8`, `f16`, `f32`, `f64`

**Vector examples:** `f64_3`, `u64_16`, ...

### Classes, Structs, and Enums

Classes, structs, and enums in Shamrock follow PascalCase naming scheme, where each word starts with a capital letter.

**Example:** `IMeanIKindaLikeThisCaseTheOthersAreLessReadableToMe`

### Functions

Functions in Shamrock use snake_case to distinguish them from classes.

**Example:** `is_this_informatics_or_physics(...)`

### Variables, Members, and Constants

Local variables, function parameters, and class/struct member variables
use `lower_case`.

This also applies to constants, including `static constexpr` class
members that mirror a math/physics symbol (e.g. a kernel radius or a
side count) — name the identifier `lower_case` like any other member
(`rkern`, `nside`) rather than capitalizing it to look like the symbol.

### Namespaces

Namespaces use a single lowercase word, with no underscores
(`shamrock`, `shammodels`, `solvergraph`).

### Macros

Preprocessor macros use `UPPER_CASE` (`MPICHECK`, `NODE_EDGES`).

### Enum Values

Enum values (the enumerators inside a `class`/`enum class`) use
**acronym-preserving CamelCase**, same as the enum type itself
(`Periodic`, `Reflective`, `VanLeer`):

- Each word is capitalized.
- A word that is a recognized acronym is kept fully capitalized as that
  one word, instead of only capitalizing its first letter — e.g.
  `AMD`, `CPU`, `GPU`, `CUDA`, `CG`, `PCG`, `HLL`.
- A plain English word typed in caps only for visual consistency with its
  neighbors is *not* an acronym and should be normalized to a single
  capital letter — `UNKNOWN` → `Unknown`, `MULTIGRID` → `Multigrid`.
- When the value names a specific external technology, library, or
  published algorithm that already has its own established spelling,
  match that spelling instead of deriving one mechanically — `OpenMP`
  (never `OPENMP`/`Openmp`), `ROCm` (not `ROCM`), `BiCGSTAB` (not
  `BICGSTAB`/`Bicgstab`).

Note that `readability-identifier-naming`'s `CamelCase` check in
`.clang-tidy` can't distinguish an acronym from a plain word typed in caps
by mistake — it accepts both `AMD` and `UNKNOWN` without a warning, so the
second bucket above has to be caught by hand.

### File Naming

- A file that implements a single class or struct is named after that
  type in CamelCase, matching the type name exactly (e.g.
  `PatchDataField.hpp` for `class PatchDataField`).
- A file that holds a free-function algorithm, a kernel, or a set of
  related utility functions with no single owning type uses
  `lower_case` (e.g. `compute_ranges.hpp`, `key_morton_sort.hpp`).
- This isn't enforced by clang-tidy — there's no clang-tidy check for
  file names — so it's a convention to apply during review, not a
  generated warning.

## Template Conventions

Type template parameters use `CamelCase`, usually prefixed with `T`
(`Tvec`, `Tscal`, see below). Non-type template parameters (a `u32`,
`int`, or `bool` template value) use `lower_case`, same as a regular
variable (e.g. `template<class Tile, u32 group_size>`).

### Vector and Scalar Templates

Since many models can be implemented in Shamrock, utilities/classes are implemented for any primitive types. Generic classes use the following pattern:

```c++
template<class Tvec>
class Whateva {
    using Tscal = shambase::VecComponent<Tvec>;
    static constexpr u32 dimension = shambase::VectorProperties<Tvec>::dimension;
};
```

`Tvec` is sufficient to infer both the scalar type and the dimension, simplifying the template.

**Conventions:**
- `Tscal` for template scalar types
- `Tvec` for template vector types

### Special Template Types

#### Morton & Hilbert Codes

Morton codes and Hilbert codes shall be named `Tmorton` and `THilbert` respectively, since they will be templated.

## Documentation Standards

### File Headers

Every C++ file must start with the license banner followed by `#pragma once`:

```c++
// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once
```

### Pragma once

Every header file must include `#pragma once` after the license banner to prevent multiple inclusions. This is faster and more convenient than traditional include guards.

**Required header structure:**
```c++
// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

// ... rest of header content
```

**Note:** The `buildbot/check_pragma_once.py` utility checks for correct pragma once usage and will report files that don't have it.

### Doxygen Documentation

Every C++ file must include a Doxygen file header comment block:

```c++
/**
 * @file filename.cpp/hpp
 * @author Name (email)
 * @brief Brief description of the file
 *
 */
```

**Author format:** Use `@author Name (email)` format in docstrings.

### Code Documentation

- Use generic Doxygen documentation that focuses on API usage in code examples
- Expected outputs from specific input data can be documented below the code example
- Keep code examples abstract and not tied to specific data construction steps

## Testing Conventions

For detailed information on how to run tests, see the [Testing Guide](../testing.md).

**Quick reference:**
- Build the project using `shammake`
- Run unit tests with `./shamrock_test --unittest`
- Run MPI tests with `mpirun -np <ranks> ./shamrock_test --unittest`
- When running tests with MPI, providing the wrong number of MPI ranks will cause the test to be skipped
