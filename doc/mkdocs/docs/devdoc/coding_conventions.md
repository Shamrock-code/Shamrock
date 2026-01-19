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

## Template Conventions

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
