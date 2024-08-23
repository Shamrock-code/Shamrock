# Coding Conventions

The following avec coding conventions i'm trying to stick to when i devellop shamrock, but in pratice it happends that i might have deviated slightly from them 😅. Don't hesitate to notify me or raise an issue if the following is not followed somewhere in the code.

## Type naming

in shamrock we distinguish between various types of objects:

### primitive types

primitive types are basic types representable by the actual hardware, typically ints, floats, sycl vectors.

Since in shamrock some binary manipulation tend to be used all types are named by a prefix (`u` for unsigned, `i` for int, `f` for floats) followed by the number of bits representing it. This cal optionally be followed by `_x` where `x` is the number of elements in a vector.

primitives : `i64`,`i32`,`i16`,`i8 `,`u64`,`u32`,`u16`,`u8 `,`f16`,`f32`,`f64`

exemples of vectors : `f64_3`, `u64_16`, ...

### Classes, structs, enums

Classes, structs and enums in shamrock follows the PascalCase naming scheme, each word is starts with a captial letter,

for exemple : `IMeanIKindaLikeThisCaseTheOthersAreLessReadableToMe`

### functions

Functions in shamrock tend to be called using snake_case to distinguish them from classes.

for exemple : `is_this_informatics_or_physics(....)`

### Special types


#### vectors and scalar templates

Since many models can be implemented in shamrock it is required to have some utilities/classes implemented for any primitive types. To deal with that generic classes are implemented using the following patern.

```c++
template<class Tvec>
class Whateva{

    using VectorProperties = VectorProperties<Tvec>;
    using Tscal = VectorProperties::Tscal;
    static constexpr u32 dimension = VectorProperties::dimension;

}
```

`Tvec` is sufficent to infer both the scalar type and the dimension, simplifying the actual template.

The following convention applies : `Tscal` for template scalar, `Tvec` for template vector

#### Morton & Hilbert codes

Morton codes and hilbert code shall be named `Tmorton`, `THilbert` since they will be templated


## C++ style guide

 - no tabs
 - no raw pointers without wrapper or smart pointer.
 - no inheritance
