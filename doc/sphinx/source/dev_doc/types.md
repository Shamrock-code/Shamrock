# Shamrock primitive types

In the Shamrock codebase, some binary utilities tend to be used. But the C++ standard doesn't specify the bit count of types such as `int`, `float`, ... In order to circumvent such issues, we rely on primitive types that have explicit bit count such as `uint_32t`, ... But they are cumbersome due to their ugly naming :). Hence the following type list


unsigned integers :
`u8`,
`u16`,
`u32`,
`u64`

signed integers :
`i8`,
`i16`,
`i32`,
`i64`

floating point numbers :
`f16`,
`f32`,
`f64`

Here the prefix letter describes the nature of the object, and the number describes the number of bits. It is also possible to add a subscript with a number to specify a vector of such objects, for example: `f64_3` describes a dimension 3 `f64` vector. The possible sizes are: 2, 3, 4, 8, 16

## Literals

All of those types can be invoked using literals by specifying a value underscore the wanted type, for example:

```c++
u64 a = 15486_u64
```
