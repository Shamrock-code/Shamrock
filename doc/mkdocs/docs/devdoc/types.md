# Shamrock primitive types

In shamrock codebase, some binary utilities tend to be used. But the c++ standard doesn't specify the bit count of types such as `int`, `float`, ... In order to circuvent such issue. We rely on primitive types that have explicit bit count such as `uint_32t`, ... But they are cumbursome due to their ugly naming :). Hence to following type list


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

Here the prefix letter describe the nature of the object and the number, the number of bits. It is also possible to add a subscript with a number to specify a vector of such object exemple : `f64_3` describe a dimension 3 `f64` vector. The possible sizes are : 2,3,4,8,16

## literals

all of those types can be invoked using literals by specifying a value undescore the wanted type, exemple :

```c++
u64 a = 15486_u64
```
