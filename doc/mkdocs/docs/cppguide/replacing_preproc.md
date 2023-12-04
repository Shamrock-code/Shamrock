# Replacing C preprocessors

## X macros

```cpp
template< template<class> class Container >
class VariantContainer {
    std::variant<Container<f32>,Container<f64>> variant;
}
```

Such Variant container can be used as such 

```cpp
template<class T>
struct Field{
    std::vector<T> vec;
}

using VariantField = VariantContainer<Field>;
```

here VariantField is equivalent to a type like this : 

```cpp
class VariantField {
    std::variant<Field<f32>,Field<f64>> variant;
}
```

# If defs

Usually you may have a Compile flag to toogle some features in C or Fortran code, in order to do so we usually write something like this

```cpp
void func(){
    #ifdef DOSTUFF
        do_stuff();
    #else
        dont_do_stuff();
    #endif
}
```

In c++ such patern can be replaced by constexpr bool's.

```cpp
// somewhere in the code
#ifdef DOSTUFF
    constexpr bool do_stuff_defined = true;
#else
    constexpr bool do_stuff_defined = false;
#endif

void func(){
    if constexpr (do_stuff_defined){
        do_stuff();
    }else{
        dont_do_stuff();
    }
}
```

since this if is marked as `constexpr` it will be evaluated at compile time, even without optimisation: 

For `clang 13.0.1` without any flags : 

```x86asm
func():                               # @func()
    push    rbp
    mov     rbp, rsp
    call    dont_do_stuff()
    pop     rbp
    ret
```

here we have no comparaison, and `do_stuff` is not even referenced as expected