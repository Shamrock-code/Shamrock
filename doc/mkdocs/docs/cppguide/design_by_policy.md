# Design : Policy based

The idea is to replace a specific usage patern of preprocessors.

Imagine you have a black box function `do_smth` , this function may have multiple implementation `do_smth_int1`, `do_smth_int2` where one or the other are more suitable to specific architectures. Usually in `c` it would be implemented like this. 

```c++
void do_smth( args ...){
    #ifdef _ARCHI1
    do_smth_int1(args...)
    #else
    do_smth_int2(args...)
    #endif
}
```

Such patern is not harmfull in itself but may be hard to read/debug for more complex usages.

## moving to policy design

The idea would be to setup something similar as "replacing if defs", but in a more general way to be able to pass more complex objects and to avoid exposing the name globally (if you use a lot of autocompletion you will like this :) )
```c++
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

The key part in such design is to have very abstract implementations :

Instead of this :

```c++
void do_stuff_arch1();
void do_stuff_arch2();
void do_stuff_arch3();
void do_stuff_arch4();
```

you want to have function like this : 

```c++
template<u32 arch_code>
void do_stuff();
```

then the original exemple may be rewritten as such : 

```c++
enum ArchCodes{
    ARCHI_1, ARCHI_UNKNOWN
};

#ifdef _ARCHI_1
constexpr ArchCodes arch_code = ARCHI_1;
#else
constexpr ArchCodes arch_code = ARCHI_UNKNOWN;
#endif

void do_smth( args ...){
    do_smth_int<arch_code>(args...);
}
```

## A better exemple

Imagine you are coding a GPU kernel but some parameters may have to be tweaked to squeeze the best performance out of the card. The policy design might be very revelant.

The global definition is : 
```c++
enum ArchCodes{
    ARCHI_1, ARCHI_UNKNOWN
};

#ifdef _ARCHI_1
constexpr ArchCodes arch_code = ARCHI_1;
#else
constexpr ArchCodes arch_code = ARCHI_UNKNOWN;
#endif
```

When implementing your function you can do something like this.

```c++
struct SMTHPolicy{
    enum ExecutionParams{
        WorkerCount = (arch_code == ARCHI_1) ? 16 : 8;
    };
    using Operator = std::plus<u32>;
}
```

or equivalently : 

```c++
struct SMTHPolicy{
    static constexpr u32 WorkerCount = (arch_code == ARCHI_1) ? 16 : 8;
    using Operator = std::plus<u32>;
}
```

and then : 

```c++
void do_smth( args ...){
    do_smth_int<SMTHPolicy>(args...);
}
```

The advantage of the using the `enum` case is that the type of `WorkerCount` is not specified and therefor can be better optimized