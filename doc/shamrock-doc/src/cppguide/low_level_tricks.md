# Low Level tricks

The tricks described below must only be used in last resort.

## Move semantics and unique_ptr

```c++
struct Foo
{
    static inline int count{ 0 };
    int id;

    Foo() : id(count++) { std::cout << "(id: " << id << ") CTOR" << std::endl; }
    Foo(const Foo& other) : id(count++) { std::cout << "(id: " << id << ") Copy CTOR From ID " << other.id << std::endl; }
    Foo(Foo&& other) : id(count++) { std::cout << "(id: " << id << ") Move CTOR From ID " << other.id << std::endl; }
    const Foo& operator=(const Foo& other) { std::cout << "(id: " << id << ") Copy Assignment Operator RHS ID " << other.id << std::endl; return *this; }
    const Foo& operator=(Foo&& other) { std::cout << "(id: " << id << ") Move Assignment Operator RHS ID " << other.id << std::endl; return *this; }
    ~Foo() { std::cout << "(id: " << id << ") DTOR" << std::endl; }
};

template<typename T>
auto take(std::unique_ptr<T>& o) -> T {
    if(!o) throw "";
    std::unique_ptr<T> tmp = std::exchange(o,{});
    return T(std::move(*tmp));
}

int main(int argc, char* argv[])
{

    std::unique_ptr<int> oi = std::make_unique<int>(1);
    int i = take(oi);
    assert(i == 1);
    assert(!bool(oi));

    auto x = std::make_unique<Foo>();
    Foo f (take(x));
    return 0;
}
```

The take function trigger the move and not the copy

## Avoid if inclusions

Let say that we have 4 checks and the following function

```cpp
bool check1(int a);
bool check2(int a);
bool check3(int a);
bool check4(int a);

int func(int i){
    if(check1(i)){
        if(check2(i)){
            if(check3(i)){
                if(check4(i)){
                    return i;
                }
            }
        }
    }
    return 0;
}
```

In good old C multiple way to handle those checks in a cleaner manner.

 - Using gotos
```cpp
int func_goto(int i){
    if(!check1(i)){
        goto finish;
    }
    if(!check2(i)){
        goto finish;
    }
    if(!check3(i)){
        goto finish;
    }
    if(!check4(i)){
        goto finish;
    }
    return i;
finish:
    return 0;
}
```

 - Using do while
```cpp
int func_do_while(int i){
    do{
        if(!check1(i)) break;
        if(!check2(i)) break;
        if(!check3(i)) break;
        if(!check4(i)) break;

        return i;
    }while(0);

    return 0;
}
```

Note that the 3 function showed previously exibit exactly the same assembly code but the last one with a lambda exibit a slightly different assembly 


```cpp
int func_lambda(int i){
    auto ret_case = [](){
        return 0;
    };

    if(!check1(i)) return ret_case();
    if(!check2(i)) return ret_case();
    if(!check3(i)) return ret_case();
    if(!check4(i)) return ret_case();

    return i;
}
```


<table>
<tr>
<th>Original</th>
<th>Lambda return</th>
</tr>
<tr>
<td valign="top">


```x86asm
func(int):
        push    rbx
        mov     ebx, edi
        call    check1(int)
        test    al, al
        je      .LBB0_4
        mov     edi, ebx
        call    check2(int)
        test    al, al
        je      .LBB0_4
        mov     edi, ebx
        call    check3(int)
        test    al, al
        je      .LBB0_4
        mov     edi, ebx
        call    check4(int)
        test    al, al
        jne     .LBB0_5
.LBB0_4:
        xor     ebx, ebx
.LBB0_5:
        mov     eax, ebx
        pop     rbx
        ret
```


</td>
<td valign="top">

```x86asm
func(int):
        push    rbp
        push    rbx
        push    rax
        mov     ebx, edi
        call    check1(int)
        xor     ebp, ebp
        test    al, al
        je      .LBB0_4
        mov     edi, ebx
        call    check2(int)
        test    al, al
        je      .LBB0_4
        mov     edi, ebx
        call    check3(int)
        test    al, al
        je      .LBB0_4
        mov     edi, ebx
        call    check4(int)
        xor     ebp, ebp
        test    al, al
        cmovne  ebp, ebx
.LBB0_4:
        mov     eax, ebp
        add     rsp, 8
        pop     rbx
        pop     rbp
        ret
```

</td>
</tr>
</table>

## Read only exported variables
(source : https://stackoverflow.com/questions/599365/what-is-your-favorite-c-programming-trick)

For creating a variable which is read-only in all modules except the one it's declared in:

 - Header1.h:
```c
#ifndef SOURCE1_C
   extern const int MyVar;
#endif
```

 - Source1.c:
```c
#define SOURCE1_C
#include Header1.h // MyVar isn't seen in the header

int MyVar; // Declared in this file, and is writeable
```

 - Source2.c
```c
#include Header1.h // MyVar is seen as a constant, declared elsewhere
```

## Bit shift up to bit length
(source : https://stackoverflow.com/questions/599365/what-is-your-favorite-c-programming-trick)

Bit-shifts are only defined up to a shift-amount of 31 (on a 32 bit integer)..

What do you do if you want to have a computed shift that need to work with higher shift-values as well? Here is how the Theora vide-codec does it:

```cpp
unsigned int shiftmystuff (unsigned int a, unsigned int v)
{
  return (a>>(v>>1))>>((v+1)>>1);
}
```
Or much more readable:

```cpp
unsigned int shiftmystuff (unsigned int a, unsigned int v)
{
  unsigned int halfshift = v>>1;
  unsigned int otherhalf = (v+1)>>1;

  return (a >> halfshift) >> otherhalf; 
}
```
Performing the task the way shown above is a good deal faster than using a branch like this:

```cpp
unsigned int shiftmystuff (unsigned int a, unsigned int v)
{
  if (v<=31)
    return a>>v;
  else
    return 0;
}
```

## X-macros

imagine you want to define multiple variables or function by only referencing a list in one place.

```cpp
#define XMACRO \
   X(float)    \
   X(int)      \
   X(double)


#define X(_arg_) int var##_arg_;
XMACRO
#undef X
```

this code is equivalent to 

```cpp
int varfloat;
int varint;
int vardouble;
```

It can be usefull to avoid copy-paste errors

## State machine and jump table

Jump to a function depending on an enum state

```cpp
#include <functional>

#define XMACRO         \
   X(STATE1, func1)    \
   X(STATE2, func2)    \
   X(STATE3, func3)

enum States{
    #define X(_state_, _func_) _state_,
    XMACRO
    #undef X
};

void func1();
void func2();
void func3();

std::function<void()> jumptable [3] ={
    #define X(_state_, _func_) _func_,
        XMACRO
    #undef X
};
```

## Hinting the compiler with branch info

```cpp
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

void foo(int arg)
{
    if (unlikely(arg == 0)) {
        return;
    }

    if (likely(arg == 0)) {
        return;
    }
}
```


## Bit fields

```cpp
struct strc {
    int a:3; // 3 bits for a
    int b:4; // 4 bits for b
};

strc make_strc(){
    strc ret;
    ret.a = 3;
    ret.b = 7;
    return ret;
}
```

The compiler even gives warnings if the input variable is not in the correct range.

## Bit fields with 0 len

Source : https://stackoverflow.com/questions/132241/hidden-features-of-c

I discoverd recently 0 bitfields.
```cpp
struct {
  int    a:3;
  int    b:2;
  int     :0;
  int    c:4;
  int    d:3;
};
```
which will give a layout of
```cpp
000aaabb 0ccccddd
```
instead of without the :0;
```cpp
0000aaab bccccddd
```
The 0 width field tells that the following bitfields should be set on the next atomic entity (char)

## Duff's device

https://en.wikipedia.org/wiki/Duff%27s_device


## Analog litterals
source : http://web.archive.org/web/20111026205138/http://weegen.home.xs4all.nl/eelis/analogliterals.xhtml
```cpp
// Note: The following is all standard-conforming C++, this is not a hypothetical language extension.

#include "analogliterals.hpp"
#include <cassert>

int main ()
{
  using namespace analog_literals::symbols;

// Consider:

  unsigned int a = 4;

// Have you ever felt that integer literals like "4" don't convey the true size of the value they denote? If so, use an analog integer literal instead:

  unsigned int b = I---------I;

  assert( a == b );

// Due to the way C++ operators work, we must use N*2+1 dashes between the I's to get a value of N:

  assert( I-I == 0 );
  assert( I---I == 1 );
  assert( I-----I == 2 );
  assert( I-------I == 3 );

// These one-dimensional analog literals are of type analog_literals::line<N>, which is convertible to unsigned int.

// In some cases, two-dimensional analog literals are appropriate:

  unsigned int c = ( o-----o
                     |     !
                     !     !
                     !     !
                     o-----o ).area;

  assert( c == (I-----I) * (I-------I) );

  assert( ( o-----o
            |     !
            !     !
            !     !
            !     !
            o-----o ).area == ( o---------o
                                |         !
                                !         !
                                o---------o ).area );

// Two-dimensional analog literals are of type analog_literals::rectangle<X, Y> which exposes static member constants width, height, and area.

/* As an example use-case, imagine specifying window dimensions in a GUI toolkit API using:

   window.dimensions = o-----------o
                       |           !
                       !           !
                       !           !
                       !           !
                       o-----------o ;

Who said C++ was unintuitive!? */

// But wait, there's more. We can use three-dimensional analog literals, too:

  assert( ( o-------------o
            |L             \
            | L             \
            |  L             \
            |   o-------------o
            |   !             !
            !   !             !
            o   |             !
             L  |             !
              L |             !
               L|             !
                o-------------o ).volume == ( o-------------o
                                              |             !
                                              !             !
                                              !             !
                                              o-------------o ).area * int(I-------------I) );

// Three-dimensional analog literals are of type analog_literals::cuboid<X, Y, Z> which exposes static member constants width, height, depth, and volume. In addition, three free-standing functions top, side, and front are provided which yield rectangles:

  assert( top( o-------o
               |L       \
               | L       \
               |  o-------o
               |  !       !
               !  !       !
               o  |       !
                L |       !
                 L|       !
                  o-------o ) == ( o-------o
                                   |       !
                                   !       !
                                   o-------o ) );

// The current implementation has one restriction on cuboid dimensions: the height of a cuboid literal must be at least its depth + 2.

// Note that storing these literals directly in a variable requires you to specify the dimension sizes:

  analog_literals::rectangle<4, 2> r = o---------o
                                       |         !
                                       !         !
                                       o---------o;

// This of course defeats the purpose of using the analog literal. C++0x's proposed `auto' feature would come in quite handy here. We can actually fix this problem partially (using the stack-ref-to-temporary's-base trick used by Alexandrescu's ScopeGuard), but we would no longer be able to use the values in ICE's, and frankly I think this madness has gone far enough already.

}
```

## Switch with multiple variables

```cpp
int func(int i, int j){
    if(i == 0 && j ==0){
        return fct0();
    }else if(i == 1 && j ==0){
        return fct1();
    }else if(i == 0 && j ==1){
        return fct2();
    }else if(i == 1 && j ==1){
        return fct3();
    }

    return -1;
}
```

This may be slow looking at the assembly (O3) : 

```x86asm
func(int, int):                              # @func(int, int)
        mov     eax, edi
        or      eax, esi
        jne     .LBB0_1
        jmp     fct0()                        # TAILCALL
.LBB0_1:
        cmp     edi, 1
        jne     .LBB0_3
        test    esi, esi
        jne     .LBB0_3
        jmp     fct1()                        # TAILCALL
.LBB0_3:
        test    edi, edi
        jne     .LBB0_5
        cmp     esi, 1
        jne     .LBB0_5
        jmp     fct2()                        # TAILCALL
.LBB0_5:
        cmp     edi, 1
        jne     .LBB0_7
        cmp     esi, 1
        jne     .LBB0_7
        jmp     fct3()                        # TAILCALL
.LBB0_7:
        mov     eax, -1
        ret
```

we have multiple test and a lot of branching whereas the following case exibit no branching

```cpp
int func(int i, int j){
    switch(i | (j << 1)){
        case 0: return fct0();break;
        case 1: return fct1();break;
        case 2: return fct2();break;
        case 3: return fct3();break;
    }
    return -1;
}
```

assembly (O3)
```x86asm
func(int, int):                              # @func(int, int)
        add     esi, esi
        or      esi, edi
        cmp     esi, 3
        ja      .LBB0_6
        jmp     qword ptr [8*rsi + .LJTI0_0]
.LBB0_2:
        jmp     fct0()                        # TAILCALL
.LBB0_6:
        mov     eax, -1
        ret
.LBB0_3:
        jmp     fct1()                        # TAILCALL
.LBB0_4:
        jmp     fct2()                        # TAILCALL
.LBB0_5:
        jmp     fct3()                        # TAILCALL
.LJTI0_0:
        .quad   .LBB0_2
        .quad   .LBB0_3
        .quad   .LBB0_4
        .quad   .LBB0_5
```
