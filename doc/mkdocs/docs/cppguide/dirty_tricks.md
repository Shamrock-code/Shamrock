
# Dirty tricks

## Fixing constructor issues (The dirty way)

if you have a class `MyClass` having some custom constructors or deleted constructors. A way to avoid dealing with constructors issues is to wrap the class in a `std::unique_ptr`.

Exemple :

```cpp
struct MyClass{
    int* ptr;

    MyClass(){
        prt = new int(0);
    }

    ~MyClass(){
        delete ptr;
    }

}
```

Such class is ill formed in a non trivial way because it doesn't follow the rule of Three/Five/whatever c++ weird mechanics.

But if you wrap this type in a `std::unique_ptr` then it will behave correctly.

## Container template

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




## Register function with static init

```cpp

using fct_sig = std::function<void()>;

inline std::vector<fct_sig> static_init_fct_list {};

struct StaticInitClass{
    inline explicit MPIDTypeinit(fct_sig t){
        static_init_fct_list.push_back(std::move(t));
    }
};

```

If in the code you write the following code block :

```cpp
void fct_to_register(){...}

void (*fct_ptr)() =fct_to_register;

StaticInitClass static_init_instance (fct_ptr);
```

The function `fct_to_register` will be in `static_init_fct_list` when the main function is ran.


## Defining a protocol with hiden implementation

```cpp

enum ProtocolMode{
    Mode1, Mode2, Mode3
};

namespace details{
    template<class T, ProtocolMode mode>
    class ProtocolImpl;
}


template<class T>
class VariantProtocol{public:

    template<ProtocolMode mode>
    using Impl = std::unique_ptr<details::ProtocolImpl<T,mode>>;

    std::variant<
        Impl<Mode1>, Impl<Mode2>, Impl<Mode3>
    > var_protocol;

    VariantProtocol(T arg, ProtocolMode mode){
        switch(mode){
        case Mode1:
            var_protocol = std::make_unique<details::ProtocolImpl<T,Mode1>>(arg);
            break;
        case Mode2:
            var_protocol = std::make_unique<details::ProtocolImpl<T,Mode2>>(arg);
            break;
        case Mode3:
            var_protocol = std::make_unique<details::ProtocolImpl<T,Mode3>>(arg);
            break;
        default:
            throw std::invalid_argument("unknown mode");
            break;
        }
    }

    void call(){
        std::visit([](auto & protocol){
            if(!protocol){
                throw std::invalid_argument("the protocol is not initialized");
            }
            protocol->call();
        }, var_protocol);
    }

};


int main(void){

    float in = 0.f;

    VariantProtocol prot {in, Mode1};

    prot.call();

}
```

```cpp
namespace details{
    template<class T>
    class ProtocolImpl<T,Mode1>{
        T val;
        public:
        ProtocolImpl(T arg) : val(arg) {}
        void call(){
            std::cout << "protocol mode 1 " << val << std::endl;
        }
    };

    template<class T>
    class ProtocolImpl<T,Mode2>{
        T val;
        public:
        ProtocolImpl(T arg) : val(arg) {}
        void call(){
            std::cout << "protocol mode 2 " << val << std::endl;
        }
    };

    template<class T>
    class ProtocolImpl<T,Mode3>{
        T val;
        public:
        ProtocolImpl(T arg) : val(arg) {}
        void call(){
            std::cout << "protocol mode 3 " << val << std::endl;
        }
    };
}
```
