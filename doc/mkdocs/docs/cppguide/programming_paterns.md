## Programming paterns

# Implementation selection

```c++
#include <variant>
#include <iostream>

enum Implementation{
    CPU, GPU
};

namespace details{

    class SortTableGPU{ public:
        void do_stuff(){
            std::cout << "Hello from GPU" << std::endl;
        }
    };

    class SortTableCPU{ public:
        void do_stuff(){
            std::cout << "Hello from CPU" << std::endl;
        }
    };

    using sort_var_t = std::variant<
        details::SortTableCPU,
        details::SortTableGPU
    >;

    inline constexpr sort_var_t makeSortTable(Implementation impl){
        if(impl == GPU){
            return SortTableGPU();
        }
        return SortTableCPU();
    }
}


class SortTable{

    using var_t = std::variant<
        details::SortTableCPU,
        details::SortTableGPU
    >;

    var_t impl;

    public:

    SortTable(Implementation choice): impl(details::makeSortTable(choice)){}

    void do_stuff(){
        std::visit([](auto && tmp){
            tmp.do_stuff();
        },impl);
    }
};

int main(){
    SortTable{CPU}.do_stuff();
    SortTable{GPU}.do_stuff();
}
```
