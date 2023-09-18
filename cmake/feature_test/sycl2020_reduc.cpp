#include <sycl/sycl.hpp>

int main(void){
    const int N = 10000;
    sycl::buffer<int> buf_int(10000);
    
    sycl::buffer<int> recov{1};

    sycl::queue{}.submit([&](sycl::handler &cgh) {
        sycl::accessor global_mem{buf_int, cgh, sycl::read_only};

        auto reduc = sycl::reduction(recov, cgh, sycl::plus<>{});


        cgh.parallel_for(sycl::range<1>{N}, reduc, [=](sycl::id<1> idx, auto &sum) {
            sum.combine(global_mem[idx]);
        });
    });

    int rec;
    {
        sycl::host_accessor acc{recov, sycl::read_only};
        rec = acc[0];
    }
}