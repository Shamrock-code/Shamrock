# Sycl tricks


## SYCL in-order queues

```c++
sycl::queue q {..., sycl::property::queue::in_order}
```


## Dealing with sycl accessors in loop macros

### SYCL parallel for

```cpp
shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

    sycl::accessor ...

    passing variables

    cgh.parallel_for([](sycl::range) ...,[=](sycl::item ...){
        ... kernel space
    });
});
```

### Accessor class design

```cpp

struct CustomAcc{
    sycl::accessor ...
    passing variables
};

//withing a function templated on the accessor and the functor
shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

    CustomAcc local = acc_arg;

    cgh.parallel_for([](sycl::range) ...,[=](sycl::item ... gid){
       func(gid, acc_arg);
    });
});
```

## Prefetching

https://developer.nvidia.com/blog/boosting-application-performance-with-gpu-memory-prefetching/
```cuda
double v0, v1, v2, v3;
for (i=threadIdx.x, ctr=0; i<imax; i+= BLOCKDIMX, ctr++) {
  ctr_mod = ctr%4;
  if (ctr_mod==0) { // only fill the buffer each 4th iteration
    v0=arr[i+0* BLOCKDIMX];
    v1=arr[i+1* BLOCKDIMX];
    v2=arr[i+2* BLOCKDIMX];
    v3=arr[i+3* BLOCKDIMX];
  }
  switch (ctr_mod) { // pull one value out of the prefetched batch
    case 0: locvar = v0; break;
    case 1: locvar = v1; break;
    case 2: locvar = v2; break;
    case 3: locvar = v3; break;
  }
  <lots of instructions using locvar, for example, transcendentals>
}
```
