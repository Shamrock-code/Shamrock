# Communication Protocol



The idea of the communication protocol is to abstract the mpi part of communicating various objects in shamrock through the use of a wrapper buffer.


## Protocols

The type of protocol is selected when building the object, by passing a value of the Protocol enum as the last parameter, those can be either :
 - CopyToHost (copy from the device back to the host for the communication)
 - DirectGPU (communicate directly from the device)
 - DirectGPUFlatten (communicate directly from the device but by flattening any sycl vector type)


```{note}
`DirectGPUFlatten` is to by-pass the bug of openmpi when sending structs with direct GPU
```

## Building a buffer
To build a buffer you can either :

 - build it from size information you may have :
```cpp
CommDetails<...type...> det = ...;
CommBuffer buf {det,DirectGPU};
```
 - build it from a copy of an object :
```cpp
CommBuffer buf {...obj...,DirectGPU};
```
 - build it from an object :
```cpp
CommBuffer buf {std::move(...obj...),DirectGPU};
```
 - build it from a copy of an object & specify infos:
```cpp
CommDetails<...type...> det = ...;
CommBuffer buf {...obj...,det,DirectGPU};
```
 - build it from an object & specify infos:
```cpp
CommDetails<...type...> det = ...;
CommBuffer buf {std::move(...obj...),det,DirectGPU};
```

## Recovering data from a buffer

You can :
- copy data back from the buffer :
```cpp
CommBuffer buf;
auto obj = buf.copy_back();
```
- destruct the buffer and get the object :
```cpp
CommBuffer buf;
auto obj = CommBuffer<..type...>::convert(std::move(buf));
```

## Exemple use of a buffer

on the sender side :
```cpp

CommBuffer buf {...obj to send..., DirectGPU};

CommRequests rqs;
buf.isend(rqs, 1,0,MPI_COMM_WORLD);

rqs.wait_all();

```

on the receiver side :

```cpp
CommDetails<sycl::buffer<T>> details;

details.comm_len = npart;

CommBuffer buf {details,DirectGPU};

CommRequests rqs;
buf.irecv(rqs, 0,0,MPI_COMM_WORLD);
rqs.wait_all();

sycl::buffer<T> buf_comp2 = buf.copy_back();
```
