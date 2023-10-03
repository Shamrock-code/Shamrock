<picture>
   <source media="(prefers-color-scheme: dark)" srcset="doc/shamrock-doc/src/images/no_background_nocolor.png"  width="600">
   <img alt="text" src="doc/logosham_white.png" width="600">
 </picture>

# Shamrock Backends Library

This is the backends library in shamrock, this is used to provide abstraction 
to multiple compute backends (Sycl, Cuda, Rocm, ...), through a header only approach.
It's aim is to provide ways to programm generically with multiple backends, but also try 
to interfere as little as possible with the underliying API.

