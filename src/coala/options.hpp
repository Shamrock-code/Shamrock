#ifndef OPTIONS_HPP_INCLUDED
#define OPTIONS_HPP_INCLUDED

#include <stdlib.h> 
#include <string>
// #include <quadmath.h>
#include <ostream>
// #include <CL/sycl.hpp>
#include <sycl/sycl.hpp>


//definitions types
//allow to choose between simple, double precision and quadruple precision
// typedef float flt;
typedef double flt;
// typedef long double flt;
typedef long double flt_ld;
// typedef __float128 flt_ld;


typedef unsigned int u32;

// typedef sycl::accessor<flt, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer, sycl::access::placeholder::false_t, sycl::ext::oneapi::accessor_property_list<>> accfltrw_t;
typedef sycl::accessor<flt, 1, sycl::access::mode::read_write, sycl::access::target::device> accfltrw_t;
// typedef sycl::accessor<flt_ld, 1, sycl::access::mode::read_write, sycl::access::target::device> accfltldrw_t;

// typedef sycl::accessor<flt, 1, sycl::access::mode::discard_write, sycl::access::target::host_buffer> accfltw_t;
typedef sycl::accessor<flt, 1, sycl::access::mode::discard_write, sycl::access::target::device> accfltw_t;

// typedef sycl::accessor<flt, 1, sycl::access::mode::read, sycl::access::target::global_buffer> accfltr_t;
typedef sycl::accessor<flt, 1, sycl::access::mode::read, sycl::access::target::device> accfltr_t;

// typedef sycl::accessor<u32, 1, sycl::access::mode::read, sycl::access::target::global_buffer> accu32r_t;
typedef sycl::accessor<u32, 1, sycl::access::mode::read, sycl::access::target::device> accu32r_t;

typedef sycl::accessor<u32, 1, sycl::access::mode::discard_write, sycl::access::target::host_buffer> accu32w_t;


///////////////////////////////////////////////////////////
//sycl stuff
///////////////////////////////////////////////////////////
//define queue pointer


// void wait(sycl::queue* queue){
//     try {
//         queue->wait_and_throw();
//     } catch (sycl::exception const& e) {
//         printf("Caught synchronous SYCL exception: %s\n",e.what());
//     }
// }




///////////////////////////////////////////////////////////
//sycl buffers
///////////////////////////////////////////////////////////
/// full TDC copyright
#define __FILENAME__ std::string(strstr(__FILE__, "/src/") ? strstr(__FILE__, "/src/")+1  : __FILE__)
#define throw_w_pos(...) throw std::runtime_error( __VA_ARGS__ " ("+ __FILENAME__ +":" + std::to_string(__LINE__) +")");
#define _FREE(...)      {if(__VA_ARGS__ != NULL){ delete   __VA_ARGS__; __VA_ARGS__ = NULL; }else{ throw_w_pos("trying to free \"" #__VA_ARGS__ "\" but it was already free'd");}}
#define _FREE_ARR(...)  {if(__VA_ARGS__ != NULL){ delete[] __VA_ARGS__; __VA_ARGS__ = NULL; }else{ throw_w_pos("trying to free array \"" #__VA_ARGS__ "\" but it was already free'd");}}







#endif // OPTIONS_HPP_INCLUDED
