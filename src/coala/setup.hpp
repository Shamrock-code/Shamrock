#include "options.hpp"
#include <stdbool.h> 
#include <math.h>


// extern unsigned int nbins;
// extern unsigned int kflux;
// extern unsigned int kp;
// extern unsigned int Q;
// extern flt dthydro;
// extern unsigned int ndthydro;
// extern flt massmax;
// extern flt massmin;
// extern flt eps;

extern unsigned int scheme;
extern unsigned int kernel;
extern bool process;
extern bool results;
extern bool save;
extern unsigned int grid;

void init_path_files(unsigned int nbins, unsigned int kflux,unsigned int scheme, unsigned int kernel, char* path_data);

// void printf_qp(__float128 var);

// inline __float128 pow( __float128 x, __float128 y){
//     return powq(x,y);
// }

// inline __float128 log( __float128 x){
//     return logq(x);
// }

// inline __float128 exp( __float128 x){
//     return expq(x);
// }

// inline __float128 abs( __float128 x){
//     return fabsq(x);
// }

// inline __float128 sqrt( __float128 x){
//     return sqrtq(x);
// }

// inline __float128 fmin( __float128 x, __float128 y){
//     return fminq(x,y);
// }

// inline __float128 isnan( __float128 x){
//     return isnanq(x);
// }