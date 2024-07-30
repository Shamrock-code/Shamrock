//------------------------------
// Mathematic functions
//------------------------------
#include "options.hpp"
#include <math.h>

flt acoth(flt x){
    return ((flt)5e-1)*log((((flt)1) + x)/(x - ((flt)1)));
}
