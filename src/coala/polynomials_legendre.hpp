#include "options.hpp"

flt coeffnorm(u32 i);
void compute_coeff_Leg(u32 k,flt coeff_Leg[]);
void compute_coeff_Leg_sycl(u32 i,u32 k,const accfltrw_t coeff_Leg);
flt Leg_P(u32 k,flt x);
flt Leg_P_sycl(u32 i,u32 k,const accfltrw_t coeff_Leg,flt x);


