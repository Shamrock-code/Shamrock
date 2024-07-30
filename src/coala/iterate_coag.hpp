#include "options.hpp"

void iterate_coag(u32 nparts, u32 nbins, u32 kflux, u32 kp, u32 Q,flt* massgrid, flt* massbins,flt* venodes, flt* vecweights, flt dthydro, u32 ndthydro,sycl::queue* queue);