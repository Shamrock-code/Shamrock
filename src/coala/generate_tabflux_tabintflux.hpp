#include "options.hpp"
void compute_tabflux_k0(u32 nbins,u32 kp,flt* massgrid,flt* massbins,flt* tabflux);
void compute_tabflux_tabintflux(u32 nbins,u32 kflux,u32 kp,flt* massgrid,flt* massbins,
                                 flt* tabipiflux,flt* tabipiAintflux,flt* tabipiBintflux);
