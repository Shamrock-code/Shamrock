#include "options.hpp"

flt xhat(flt hj,flt xj,flt node);
void L2proj_gij_GQ(u32 nbins, u32 kflux, flt* massgrid, flt* massbins, u32 Q,flt* vecnodes,flt* vecweights, flt* gij);
void L2proj_gij_GQ_kmul(u32 nbins, u32 kflux, flt* massgrid, flt* massbins, u32 Q,flt* vecnodes,flt* vecweights, flt* gij);
void L2proj_kadd_GQ(u32 nbins, u32 kp, flt* massgrid, flt* massbins, u32 Q,flt* vecnodes,flt* vecweights,
					flt* tabK1F1, flt* tabK2F1, flt* tabK1F2, flt* tabK2F2);