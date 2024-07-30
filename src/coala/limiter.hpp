#include "options.hpp"

flt minvalpolk1(flt a,flt b,flt xmin,flt xmax);
flt minvalpolk2(flt a,flt b,flt c,flt xmin,flt xmax);
flt minvalpolk3(flt a,flt b,flt c, flt d,flt xmin,flt xmax);
void minvalgh(u32 i,u32 kflux,u32 nbins,const accfltr_t massgrid,const accfltr_t massbins,const accfltrw_t gij,const accfltrw_t coeff_Leg,const accfltrw_t coeff_gh,const accfltrw_t tabminvalgh);
void gammafunction(u32 i,u32 kflux,u32 nbins,const accfltr_t massgrid,const accfltr_t massbins,const accfltrw_t tabgij,const accfltrw_t coeff_Leg,const accfltrw_t coeff_gh,const accfltrw_t tabminvalgh,const accfltrw_t tabgamma);

