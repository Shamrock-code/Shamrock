#include "options.hpp"

void compute_flux_k0_kconst(u32 ipart,u32 nbins,u32 kp,const accfltrw_t gij,const accfltr_t tabflux,const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux);
void compute_flux_k0_kadd(u32 ipart,u32 nbins,u32 kp,const accfltrw_t gij,const accfltr_t tabflux,
                                                   const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,
                                                   const accfltrw_t flux);
void compute_flux_k0_kmul(u32 ipart,u32 nbins,u32 kp,const accfltrw_t gij,const accfltr_t tabflux,const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux);

void compute_flux_kconst(u32 ipart,u32 nbins,u32 kflux,u32 kp,const accfltrw_t gij,const accfltr_t tabipiflux,const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux);
void compute_flux_kadd(u32 ipart,u32 nbins,u32 kflux,u32 kp,const accfltrw_t gij,
                                                const accfltr_t tabipiflux,
                                                const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,
                                                const accfltrw_t flux);
void compute_flux_kmul(u32 ipart,u32 nbins,u32 kflux,u32 kp,const accfltrw_t gij,
                                                const accfltr_t tabipiflux,
                                                const accfltr_t tabK1,const accfltr_t tabK2,
                                                const accfltrw_t flux);


void compute_flux_intflux_kconst(u32 ipart,u32 nbins,u32 kflux,u32 kp,const accfltrw_t gij,
                                                         const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                                                         const accfltr_t tabK1,const accfltr_t tabK2,
                                                         const accfltrw_t flux,const accfltrw_t intflux);
void compute_flux_intflux_kadd(u32 ipart,u32 nbins,u32 kflux,u32 kp,const accfltrw_t gij,
                                                          const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                                                          const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,
                                                          const accfltrw_t flux,const accfltrw_t intflux);
void compute_flux_intflux_kmul(u32 ipart,u32 nbins,u32 kflux,u32 kp,const accfltrw_t gij,
                                                        const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                                                        const accfltr_t tabK1,const accfltr_t tabK2,
                                                        const accfltrw_t flux,const accfltrw_t intflux);
                                                                                              