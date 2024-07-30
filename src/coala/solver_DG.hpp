#include "options.hpp"

flt CFL_k0_kconst(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltr_t tabflux, const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux, const accfltrw_t tabdflux, const accfltrw_t tabdtCFL);
void Lk0_func_kconst(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltr_t tabflux, const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux,const accfltrw_t Lk0);
void solver_k0_kconst(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,const accfltr_t tabflux,  const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux,const accfltrw_t Lk0,const accfltrw_t Lk0_1,const accfltrw_t Lk0_2, flt dt);

flt CFL_k0_kadd(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltr_t tabflux,const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,const accfltrw_t flux,const accfltrw_t tabdflux,const accfltrw_t tabdtCFL);
void Lk0_func_kadd(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltr_t tabflux,const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,const accfltrw_t flux,const accfltrw_t Lk0);
void solver_k0_kadd(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,const accfltr_t tabflux,const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,const accfltrw_t flux,const accfltrw_t Lk0,const accfltrw_t Lk0_1,const accfltrw_t Lk0_2, flt dt);

flt CFL_k0_kmul(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltr_t tabflux,const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux,const accfltrw_t tabdflux,const accfltrw_t tabdtCFL);
void Lk0_func_kmul(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltr_t tabflux,const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux,const accfltrw_t Lk0);
void solver_k0_kmul(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,const accfltr_t tabflux,const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux,const accfltrw_t Lk0,const accfltrw_t Lk0_1,const accfltrw_t Lk0_2, flt dt);


flt CFL_kconst(u32 i,u32 nbins,u32 kflux,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,
                                    const accfltr_t tabipiflux,const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux,const accfltrw_t tabdflux,const accfltrw_t tabdtCFL);

void Lk_func_kconst(u32 i,u32 nbins,u32 kflux,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,
                                            const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                                            const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t coeff_Leg,
                                            const accfltrw_t flux,const accfltrw_t intflux,const accfltrw_t Lk);
void solver_kconst(u32 i,u32 nbins, u32 kflux,u32 kp, const accfltr_t massgrid,const accfltr_t massbins,const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,
                                            const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                                            const accfltr_t tabK1,const accfltr_t tabK2,
                                            const accfltrw_t flux, const accfltrw_t intflux, const accfltrw_t coeff_Leg,const accfltrw_t coeff_gh,const accfltrw_t tabminvalgh,const accfltrw_t tabgamma,const accfltrw_t Lk,const accfltrw_t Lk_1,const accfltrw_t Lk_2,flt dt);

flt CFL_kadd(u32 i,u32 nbins,u32 kflux,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,
                                    const accfltr_t tabipiflux,const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,
                                    const accfltrw_t flux,const accfltrw_t tabdflux,const accfltrw_t tabdtCFL);
void Lk_func_kadd(u32 i,u32 nbins,u32 kflux,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,
                                        const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                                        const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,
                                        const accfltrw_t coeff_Leg,
                                        const accfltrw_t flux,const accfltrw_t intflux,const accfltrw_t Lk);
void solver_kadd(u32 i,u32 nbins, u32 kflux, u32 kp,const accfltr_t massgrid,const accfltr_t massbins,const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,
                                        const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                                        const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,
                                        const accfltrw_t flux,const accfltrw_t intflux,const accfltrw_t coeff_Leg,const accfltrw_t coeff_gh,const accfltrw_t tabminvalgh,const accfltrw_t tabgamma,const accfltrw_t Lk,const accfltrw_t Lk_1,const accfltrw_t Lk_2,flt dt);

flt CFL_kmul(u32 i,u32 nbins,u32 kflux,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,
                                    const accfltr_t tabipiflux,const accfltr_t tabK1,const accfltr_t tabK2,
                                    const accfltrw_t flux,const accfltrw_t tabdflux,const accfltrw_t tabdtCFL);
void Lk_func_kmul(u32 i,u32 nbins,u32 kflux,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,
                                        const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                                        const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t coeff_Leg,
                                        const accfltrw_t flux,const accfltrw_t intflux,const accfltrw_t Lk);
void solver_kmul(u32 i,u32 nbins, u32 kflux, u32 kp,const accfltr_t massgrid,const accfltr_t massbins,const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,
                                        const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                                        const accfltr_t tabK1,const accfltr_t tabK2,
                                        const accfltrw_t flux,const accfltrw_t intflux,const accfltrw_t coeff_Leg,const accfltrw_t coeff_gh,const accfltrw_t tabminvalgh,const accfltrw_t tabgamma,const accfltrw_t Lk,const accfltrw_t Lk_1,const accfltrw_t Lk_2,flt dt);

   

