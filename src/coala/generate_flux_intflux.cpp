//-------------------------------------------
// Generate flux and intflux 
// Equations refere to Lombart & Laibe (2020) (doi:10.1093/mnras/staa3682)
//-------------------------------------------
#include "options.hpp"
#include <iostream>

//------------------------------
// compute flux k0 Eq.25 for conservative flux Eq.21
// kernel approx
// constant kernel
//------------------------------
void compute_flux_k0_kconst(u32 ipart,u32 nbins,u32 kp,const accfltrw_t gij,const accfltr_t tabflux,const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux){
   for (u32 j=0;j<nbins;j++){
      if (j==nbins-1){
         flux[j+ipart*nbins] = 0;
      }else{
         flt sumflux = ((flt)0);
         for (u32 lp=0;lp<=j;lp++){
            for (u32 l=0;l<nbins;l++){
               for (u32 pp=0;pp<=kp;pp++){
                  for (u32 p=0;p<=kp;p++){
                     sumflux += gij[lp+ipart*nbins]*gij[l+ipart*nbins]
                                 *(tabK1[pp+lp*(kp+1)]*tabK2[p+l*(kp+1)])
                                 *tabflux[p + pp*(kp+1) + l*(kp+1)*(kp+1) + lp*(kp+1)*(kp+1)*nbins + j*(kp+1)*(kp+1)*nbins*nbins];
                  }
               }

            }
         }
         flux[j+ipart*nbins] = sumflux;
      }
   }
}

//------------------------------
// compute flux k0 Eq.25 for conservative flux Eq.21
// kernel approx
// additive kernel
//------------------------------
void compute_flux_k0_kadd(u32 ipart,u32 nbins,u32 kp,const accfltrw_t gij,const accfltr_t tabflux,
                           const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,
                           const accfltrw_t flux){

   for (u32 j=0;j<nbins;j++){
      if (j==nbins-1){
         flux[j+ipart*nbins] = 0;
      }else{
         flt sumflux = ((flt)0);
         for (u32 lp=0;lp<=j;lp++){
            for (u32 l=0;l<nbins;l++){
               for (u32 pp=0;pp<=kp;pp++){
                  for (u32 p=0;p<=kp;p++){
                     sumflux += gij[lp+ipart*nbins]*gij[l+ipart*nbins]
                                 *(tabK1F1[pp+lp*(kp+1)]*tabK2F1[p+l*(kp+1)] + tabK1F2[pp+lp*(kp+1)]*tabK2F2[p+l*(kp+1)])
                                 *tabflux[p + pp*(kp+1) + l*(kp+1)*(kp+1) + lp*(kp+1)*(kp+1)*nbins + j*(kp+1)*(kp+1)*nbins*nbins];
                  }
               }
            }
         }
         flux[j+ipart*nbins] = sumflux;
      }
   }
}


//------------------------------
// compute flux k0 Eq.25 for conservative flux Eq.21
// kernel approx
// multiplicative kernel
//------------------------------
void compute_flux_k0_kmul(u32 ipart,u32 nbins,u32 kp,const accfltrw_t gij,const accfltr_t tabflux,const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux){
   for (u32 j=0;j<nbins;j++){
      if (j==nbins-1){
         flux[j+ipart*nbins] = 0;
      }else{
         flt sumflux = ((flt)0);
         for (u32 lp=0;lp<=j;lp++){
            for (u32 l=0;l<nbins;l++){
               for (u32 pp=0;pp<=kp;pp++){
                  for (u32 p=0;p<=kp;p++){
                     sumflux += gij[lp+ipart*nbins]*gij[l+ipart*nbins]
                                 *(tabK1[pp+lp*(kp+1)]*tabK2[p+l*(kp+1)])
                                 *tabflux[p + pp*(kp+1) + l*(kp+1)*(kp+1) + lp*(kp+1)*(kp+1)*nbins + j*(kp+1)*(kp+1)*nbins*nbins];
                  }
               }

            }
         }
         flux[j+ipart*nbins] = sumflux;
      }
   }
}




//------------------------------
// compute flux k>0 Eq.25 for conservative flux Eq.21
// kernel approx
// constant kernel
//------------------------------
void compute_flux_kconst(u32 ipart,u32 nbins,u32 kflux,u32 kp,const accfltrw_t gij,const accfltr_t tabipiflux,const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux){
   flt sumflux;

   for (u32 j=0;j<nbins;j++){
      if (j==nbins-1){
         flux[j+ipart*nbins] = 0;
      }else{
         sumflux = ((flt)0);

         for (u32 lp=0;lp<=j;lp++){
            for (u32 l=0;l<nbins;l++){
               for (u32 ip=0;ip<=kflux;ip++){
                  for (u32 i=0;i<=kflux;i++){
                     for (u32 pp=0;pp<=kp;pp++){
                        for (u32 p=0;p<=kp;p++){
                           sumflux += gij[ip+lp*(kflux+1)+ipart*nbins*(kflux+1)]*gij[i+l*(kflux+1)+ipart*nbins*(kflux+1)]
                                          *(tabK1[pp+lp*(kp+1)]*tabK2[p+l*(kp+1)])
                                          *tabipiflux[p + pp*(kp+1) + i*(kp+1)*(kp+1) + ip*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];
                        }
                     }
                  
                  }
               }
            }
         }
         flux[j+ipart*nbins] = sumflux;
      }
   }
}

//------------------------------
// compute flux k>0 Eq.25 for conservative flux Eq.21
// kernel approx
// additive kernel
//------------------------------
void compute_flux_kadd(u32 ipart,u32 nbins,u32 kflux,u32 kp,const accfltrw_t gij,
                        const accfltr_t tabipiflux,
                        const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,
                        const accfltrw_t flux){
   flt sumflux;

   for (u32 j=0;j<nbins;j++){
      if (j==nbins-1){
         flux[j+ipart*nbins] = 0;
      }else{
         sumflux = ((flt)0);

         for (u32 lp=0;lp<=j;lp++){
            for (u32 l=0;l<nbins;l++){
               for (u32 ip=0;ip<=kflux;ip++){
                  for (u32 i=0;i<=kflux;i++){
                     for (u32 pp=0;pp<=kp;pp++){
                        for (u32 p=0;p<=kp;p++){
                           sumflux += gij[ip+lp*(kflux+1)+ipart*nbins*(kflux+1)]*gij[i+l*(kflux+1)+ipart*nbins*(kflux+1)]
                                                *(tabK1F1[pp+lp*(kp+1)]*tabK2F1[p+l*(kp+1)] + tabK1F2[pp+lp*(kp+1)]*tabK2F2[p+l*(kp+1)])
                                                *tabipiflux[p + pp*(kp+1) + i*(kp+1)*(kp+1) + ip*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];
                        }
                     }
                  }
               }
            }
         }
         flux[j+ipart*nbins] = sumflux;
      }
   }
}

//------------------------------
// compute flux k>0 Eq.25 for conservative flux Eq.21
// kernel approx
// multiplicative kernel
//------------------------------
void compute_flux_kmul(u32 ipart,u32 nbins,u32 kflux,u32 kp,const accfltrw_t gij,
                        const accfltr_t tabipiflux,
                        const accfltr_t tabK1,const accfltr_t tabK2,
                        const accfltrw_t flux){
   flt sumflux;

   for (u32 j=0;j<nbins;j++){
      if (j==nbins-1){
         flux[j+ipart*nbins] = 0;
      }else{
         sumflux = ((flt)0);

         for (u32 lp=0;lp<=j;lp++){
            for (u32 l=0;l<nbins;l++){
               for (u32 ip=0;ip<=kflux;ip++){
                  for (u32 i=0;i<=kflux;i++){
                     for (u32 pp=0;pp<=kp;pp++){
                        for (u32 p=0;p<=kp;p++){
                           sumflux += gij[ip+lp*(kflux+1)+ipart*nbins*(kflux+1)]*gij[i+l*(kflux+1)+ipart*nbins*(kflux+1)]
                                          *(tabK1[pp+lp*(kp+1)]*tabK2[p+l*(kp+1)])
                                          *tabipiflux[p + pp*(kp+1) + i*(kp+1)*(kp+1) + ip*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];
                        }
                     }

                  }
               }
            }
         }
         flux[j+ipart*nbins] = sumflux;
      }
   }
}




//------------------------------
// compute flux and intflux k>0 Eq.25 and Eq.26 for conservative flux Eq.21
// kernel approx
// constant kernel
//------------------------------
void compute_flux_intflux_kconst(u32 ipart,u32 nbins,u32 kflux,u32 kp,const accfltrw_t gij,
                                 const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                                 const accfltr_t tabK1,const accfltr_t tabK2,
                                 const accfltrw_t flux,const accfltrw_t intflux){

   flt sumflux;flt sumAintflux;flt sumBintflux;

   for (u32 j=0;j<nbins;j++){
      sumflux = ((flt)0);

      for (u32 k=0;k<=kflux;k++){
         sumAintflux = ((flt)0);
         sumBintflux = ((flt)0);

         for (u32 lp=0;lp<=j;lp++){
            for (u32 l=0;l<nbins;l++){
               for (u32 ip=0;ip<=kflux;ip++){
                  for (u32 i=0;i<=kflux;i++){
                     for (u32 pp=0;pp<=kp;pp++){
                        for (u32 p=0;p<=kp;p++){

                           //flux
                           if (k==0){
                              sumflux += gij[ip+lp*(kflux+1)+ipart*nbins*(kflux+1)]*gij[i+l*(kflux+1)+ipart*nbins*(kflux+1)]
                                                *(tabK1[pp+lp*(kp+1)]*tabK2[p+l*(kp+1)])
                                                *tabipiflux[p + pp*(kp+1) + i*(kp+1)*(kp+1) + ip*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];
                           }
                  
                           //intflux
                           //termA
                           if (j>0 && lp<j){
                              sumAintflux += gij[ip+lp*(kflux+1)+ipart*nbins*(kflux+1)]*gij[i+l*(kflux+1)+ipart*nbins*(kflux+1)]
                                                *(tabK1[pp+lp*(kp+1)]*tabK2[p+l*(kp+1)])
                                                *tabipiAintflux[p + pp*(kp+1) + k*(kp+1)*(kp+1) + i*(kflux+1)*(kp+1)*(kp+1) + ip*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];
                           }

                           //termB
                           if (lp==0){
                              sumBintflux += gij[ip+j*(kflux+1)+ipart*nbins*(kflux+1)]*gij[i+l*(kflux+1)+ipart*nbins*(kflux+1)]
                                                *(tabK1[pp+j*(kp+1)]*tabK2[p+l*(kp+1)])
                                                *tabipiBintflux[p + pp*(kp+1) + k*(kp+1)*(kp+1) + i*(kflux+1)*(kp+1)*(kp+1) + ip*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) +  j*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];
                              
                              
                           }
                        }
                     }
                  }
               }
            }
         }

         intflux[k+j*(kflux+1)+ipart*nbins*(kflux+1)] = sumAintflux + sumBintflux;


      }
      if (j==nbins-1){
         flux[j+ipart*nbins]=((flt)0);
      }else{
         flux[j+ipart*nbins] = sumflux;
      }
   }

}


//------------------------------
// compute flux and intflux k>0 Eq.25 and Eq.26 for conservative flux Eq.21
// kernel approx
// additive kernel
//------------------------------
void compute_flux_intflux_kadd(u32 ipart,u32 nbins,u32 kflux,u32 kp,const accfltrw_t gij,
                              const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                              const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,
                              const accfltrw_t flux,const accfltrw_t intflux){

   flt sumflux;
   flt sumAintflux;
   flt sumBintflux;

   for (u32 j=0;j<nbins;j++){

      sumflux = ((flt)0);

      for (u32 k=0;k<=kflux;k++){
         sumAintflux = ((flt)0);
         sumBintflux = ((flt)0);

         for (u32 lp=0;lp<=j;lp++){
            for (u32 l=0;l<nbins;l++){
               for (u32 ip=0;ip<=kflux;ip++){
                  for (u32 i=0;i<=kflux;i++){
                     for (u32 pp=0;pp<=kp;pp++){
                        for (u32 p=0;p<=kp;p++){
                           //flux
                           if (k==0){
                              sumflux += gij[ip+lp*(kflux+1)+ipart*nbins*(kflux+1)]*gij[i+l*(kflux+1)+ipart*nbins*(kflux+1)]
                                                      *(tabK1F1[pp+lp*(kp+1)]*tabK2F1[p+l*(kp+1)] + tabK1F2[pp+lp*(kp+1)]*tabK2F2[p+l*(kp+1)])
                                                      *tabipiflux[p + pp*(kp+1) + i*(kp+1)*(kp+1) + ip*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];

                           }
                        
                           //intflux
                           //termA
                           if (j>0 && lp<j){
                              sumAintflux += gij[ip+lp*(kflux+1)+ipart*nbins*(kflux+1)]*gij[i+l*(kflux+1)+ipart*nbins*(kflux+1)]
                                                      *(tabK1F1[pp+lp*(kp+1)]*tabK2F1[p+l*(kp+1)] + tabK1F2[pp+lp*(kp+1)]*tabK2F2[p+l*(kp+1)])
                                                      *tabipiAintflux[p + pp*(kp+1) + k*(kp+1)*(kp+1) + i*(kflux+1)*(kp+1)*(kp+1) + ip*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];
                           }

                           //termB
                           if (lp==0){
                              sumBintflux += gij[ip+j*(kflux+1)+ipart*nbins*(kflux+1)]*gij[i+l*(kflux+1)+ipart*nbins*(kflux+1)]
                                                      *(tabK1F1[pp+j*(kp+1)]*tabK2F1[p+l*(kp+1)] + tabK1F2[pp+j*(kp+1)]*tabK2F2[p+l*(kp+1)])
                                                      *tabipiBintflux[p + pp*(kp+1) + k*(kp+1)*(kp+1) + i*(kflux+1)*(kp+1)*(kp+1) + ip*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) +  j*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];
                           }
                        }
                     }
                  }
               }
            }
         }

         intflux[k+j*(kflux+1)+ipart*nbins*(kflux+1)] = sumAintflux + sumBintflux;
         

      }
      if (j==nbins-1){
         flux[j+ipart*nbins] = ((flt)0);
      }else{
         flux[j+ipart*nbins] = sumflux;
      }

      
   }
}


//------------------------------
// compute flux and intflux k>0 Eq.25 and Eq.26 for conservative flux Eq.21
// kernel approx
// multiplicative kernel
//------------------------------
void compute_flux_intflux_kmul(u32 ipart,u32 nbins,u32 kflux,u32 kp,const accfltrw_t gij,
                                 const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                                 const accfltr_t tabK1,const accfltr_t tabK2,
                                 const accfltrw_t flux,const accfltrw_t intflux){

   flt sumflux;
   flt sumAintflux;
   flt sumBintflux;

   for (u32 j=0;j<nbins;j++){
      sumflux = ((flt)0);
      for (u32 k=0;k<=kflux;k++){
         sumAintflux = ((flt)0);
         sumBintflux = ((flt)0);

         for (u32 lp=0;lp<=j;lp++){
            for (u32 l=0;l<nbins;l++){
               for (u32 ip=0;ip<=kflux;ip++){
                  for (u32 i=0;i<=kflux;i++){
                     for (u32 pp=0;pp<=kp;pp++){
                        for (u32 p=0;p<=kp;p++){

                           //flux
                           if (k==0){
                              sumflux += gij[ip+lp*(kflux+1)+ipart*nbins*(kflux+1)]*gij[i+l*(kflux+1)+ipart*nbins*(kflux+1)]
                                                *(tabK1[pp+lp*(kp+1)]*tabK2[p+l*(kp+1)])
                                                *tabipiflux[p + pp*(kp+1) + i*(kp+1)*(kp+1) + ip*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];
                           }
                  
                           //intflux
                           //termA
                           if (j>0 && lp<j){
                              sumAintflux += gij[ip+lp*(kflux+1)+ipart*nbins*(kflux+1)]*gij[i+l*(kflux+1)+ipart*nbins*(kflux+1)]
                                                *(tabK1[pp+lp*(kp+1)]*tabK2[p+l*(kp+1)])
                                                *tabipiAintflux[p + pp*(kp+1) + k*(kp+1)*(kp+1) + i*(kflux+1)*(kp+1)*(kp+1) + ip*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];
                           }

                           //termB
                           if (lp==0){
                              sumBintflux += gij[ip+j*(kflux+1)+ipart*nbins*(kflux+1)]*gij[i+l*(kflux+1)+ipart*nbins*(kflux+1)]
                                                *(tabK1[pp+j*(kp+1)]*tabK2[p+l*(kp+1)])
                                                *tabipiBintflux[p + pp*(kp+1) + k*(kp+1)*(kp+1) + i*(kflux+1)*(kp+1)*(kp+1) + ip*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) +  j*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];
                              
                              
                           }
                        }
                     }
                  }
                     

                  
               }
            }
         }

         intflux[k+j*(kflux+1)+ipart*nbins*(kflux+1)] = sumAintflux + sumBintflux;

         

      }
      if (j==nbins-1){
         flux[j+ipart*nbins] = ((flt)0);
      }else{
         flux[j+ipart*nbins] = sumflux;
      }
      
   }
}





