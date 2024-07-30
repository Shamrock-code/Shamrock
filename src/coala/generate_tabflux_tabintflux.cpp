//-------------------------------------------
// Generate tabflux and tabintflux 
// Equations refere to Lombart & Laibe (2020) (doi:10.1093/mnras/staa3682)
//-------------------------------------------
#include "./kernel_approx/flux/flux_function.hpp"
#include "./kernel_approx/intflux/termA/termA_intflux_function.hpp"
#include "./kernel_approx/intflux/termB/termB_intflux_function.hpp"
#include <iostream>
#include "options.hpp"

//------------------------------
// Tabflux for k0 corresponding to term T in Eq.25 applied for conservative flux Eq.21
// kernel approx
// constant kernel
//------------------------------
void compute_tabflux_k0(u32 nbins,u32 kp,flt* massgrid,flt* massbins,flt* tabflux){
   flt res;
   for (u32 j=0;j<nbins;j++){
      for (u32 lp=0;lp<=j;lp++){
         for (u32 l=0;l<nbins;l++){
            for (u32 pp=0;pp<=kp;pp++){
               for (u32 p=0;p<=kp;p++){
                  res = fluxfunction(nbins,0,massgrid,massbins,j,lp,l,0,0,pp,p);

                  if (isnan(res)){
                     printf("for k=0, NAN in fluxfunction \n");
                     // exit(-1);
                  }
                  tabflux[p + pp*(kp+1) + l*(kp+1)*(kp+1) + lp*(kp+1)*(kp+1)*nbins + j*(kp+1)*(kp+1)*nbins*nbins] = ((flt)res);
               }
            }
         }
      }
   } 
}


//------------------------------
// Tabflux and tabintflux for k>0 corresponding to term T in Eq.25 and Eq.26 applied for conservative flux Eq.21
// kernel approx
// constant kernel which is split in 2 terms to generate 2 fluxes with tablfux1, tabflux2
//------------------------------
void compute_tabflux_tabintflux(u32 nbins,u32 kflux,u32 kp,flt* massgrid,flt* massbins,
                                 flt* tabipiflux,flt* tabipiAintflux,flt* tabipiBintflux){

   flt res;
   for (u32 j=0;j<nbins;j++){
      for (u32 lp=0;lp<=j;lp++){
         for (u32 l=0;l<nbins;l++){
            for (u32 ip=0;ip<=kflux;ip++){
               for (u32 i=0; i<=kflux;i++){
                  for (u32 pp=0;pp<=kp;pp++){
                     for (u32 p=0;p<=kp;p++){
                        //flux
                        res = fluxfunction(nbins,kflux,massgrid,massbins,j,lp,l,ip,i,pp,p);
                        if (isnan(res)){
                           printf("for k=1, NAN in tabipiflux \n");
                           // exit(-1);
                        }
                        tabipiflux[p + pp*(kp+1) + i*(kp+1)*(kp+1) + ip*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)] = ((flt)res);


                        //intflux terms
                        for (u32 k=0;k<=kflux;k++){
                           //termA
                           if (j>0 && lp<j){
                              //ipi
                              res = termA_intfluxfunction(nbins,kflux,massgrid,massbins,j,k,lp,l,ip,i,pp,p);
                              if (isnan(res)){
                                 printf("for k=1, NAN in tabipiAintflux \n");
                                 // exit(-1);
                              }
                              tabipiAintflux[p + pp*(kp+1) + k*(kp+1)*(kp+1) + i*(kflux+1)*(kp+1)*(kp+1) + ip*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)] = ((flt)res);

                           }


                           //termB
                           if (lp==0){
                              //ipi
                              res = termB_intfluxfunction(nbins,kflux,massgrid,massbins,j,k,l,ip,i,pp,p);
                              if (isnan(res)){
                                 printf("for k=1, NAN in tabipiBintflux \n");
                                 // exit(-1);
                              }
                              tabipiBintflux[p + pp*(kp+1) + k*(kp+1)*(kp+1) + i*(kflux+1)*(kp+1)*(kp+1) + ip*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) +  j*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)] = ((flt)res);
                              
                              
                           }
                        }
                     }
                  }
               }
            }            
         }
      }
   } 
}






