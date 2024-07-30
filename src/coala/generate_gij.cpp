//-------------------------------------------
// Generate gij vector for all particles
// dim = nparts*nbins*(kflux+1)
//-------------------------------------------

#include "options.hpp"
#include "L2proj_GQ.hpp"

void generate_gij(u32 nparts, u32 nbins, u32 kflux, u32 Q,flt* vecnodes, flt* vecweights, flt* massgrid, flt* massbins,flt* gij){

    flt* local_gij = new flt[nbins*(kflux+1)];
    L2proj_gij_GQ(nbins, kflux,massgrid,massbins,Q,vecnodes,vecweights,local_gij);


    for (u32 i=0;i<nparts;i++){
        for (u32 j=0;j<=nbins-1;j++){
            for (u32 k=0;k<=kflux;k++){
                gij[k+j*(kflux+1)+i*nbins*(kflux+1)] = ((flt)1 + ((flt)i)/((flt)nparts))*local_gij[k+j*(kflux+1)];

                if(abs(gij[k+j*(kflux+1)+i*nbins*(kflux+1)])< ((flt)1e-15)){
                    if (gij[k+j*(kflux+1)+i*nbins*(kflux+1)] < ((flt)0)){
                        gij[k+j*(kflux+1)+i*nbins*(kflux+1)] = -((flt)1e-15);
                    }else{
                        gij[k+j*(kflux+1)+i*nbins*(kflux+1)] = ((flt)1e-15);
                    }
                }
            }
        }

    }

    printf("gij start p1\n");
    for(u32 j=0;j<nbins;j++){
        for(u32 k=0;k<=kflux;k++){
             printf("%.4e, ",gij[k+j*(kflux+1)]);
        }
        printf("\n");
    }

    if (nparts>1){
        printf("gij start p%d \n",nparts);
        for(u32 j=0;j<nbins;j++){
            for(u32 k=0;k<=kflux;k++){
                printf("%.4e, ",gij[k+j*(kflux+1)+(nparts-1)*nbins*(kflux+1)]);
            }
            printf("\n");
        }
    } 
}