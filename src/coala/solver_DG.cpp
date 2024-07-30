//-------------------------------------------
// Functions for time solver DG scheme 
// Equations refere to Lombart & Laibe (2020) (doi:10.1093/mnras/staa3682)
//-------------------------------------------
#include "options.hpp"
#include "generate_flux_intflux.hpp"
#include "polynomials_legendre.hpp"
#include "limiter.hpp"
#include "solver_DG.hpp"
#include <iostream>
#include "options.hpp"



//-------------------------------
// Function for the CFL condition for k=0 in Eq.33
// for constant kernel
//-------------------------------
flt CFL_k0_kconst(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltr_t tabflux, const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux, const accfltrw_t tabdflux, const accfltrw_t tabdtCFL){

    //compute flux vector
    compute_flux_k0_kconst(i,nbins,kp,gij,tabflux,tabK1,tabK2,flux);
    // for (u32 j=0;j<nbins;j++){
    //     printf("j=%d, flux=%.15e\n",j,flux[j]);
    // }
    // exit(-1);

    //CFL condition
    tabdflux[0+i*nbins] = flux[0+i*nbins];
    tabdtCFL[0+i*nbins] = fabs(gij[0+i*nbins]*(massgrid[1]-massgrid[0])/(tabdflux[0+i*nbins]));


    for (u32 j=1;j<nbins;j++){
      //derivative of flux at order kflux (1st order derivative discretisation)
      tabdflux[j+i*nbins] = flux[j+i*nbins]-flux[j-1 + i*nbins];
      tabdtCFL[j+i*nbins] = fabs(gij[j+i*nbins]*(massgrid[j+1]-massgrid[j])/(tabdflux[j+i*nbins]));
    }

    flt dtCFL = tabdtCFL[0+i*nbins];
    for(u32 i = 1; i < nbins; i ++){
    //    printf("i=%d, tabdtCFL=%.15e \n",i,tabdtCFL[i]);
       if(tabdtCFL[i] > 0){
            dtCFL = fmin(dtCFL,tabdtCFL[i]);
       }
    }

    return dtCFL;
   
}

//-------------------------
// Function L for time solver k=0 Eq.16
// for constant kernel
//-------------------------
void Lk0_func_kconst(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltr_t tabflux, const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux,const accfltrw_t Lk0){
	//compute flux vector
    compute_flux_k0_kconst(i,nbins,kp,gij,tabflux,tabK1,tabK2,flux);
		
    //compute L term for DG scheme
	Lk0[0+i*nbins] = -flux[0+i*nbins]/(massgrid[1]-massgrid[0]);
	for (u32 j=1;j<=nbins-1;j++){
	  Lk0[j+i*nbins] = (- (flux[j+i*nbins] - flux[j-1+i*nbins]))/(massgrid[j+1]-massgrid[j]);
	}
    
}

//--------------------
// Time solver SSP-RK3 k=0 Eq.38
// for constant kernel
//--------------------
void solver_k0_kconst(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,const accfltr_t tabflux,  const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux,const accfltrw_t Lk0,const accfltrw_t Lk0_1,const accfltrw_t Lk0_2, flt dt){

    //time solver
    //step 1
    Lk0_func_kconst(i,nbins,kp,massgrid,gij,tabflux,tabK1,tabK2,flux,Lk0);
    for (u32 j=0;j<=nbins-1;j++){
        gij1[j+i*nbins] = gij[j+i*nbins] + dt*Lk0[j+i*nbins];
    }
    
    //step 2
    Lk0_func_kconst(i,nbins,kp,massgrid,gij1,tabflux,tabK1,tabK2,flux,Lk0_1);
    for (u32 j=0;j<=nbins-1;j++){
        gij2[j+i*nbins] = ((flt)3)*gij[j+i*nbins]/((flt)4) + (gij1[j+i*nbins] + dt*Lk0_1[j+i*nbins])/((flt)4);
    }
    
    //step 3
    Lk0_func_kconst(i,nbins,kp,massgrid,gij2,tabflux,tabK1,tabK2,flux,Lk0_2);
    for (u32 j=0;j<=nbins-1;j++){
        gij[j+i*nbins] = gij[j+i*nbins]/((flt)3) + ((flt)2)*(gij2[j+i*nbins] + dt*Lk0_2[j+i*nbins])/((flt)3);
    }
  
}


//-------------------------------
// Function for the CFL condition for k=0 in Eq.33
// for additive kernel
//-------------------------------
flt CFL_k0_kadd(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltr_t tabflux,const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,const accfltrw_t flux,const accfltrw_t tabdflux,const accfltrw_t tabdtCFL){    
    //copmute flux vector
    compute_flux_k0_kadd(i,nbins,kp,gij,tabflux,tabK1F1,tabK2F1,tabK1F2,tabK2F2,flux);
    // for (u32 j=0;j<nbins;j++){
    //     printf("j=%d, flux=%.15e\n",j,flux[j]);
    // }
    // exit(-1);

    //CFL condition
    tabdflux[0+i*nbins] = flux[0+i*nbins];
    tabdtCFL[0+i*nbins] = fabs(gij[0+i*nbins]*(massgrid[1]-massgrid[0])/(tabdflux[0+i*nbins]));


    for (u32 j=1;j<nbins;j++){
      //derivative of flux at order kflux (1st order derivative discretisation)
      tabdflux[j+i*nbins] = flux[j+i*nbins]-flux[j-1+i*nbins];
      tabdtCFL[j+i*nbins] = fabs(gij[j+i*nbins]*(massgrid[j+1]-massgrid[j])/(tabdflux[j+i*nbins]));
    }

    flt dtCFL = tabdtCFL[0+i*nbins];
    for(u32 j = 1; j < nbins; j ++){
    //    printf("i=%d, tabdtCFL=%.15e \n",i,tabdtCFL[i]);
       if(tabdtCFL[j+i*nbins] > 0){
            dtCFL = fmin(dtCFL,tabdtCFL[j+i*nbins]);
       }
    }

    return dtCFL;
   
}

//-------------------------
// Function L for time solver k=0 Eq.16
// for additive kernel
//-------------------------
void Lk0_func_kadd(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltr_t tabflux,const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,const accfltrw_t flux,const accfltrw_t Lk0){
	//compute flux vector
    compute_flux_k0_kadd(i,nbins,kp,gij,tabflux,tabK1F1,tabK2F1,tabK1F2,tabK2F2,flux);
		
    //compute L function for DG scheme
	Lk0[0+i*nbins] = -flux[0+i*nbins]/(massgrid[1]-massgrid[0]);
	for (u32 j=1;j<=nbins-1;j++){
	  Lk0[j+i*nbins] = (- (flux[j+i*nbins] - flux[j-1+i*nbins]))/(massgrid[j+1]-massgrid[j]);
	}
    
}

//--------------------
// Time solver SSP-RK3 k=0 Eq.38
// for additive kernel
//--------------------
void solver_k0_kadd(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,const accfltr_t tabflux,const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,const accfltrw_t flux,const accfltrw_t Lk0,const accfltrw_t Lk0_1,const accfltrw_t Lk0_2, flt dt){

    //time solver
    //step 1
    Lk0_func_kadd(i,nbins,kp,massgrid,gij,tabflux,tabK1F1,tabK2F1,tabK1F2,tabK2F2,flux,Lk0);
    for (u32 j=0;j<=nbins-1;j++){
        gij1[j+i*nbins] = gij[j+i*nbins] + dt*Lk0[j+i*nbins];
    }
    
    //step 2
    Lk0_func_kadd(i,nbins,kp,massgrid,gij1,tabflux,tabK1F1,tabK2F1,tabK1F2,tabK2F2,flux,Lk0_1);
    for (u32 j=0;j<=nbins-1;j++){
        gij2[j+i*nbins] = ((flt)3)*gij[j+i*nbins]/((flt)4) + (gij1[j+i*nbins] + dt*Lk0_1[j+i*nbins])/((flt)4);
    }
    
    //step 3
    Lk0_func_kadd(i,nbins,kp,massgrid,gij2,tabflux,tabK1F1,tabK2F1,tabK1F2,tabK2F2,flux,Lk0_2);
    for (u32 j=0;j<=nbins-1;j++){
        gij[j+i*nbins] = gij[j+i*nbins]/((flt)3) + ((flt)2)*(gij2[j+i*nbins] + dt*Lk0_2[j+i*nbins])/((flt)3);
    }
    
}


//-------------------------------
// Function for the CFL condition for k=0 in Eq.33
// for multiplicative kernel
//-------------------------------
flt CFL_k0_kmul(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltr_t tabflux,const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux,const accfltrw_t tabdflux,const accfltrw_t tabdtCFL){    
    //compute flux vector
    compute_flux_k0_kmul(i,nbins,kp,gij,tabflux,tabK1,tabK2,flux);

    //CFL condition
    tabdflux[0+i*nbins] = flux[0+i*nbins];
    tabdtCFL[0+i*nbins] = fabs(gij[0+i*nbins]*(massgrid[1]-massgrid[0])/(tabdflux[0+i*nbins]));


    for (u32 j=1;j<nbins;j++){
      //derivative of flux at order kflux (1st order derivative discretisation)
      tabdflux[j+i*nbins] = flux[j+i*nbins]-flux[j-1+i*nbins];
      tabdtCFL[j+i*nbins] = fabs(gij[j+i*nbins]*(massgrid[j+1]-massgrid[j])/(tabdflux[j+i*nbins]));
    }

    flt dtCFL = tabdtCFL[0+i*nbins];
    for(u32 j = 1; j < nbins; j ++){
    //    printf("i=%d, tabdtCFL=%.15e \n",i,tabdtCFL[i]);
       if(tabdtCFL[j+i*nbins] > 0){
            dtCFL = fmin(dtCFL,tabdtCFL[j+i*nbins]);
       }
    }

   return dtCFL;
   
}

//-------------------------
// Function L for time solver k=0 Eq.16
// for multiplicative kernel
//-------------------------
void Lk0_func_kmul(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltr_t tabflux,const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux,const accfltrw_t Lk0){
	//compute flux vector
    compute_flux_k0_kmul(i,nbins,kp,gij,tabflux,tabK1,tabK2,flux);
	
    //compute L term for DG scheme	
	Lk0[0+i*nbins] = -flux[0+i*nbins]/(massgrid[1]-massgrid[0]);
	for (u32 j=1;j<=nbins-1;j++){
	  Lk0[j+i*nbins] = (- (flux[j+i*nbins] - flux[j-1+i*nbins]))/(massgrid[j+1]-massgrid[j]);
	}
    
}

//--------------------
// Time solver SSP-RK3 k=0 Eq.38
// for multiplicative kernel
//--------------------
void solver_k0_kmul(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,const accfltr_t tabflux,const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux,const accfltrw_t Lk0,const accfltrw_t Lk0_1,const accfltrw_t Lk0_2, flt dt){

    //time solver
    //step 1
    Lk0_func_kmul(i,nbins,kp,massgrid,gij,tabflux,tabK1,tabK2,flux,Lk0);
    for (u32 j=0;j<=nbins-1;j++){
        gij1[j+i*nbins] = gij[j+i*nbins] + dt*Lk0[j+i*nbins];
    }
    
    //step 2
    Lk0_func_kmul(i,nbins,kp,massgrid,gij1,tabflux,tabK1,tabK2,flux,Lk0_1);
    for (u32 j=0;j<=nbins-1;j++){
        gij2[j+i*nbins] = ((flt)3)*gij[j+i*nbins]/((flt)4) + (gij1[j+i*nbins] + dt*Lk0_1[j+i*nbins])/((flt)4);
    }
    
    //step 3
    Lk0_func_kmul(i,nbins,kp,massgrid,gij2,tabflux,tabK1,tabK2,flux,Lk0_2);
    for (u32 j=0;j<=nbins-1;j++){
        gij[j+i*nbins] = gij[j+i*nbins]/((flt)3) + ((flt)2)*(gij2[j+i*nbins] + dt*Lk0_2[j+i*nbins])/((flt)3);
    }


    
}




//-------------------------------
// Function for the CFL condition for k>0 in Eq.33
// for constant kernel
//-------------------------------
flt CFL_kconst(u32 i,u32 nbins,u32 kflux,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,
                const accfltr_t tabipiflux,const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t flux,const accfltrw_t tabdflux,const accfltrw_t tabdtCFL){
    
    //compute flux vector
    compute_flux_kconst(i,nbins,kflux,kp,gij,tabipiflux,tabK1,tabK2,flux);
    // for (u32 j=0;j<nbins;j++){
    //     printf("j=%d, flux=%.15e\n",j,flux[j]);
    // }
    // exit(-1);

    //CFL condition
    tabdflux[0+i*nbins] = flux[0+i*nbins];
    tabdtCFL[0+i*nbins] = fabs(gij[0+0*(kflux+1)+i*nbins*(kflux+1)]*(massgrid[1]-massgrid[0])/(tabdflux[0+i*nbins]));


    for (u32 j=1;j<nbins;j++){
      //derivative of flux at order kflux (1st order derivative discretisation)
      tabdflux[j+i*nbins] = flux[j+i*nbins]-flux[j-1+i*nbins];
      tabdtCFL[j+i*nbins] = fabs(gij[0+j*(kflux+1)+i*nbins*(kflux+1)]*(massgrid[j+1]-massgrid[j])/(tabdflux[j+i*nbins]));
    }

    flt dtCFL = tabdtCFL[0];
    for(u32 j = 1; j < nbins; j ++){
    //    printf("i=%d, tabdtCFL=%.15e \n",i,tabdtCFL[i]);
       if(tabdtCFL[j+i*nbins] > 0){
            dtCFL = fmin(dtCFL,tabdtCFL[j+i*nbins]);
       }
    }

    return dtCFL;
   
}


//-------------------------
// Function L for time solver k>0 Eq.16
// for constant kernel
//-------------------------
void Lk_func_kconst(u32 i,u32 nbins,u32 kflux,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,
                    const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                    const accfltr_t tabK1,const accfltr_t tabK2,
                    const accfltrw_t coeff_Leg,
                    const accfltrw_t flux,const accfltrw_t intflux,const accfltrw_t Lk){

    //compute flux and intflux vectors
    compute_flux_intflux_kconst(i,nbins,kflux,kp,gij,
                                    tabipiflux,tabipiAintflux,tabipiBintflux,
                                    tabK1,tabK2,
                                    flux,intflux);

    // for (u32 j=0;j<nbins;j++){
    //     printf("j=%d, flux=%.15e\n",j,flux[j]);
    // }

    // for (u32 j=0;j<nbins;j++){
    //     printf("j=%d, intflux=",j);
    //     for (u32 k=0;k<=kflux;k++){
    //         printf("%.15e  ",intflux[k+j*(kflux+1)+i*nbins*(kflux+1)]);

    //     }
    //     printf("\n");
    // }
    // exit(-1);
		

    //compute L term for DG scheme
    flt hj; flt c; flt LegPleft; flt LegPright;
	for (u32 j=0;j<=nbins-1;j++){
        hj = massgrid[j+1]-massgrid[j];
        for (u32 k=0;k<=kflux;k++){
            c = coeffnorm(k);
            LegPleft  = Leg_P_sycl(i,k,coeff_Leg,-((flt)1));
            LegPright = Leg_P_sycl(i,k,coeff_Leg,((flt)1));

            if (j==0){
                Lk[k+j*(kflux+1)+i*nbins*(kflux+1)] = (((flt)2)*(intflux[k+j*(kflux+1)+i*nbins*(kflux+1)]-(flux[j+i*nbins]*LegPright)))/(c*hj);
            }else{
                Lk[k+j*(kflux+1)+i*nbins*(kflux+1)] = (((flt)2)*(intflux[k+j*(kflux+1)+i*nbins*(kflux+1)]-(flux[j+i*nbins]*LegPright - flux[j-1+i*nbins]*LegPleft)))/(c*hj);
            }
        }
	}
    
}

//--------------------
// Time solver SSP-RK3 k=0 Eq.38
// for additive kernel
//--------------------
void solver_kconst(u32 i,u32 nbins, u32 kflux,u32 kp, const accfltr_t massgrid,const accfltr_t massbins,const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,
                const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                const accfltr_t tabK1,const accfltr_t tabK2,
                const accfltrw_t flux, const accfltrw_t intflux, const accfltrw_t coeff_Leg,const accfltrw_t coeff_gh,const accfltrw_t tabminvalgh,const accfltrw_t tabgamma,const accfltrw_t Lk,const accfltrw_t Lk_1,const accfltrw_t Lk_2,flt dt){

    //time solver + scaling limiter for each step
    //step 1
    Lk_func_kconst(i,nbins,kflux,kp,massgrid,gij,
                    tabipiflux,tabipiAintflux,tabipiBintflux,
                    tabK1,tabK2,coeff_Leg,
                    flux,intflux,Lk);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=0;k<=kflux;k++){
            gij1[k+j*(kflux+1)+i*nbins*(kflux+1)] = gij[k+j*(kflux+1)+i*nbins*(kflux+1)] + dt*Lk[k+j*(kflux+1)+i*nbins*(kflux+1)];
        }
    }

    //apply scaling limiter
    gammafunction(i,nbins,kflux,massgrid,massbins,gij1,coeff_Leg,coeff_gh,tabminvalgh,tabgamma);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=1;k<=kflux;k++){
            gij1[k+j*(kflux+1)+i*nbins*(kflux+1)] = tabgamma[j+i*nbins]*gij1[k+j*(kflux+1)+i*nbins*(kflux+1)];
        }
    }

    
    //step 2
    Lk_func_kconst(i,nbins,kflux,kp,massgrid,gij1,
                    tabipiflux,tabipiAintflux,tabipiBintflux,
                    tabK1,tabK2,coeff_Leg,
                    flux,intflux,Lk_1);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=0;k<=kflux;k++){
            gij2[k+j*(kflux+1)+i*nbins*(kflux+1)] = ((flt)3)*gij[k+j*(kflux+1)+i*nbins*(kflux+1)]/((flt)4) + (gij1[k+j*(kflux+1)+i*nbins*(kflux+1)] + dt*Lk_1[k+j*(kflux+1)+i*nbins*(kflux+1)])/((flt)4);
        }
    }

    //apply scaling limiter
    gammafunction(i,nbins,kflux,massgrid,massbins,gij2,coeff_Leg,coeff_gh,tabminvalgh,tabgamma);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=1;k<=kflux;k++){
            gij2[k+j*(kflux+1)+i*nbins*(kflux+1)] = tabgamma[j+i*nbins]*gij2[k+j*(kflux+1)+i*nbins*(kflux+1)];
        }
    }
    
    //step 3
    Lk_func_kconst(i,nbins,kflux,kp,massgrid,gij2,
                    tabipiflux,tabipiAintflux,tabipiBintflux,
                    tabK1,tabK2,coeff_Leg,
                    flux,intflux,Lk_2);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=0;k<=kflux;k++){
            gij[k+j*(kflux+1)+i*nbins*(kflux+1)] = gij[k+j*(kflux+1)+i*nbins*(kflux+1)]/((flt)3) + ((flt)2)*(gij2[k+j*(kflux+1)+i*nbins*(kflux+1)] + dt*Lk_2[k+j*(kflux+1)+i*nbins*(kflux+1)])/((flt)3);
        }
    }

    //apply scaling limiter
    gammafunction(i,nbins,kflux,massgrid,massbins,gij,coeff_Leg,coeff_gh,tabminvalgh,tabgamma);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=1;k<=kflux;k++){
            gij[k+j*(kflux+1)+i*nbins*(kflux+1)] = tabgamma[j+i*nbins]*gij[k+j*(kflux+1)+i*nbins*(kflux+1)];
        }
    }
    
}



//-------------------------------
// Function for the CFL condition for k>0 in Eq.33
// for additive kernel
//-------------------------------
flt CFL_kadd(u32 i,u32 nbins,u32 kflux,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,
            const accfltr_t tabipiflux,const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,
            const accfltrw_t flux,const accfltrw_t tabdflux,const accfltrw_t tabdtCFL){

    //copmute flux vector
    compute_flux_kadd(i,nbins,kflux,kp,gij,tabipiflux,tabK1F1,tabK2F2,tabK1F2,tabK2F2,flux);
    // for (u32 j=0;j<nbins;j++){
    //     printf("j=%d, flux=%.15e\n",j,flux[j]);
    // }
    // exit(-1);

    //CFL condition
    tabdflux[0+i*nbins] = flux[0+i*nbins];
    tabdtCFL[0+i*nbins] = fabs(gij[0+0*(kflux+1)+i*nbins*(kflux+1)]*(massgrid[1]-massgrid[0])/(tabdflux[0+i*nbins]));


    for (u32 j=1;j<nbins;j++){
      //derivative of flux at order kflux (1st order derivative discretisation)
      tabdflux[j+i*nbins] = flux[j+i*nbins]-flux[j-1+i*nbins];
      tabdtCFL[j+i*nbins] = fabs(gij[0+j*(kflux+1)+i*nbins*(kflux+1)]*(massgrid[j+1]-massgrid[j])/(tabdflux[j+i*nbins]));
    }

    flt dtCFL = tabdtCFL[0];
    for(u32 j = 1; j < nbins; j ++){
    //    printf("i=%d, tabdtCFL=%.15e \n",i,tabdtCFL[i]);
    //    if(tabdtCFL[j+i*nbins] > 0){
    //         dtCFL = fmin(dtCFL,tabdtCFL[j+i*nbins]);
    //    }
    }

    return dtCFL;
   
}


//-------------------------
// Function L for time solver k>0 Eq.16
// for additive kernel
//-------------------------
void Lk_func_kadd(u32 i,u32 nbins,u32 kflux,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,
                const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,
                const accfltrw_t coeff_Leg,
                const accfltrw_t flux,const accfltrw_t intflux,const accfltrw_t Lk){

    //compute flux and intflux vectors
    compute_flux_intflux_kadd(i,nbins,kflux,kp,gij,
                              tabipiflux,tabipiAintflux,tabipiBintflux,
                              tabK1F1,tabK2F2,tabK1F2,tabK2F2,
                              flux,intflux);

    // for (u32 j=0;j<nbins;j++){
    //     printf("j=%d, flux=%.15e\n",j,flux[j]);
    // }
    // exit(-1);
		

    //compute L term for DG scheme
    flt hj; flt c; flt LegPleft; flt LegPright;
	for (u32 j=0;j<=nbins-1;j++){
        hj = massgrid[j+1]-massgrid[j];
        for (u32 k=0;k<=kflux;k++){
            c = coeffnorm(k);
            LegPleft  = Leg_P_sycl(i,k,coeff_Leg,-((flt)1));
            LegPright = Leg_P_sycl(i,k,coeff_Leg,((flt)1));

            if (j==0){
                Lk[k+j*(kflux+1)+i*nbins*(kflux+1)] = (((flt)2)*(intflux[k+j*(kflux+1)+i*nbins*(kflux+1)]-(flux[j+i*nbins]*LegPright)))/(c*hj);
            }else{
                Lk[k+j*(kflux+1)+i*nbins*(kflux+1)] = (((flt)2)*(intflux[k+j*(kflux+1)+i*nbins*(kflux+1)]-(flux[j+i*nbins]*LegPright - flux[j-1+i*nbins]*LegPleft)))/(c*hj);
            }
        }
	}
    
}

//--------------------
// Time solver SSP-RK3 k=0 Eq.38
// for additive kernel
//--------------------
void solver_kadd(u32 i,u32 nbins, u32 kflux, u32 kp,const accfltr_t massgrid,const accfltr_t massbins,const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,
                const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,
                const accfltrw_t flux,const accfltrw_t intflux,const accfltrw_t coeff_Leg,const accfltrw_t coeff_gh,const accfltrw_t tabminvalgh,const accfltrw_t tabgamma,const accfltrw_t Lk,const accfltrw_t Lk_1,const accfltrw_t Lk_2,flt dt){

    //time solver + scaling limiter for each step
    //step 1
    Lk_func_kadd(i,nbins,kflux,kp,massgrid,gij,
                tabipiflux,tabipiAintflux,tabipiBintflux,
                tabK1F1,tabK2F2,tabK1F2,tabK2F2,
                coeff_Leg,
                flux,intflux,Lk);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=0;k<=kflux;k++){
            gij1[k+j*(kflux+1)+i*nbins*(kflux+1)] = gij[k+j*(kflux+1)+i*nbins*(kflux+1)] + dt*Lk[k+j*(kflux+1)+i*nbins*(kflux+1)];
        }
    }

    //apply scaling limiter
    gammafunction(i,nbins,kflux,massgrid,massbins,gij1,coeff_Leg,coeff_gh,tabminvalgh,tabgamma);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=1;k<=kflux;k++){
            gij1[k+j*(kflux+1)+i*nbins*(kflux+1)] = tabgamma[j+i*nbins]*gij1[k+j*(kflux+1)+i*nbins*(kflux+1)];
        }
    }
    
    //step 2
    Lk_func_kadd(i,nbins,kflux,kp,massgrid,gij1,
                tabipiflux,tabipiAintflux,tabipiBintflux,
                tabK1F1,tabK2F2,tabK1F2,tabK2F2,
                coeff_Leg,
                flux,intflux,Lk_1);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=0;k<=kflux;k++){
            gij2[k+j*(kflux+1)+i*nbins*(kflux+1)] = ((flt)3)*gij[k+j*(kflux+1)+i*nbins*(kflux+1)]/((flt)4) + (gij1[k+j*(kflux+1)+i*nbins*(kflux+1)] + dt*Lk_1[k+j*(kflux+1)+i*nbins*(kflux+1)])/((flt)4);
        }
    }

    //apply scaling limiter
    gammafunction(i,nbins,kflux,massgrid,massbins,gij2,coeff_Leg,coeff_gh,tabminvalgh,tabgamma);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=1;k<=kflux;k++){
            gij2[k+j*(kflux+1)+i*nbins*(kflux+1)] = tabgamma[j+i*nbins]*gij2[k+j*(kflux+1)+i*nbins*(kflux+1)];
        }
    }
    
    //step 3
    Lk_func_kadd(i,nbins,kflux,kp,massgrid,gij2,
                tabipiflux,tabipiAintflux,tabipiBintflux,
                tabK1F1,tabK2F2,tabK1F2,tabK2F2,
                coeff_Leg,
                flux,intflux,Lk_2);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=0;k<=kflux;k++){
            gij[k+j*(kflux+1)+i*nbins*(kflux+1)] = gij[k+j*(kflux+1)+i*nbins*(kflux+1)]/((flt)3) + ((flt)2)*(gij2[k+j*(kflux+1)+i*nbins*(kflux+1)] + dt*Lk_2[k+j*(kflux+1)+i*nbins*(kflux+1)])/((flt)3);
        }
    }

    //apply scaling limiter
    gammafunction(i,nbins,kflux,massgrid,massbins,gij,coeff_Leg,coeff_gh,tabminvalgh,tabgamma);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=1;k<=kflux;k++){
            gij[k+j*(kflux+1)+i*nbins*(kflux+1)] = tabgamma[j+i*nbins]*gij[k+j*(kflux+1)+i*nbins*(kflux+1)];
        }
    }
    
}


//-------------------------------
// Function for the CFL condition for k>0 in Eq.33
// for multiplicative kernel
//-------------------------------
flt CFL_kmul(u32 i,u32 nbins,u32 kflux,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,
            const accfltr_t tabipiflux,const accfltr_t tabK1,const accfltr_t tabK2,
            const accfltrw_t flux,const accfltrw_t tabdflux,const accfltrw_t tabdtCFL){
    
    //compute flux vector
    compute_flux_kmul(i,nbins,kflux,kp,gij,tabipiflux,tabK1,tabK2,flux);
    // for (u32 j=0;j<nbins;j++){
    //     printf("j=%d, flux=%.15e\n",j,flux[j]);
    // }
    // exit(-1);

    //CFL condition
    tabdflux[0+i*nbins] = flux[0+i*nbins];
    tabdtCFL[0+i*nbins] = fabs(gij[0+0*(kflux+1)+i*nbins*(kflux+1)]*(massgrid[1]-massgrid[0])/(tabdflux[0+i*nbins]));


    for (u32 j=1;j<nbins;j++){
      //derivative of flux at order kflux (1st order derivative discretisation)
      tabdflux[j+i*nbins] = flux[j+i*nbins]-flux[j-1+i*nbins];
      tabdtCFL[j+i*nbins] = fabs(gij[0+j*(kflux+1)+i*nbins*(kflux+1)]*(massgrid[j+1]-massgrid[j])/(tabdflux[j+i*nbins]));
    }

    flt dtCFL = tabdtCFL[0];
    for(u32 j = 1; j < nbins; j ++){
    //    printf("i=%d, tabdtCFL=%.15e \n",i,tabdtCFL[i]);
       if(tabdtCFL[j+i*nbins] > 0){
            dtCFL = fmin(dtCFL,tabdtCFL[j+i*nbins]);
       }
    }

    return dtCFL;
   
}


//-------------------------
// Function L for time solver k>0 Eq.16
// for multiplicative kernel
//-------------------------
void Lk_func_kmul(u32 i,u32 nbins,u32 kflux,u32 kp,const accfltr_t massgrid,const accfltrw_t gij,
                    const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                    const accfltr_t tabK1,const accfltr_t tabK2,const accfltrw_t coeff_Leg,
                    const accfltrw_t flux,const accfltrw_t intflux,const accfltrw_t Lk){

    //compute flux and intflux vectors
    compute_flux_intflux_kmul(i,nbins,kflux,kp,gij,
                              tabipiflux,tabipiAintflux,tabipiBintflux,
                              tabK1,tabK2,
                              flux,intflux);


    //copmute L function for DG scheme
    flt hj; flt c; flt LegPleft; flt LegPright;
	for (u32 j=0;j<=nbins-1;j++){
        hj = massgrid[j+1]-massgrid[j];
        for (u32 k=0;k<=kflux;k++){
            c = coeffnorm(k);
            LegPleft  = Leg_P_sycl(i,k,coeff_Leg,-((flt)1));
            LegPright = Leg_P_sycl(i,k,coeff_Leg,((flt)1));

            if (j==0){
                Lk[k+j*(kflux+1)+i*nbins*(kflux+1)] = (((flt)2)*(intflux[k+j*(kflux+1)+i*nbins*(kflux+1)]-(flux[j+i*nbins]*LegPright)))/(c*hj);
            }else{
                Lk[k+j*(kflux+1)+i*nbins*(kflux+1)] = (((flt)2)*(intflux[k+j*(kflux+1)+i*nbins*(kflux+1)]-(flux[j+i*nbins]*LegPright - flux[j-1+i*nbins]*LegPleft)))/(c*hj);
            }
        }
	}
    
}

//--------------------
// Time solver SSP-RK3 k=0 Eq.38
// for multiplicative kernel
//--------------------
void solver_kmul(u32 i,u32 nbins, u32 kflux, u32 kp,const accfltr_t massgrid,const accfltr_t massbins,const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,
                const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                const accfltr_t tabK1,const accfltr_t tabK2,
                const accfltrw_t flux,const accfltrw_t intflux,const accfltrw_t coeff_Leg,const accfltrw_t coeff_gh,const accfltrw_t tabminvalgh,const accfltrw_t tabgamma,const accfltrw_t Lk,const accfltrw_t Lk_1,const accfltrw_t Lk_2,flt dt){

    //time solver + scaling limiter for each step
    //step 1
    Lk_func_kmul(i,nbins,kflux,kp,massgrid,gij,
                tabipiflux,tabipiAintflux,tabipiBintflux,
                tabK1,tabK2,coeff_Leg,
                flux,intflux,Lk);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=0;k<=kflux;k++){
            gij1[k+j*(kflux+1)+i*nbins*(kflux+1)] = gij[k+j*(kflux+1)+i*nbins*(kflux+1)] + dt*Lk[k+j*(kflux+1)+i*nbins*(kflux+1)];
        }
    }

    //apply scaling limiter
    gammafunction(i,nbins,kflux,massgrid,massbins,gij1,coeff_Leg,coeff_gh,tabminvalgh,tabgamma);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=1;k<=kflux;k++){
            gij1[k+j*(kflux+1)+i*nbins*(kflux+1)] = tabgamma[j+i*nbins]*gij1[k+j*(kflux+1)+i*nbins*(kflux+1)];
        }
    }
    
    //step 2
    Lk_func_kmul(i,nbins,kflux,kp,massgrid,gij1,
                tabipiflux,tabipiAintflux,tabipiBintflux,
                tabK1,tabK2,coeff_Leg,
                flux,intflux,Lk_1);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=0;k<=kflux;k++){
            gij2[k+j*(kflux+1)+i*nbins*(kflux+1)] = ((flt)3)*gij[k+j*(kflux+1)+i*nbins*(kflux+1)]/((flt)4) + (gij1[k+j*(kflux+1)+i*nbins*(kflux+1)] + dt*Lk_1[k+j*(kflux+1)+i*nbins*(kflux+1)])/((flt)4);
        }
    }

    //apply scaling limiter
    gammafunction(i,nbins,kflux,massgrid,massbins,gij2,coeff_Leg,coeff_gh,tabminvalgh,tabgamma);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=1;k<=kflux;k++){
            gij2[k+j*(kflux+1)+i*nbins*(kflux+1)] = tabgamma[j+i*nbins]*gij2[k+j*(kflux+1)+i*nbins*(kflux+1)];
        }
    }
    
    //step 3
    Lk_func_kmul(i,nbins,kflux,kp,massgrid,gij2,
                tabipiflux,tabipiAintflux,tabipiBintflux,
                tabK1,tabK2,coeff_Leg,
                flux,intflux,Lk_2);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=0;k<=kflux;k++){
            gij[k+j*(kflux+1)+i*nbins*(kflux+1)] = gij[k+j*(kflux+1)+i*nbins*(kflux+1)]/((flt)3) + ((flt)2)*(gij2[k+j*(kflux+1)+i*nbins*(kflux+1)] + dt*Lk_2[k+j*(kflux+1)+i*nbins*(kflux+1)])/((flt)3);
        }
    }

    //apply scaling limiter
    gammafunction(i,nbins,kflux,massgrid,massbins,gij,coeff_Leg,coeff_gh,tabminvalgh,tabgamma);
    for (u32 j=0;j<=nbins-1;j++){
        for (u32 k=1;k<=kflux;k++){
            gij[k+j*(kflux+1)+i*nbins*(kflux+1)] = tabgamma[j+i*nbins]*gij[k+j*(kflux+1)+i*nbins*(kflux+1)];
        }
    }

    
}





