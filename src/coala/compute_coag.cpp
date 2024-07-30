//-------------------------------------------
// Functions to compute coagulation solver for 1 hydro time-step 
//-------------------------------------------
#include "options.hpp"
#include "solver_DG.hpp"
#include <stdlib.h>
#include <stdio.h>

//-------------------------------------------
// compute coagulation solver for 1 dthydro
// constant kernel k=0
//-------------------------------------------
void compute_coag_k0_kconst(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,
							const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,
							const accfltr_t tabflux,  const accfltr_t tabK1,const accfltr_t tabK2,
							const accfltrw_t flux,const accfltrw_t tabdflux, const accfltrw_t tabdtCFL,
							const accfltrw_t Lk0,const accfltrw_t Lk0_1,const accfltrw_t Lk0_2,
							flt dthydro){

	flt dtCFLsub;
	flt dtsub;
	flt dtlast;
	flt dt;

	//get CFL condition
    dtCFLsub = CFL_k0_kconst(i,nbins,kp,massgrid,gij,tabflux,tabK1,tabK2,flux,tabdflux,tabdtCFL);
	dt = fmin(dtCFLsub,dthydro);
	// dt = dthydro;

    if (dt < dthydro){
			dtsub = ((flt)0);
            //loop for subcyling coagulation time-step if required
			while (dtsub<dthydro && dthydro-dtsub>dtCFLsub){
				dtsub=dtsub+dtCFLsub;
				
				solver_k0_kconst(i,nbins,kp,massgrid,gij,gij1,gij2,tabflux,tabK1,tabK2,flux,Lk0,Lk0_1,Lk0_2,dtCFLsub);
				
				dtCFLsub = CFL_k0_kconst(i,nbins,kp,massgrid,gij,tabflux,tabK1,tabK2,flux,tabdflux,tabdtCFL);
				// printf("dtCFLsub=%.15e \n",dtCFLsub);
				 
		
			}
		
			//add last coag timestep to reach hydro timestep
			dtlast = dthydro-dtsub;
			solver_k0_kconst(i,nbins,kp,massgrid,gij,gij1,gij2,tabflux,tabK1,tabK2,flux,Lk0,Lk0_1,Lk0_2,dtlast);
		
		}else {
			//run coag with hydro timestep
			solver_k0_kconst(i,nbins,kp,massgrid,gij,gij1,gij2,tabflux,tabK1,tabK2,flux,Lk0,Lk0_1,Lk0_2,dt);
			
		}


}

//-------------------------------------------
// compute coagulation solver for 1 dthydro
// additive kernel k=0
//-------------------------------------------
void compute_coag_k0_kadd(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,
							const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,
							const accfltr_t tabflux,const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,
							const accfltrw_t flux,const accfltrw_t tabdflux, const accfltrw_t tabdtCFL,
							const accfltrw_t Lk0,const accfltrw_t Lk0_1,const accfltrw_t Lk0_2,
							flt dthydro){

	flt dtCFLsub;
	flt dtsub;
	flt dtlast;
	flt dt;

	//get CFL condition
    dtCFLsub = CFL_k0_kadd(i,nbins,kp,massgrid,gij,tabflux,tabK1F1,tabK2F2,tabK1F2,tabK2F2,flux,tabdflux,tabdtCFL);
	dt = fmin(dtCFLsub,dthydro);
	// dt = dthydro;

    if (dt < dthydro){
			flt dtsub = ((flt)0);
            //loop for subcyling coagulation time-step if required
			while (dtsub<dthydro && dthydro-dtsub>dtCFLsub){
				dtsub=dtsub+dtCFLsub;
				
				solver_k0_kadd(i,nbins,kp,massgrid,gij,gij1,gij2,tabflux,tabK1F1,tabK2F1,tabK1F2,tabK2F2,flux,Lk0,Lk0_1,Lk0_2,dtCFLsub);
				
				dtCFLsub = CFL_k0_kadd(i,nbins,kp,massgrid,gij,tabflux,tabK1F1,tabK2F2,tabK1F2,tabK2F2,flux,tabdflux,tabdtCFL);
				// printf("dtCFLsub=%.15e \n",dtCFLsub);
				 
		
			}
		
			//add last coag timestep to reach hydro timestep
			dtlast = dthydro-dtsub;
			solver_k0_kadd(i,nbins,kp,massgrid,gij,gij1,gij2,tabflux,tabK1F1,tabK2F1,tabK1F2,tabK2F2,flux,Lk0,Lk0_1,Lk0_2,dtlast);
		
		}else {
			//run coag with hydro timestep
			solver_k0_kadd(i,nbins,kp,massgrid,gij,gij1,gij2,tabflux,tabK1F1,tabK2F1,tabK1F2,tabK2F2,flux,Lk0,Lk0_1,Lk0_2,dt);
			
		}


}

//-------------------------------------------
// compute coagulation solver for 1 dthydro
// multiplicative kernel k=0
//-------------------------------------------
void compute_coag_k0_kmul(u32 i,u32 nbins,u32 kp,const accfltr_t massgrid,
							const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,
							const accfltr_t tabflux,const accfltr_t tabK1,const accfltr_t tabK2,
							const accfltrw_t flux,const accfltrw_t tabdflux, const accfltrw_t tabdtCFL,
							const accfltrw_t Lk0,const accfltrw_t Lk0_1,const accfltrw_t Lk0_2,
							flt dthydro){

	flt dtCFLsub;
	flt dtsub;
	flt dtlast;
	flt dt;

	//get CFL condition
    dtCFLsub = CFL_k0_kmul(i,nbins,kp,massgrid,gij,tabflux,tabK1,tabK2,flux,tabdflux,tabdtCFL);
	dt = fmin(dtCFLsub,dthydro);
	// flt dt = dthydro;

    if (dt < dthydro){
			dtsub = ((flt)0);
            //loop for subcyling coagulation time-step if required
			while (dtsub<dthydro && dthydro-dtsub>dtCFLsub){
				dtsub=dtsub+dtCFLsub;
				
				solver_k0_kmul(i,nbins,kp,massgrid,gij,gij1,gij2,tabflux,tabK1,tabK2,flux,Lk0,Lk0_1,Lk0_2,dtCFLsub);
				
				dtCFLsub = CFL_k0_kmul(i,nbins,kp,massgrid,gij,tabflux,tabK1,tabK2,flux,tabdflux,tabdtCFL);
				// printf("dtCFLsub=%.15e \n",dtCFLsub);
				 
		
			}
		
			//add last coag timestep to reach hydro timestep
			dtlast = dthydro-dtsub;
			solver_k0_kmul(i,nbins,kp,massgrid,gij,gij1,gij2,tabflux,tabK1,tabK2,flux,Lk0,Lk0_1,Lk0_2,dtlast);
		
		}else {
			//run coag with hydro timestep
			solver_k0_kmul(i,nbins,kp,massgrid,gij,gij1,gij2,tabflux,tabK1,tabK2,flux,Lk0,Lk0_1,Lk0_2,dt);
			
		}


}





//-------------------------------------------
// compute coagulation solver for 1 dthydro
// constant kernel k>0
//-------------------------------------------
void compute_coag_kconst(u32 i,u32 nbins, u32 kflux,u32 kp, const accfltr_t massgrid,const accfltr_t massbins,
							const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,
                            const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                            const accfltr_t tabK1,const accfltr_t tabK2,
                            const accfltrw_t flux, const accfltrw_t intflux, const accfltrw_t tabdflux, const accfltrw_t tabdtCFL,
                            const accfltrw_t coeff_Leg,const accfltrw_t coeff_gh,const accfltrw_t tabminvalgh,const accfltrw_t tabgamma,
                            const accfltrw_t Lk,const accfltrw_t Lk_1,const accfltrw_t Lk_2,
                            flt dthydro){

	flt dtCFLsub;
	flt dtsub;
	flt dtlast;
	flt dt;

	//get CFL condition
    dtCFLsub = CFL_kconst(i,nbins,kflux,kp,massgrid,gij,
    						tabipiflux,tabK1,tabK2,
							flux,tabdflux,tabdtCFL);
	dt = fmin(dtCFLsub,dthydro);
	// dt = dthydro;

	// printf("dtCFL=%.15e\n",dtCFL);


    if (dt < dthydro){
			dtsub = ((flt)0);
            //loop for subcycling coagulation time-step if required
			while (dtsub<dthydro && dthydro-dtsub>dtCFLsub){
				dtsub=dtsub+dtCFLsub;

				solver_kconst(i,nbins, kflux,kp,massgrid,massbins,gij,gij1,gij2,
                                tabipiflux,tabipiAintflux,tabipiBintflux,
                                tabK1,tabK2,
                                flux,intflux,coeff_Leg,coeff_gh,tabminvalgh,tabgamma,Lk,Lk_1,Lk_2,dtCFLsub);
	
				dtCFLsub = CFL_kconst(i,nbins,kflux,kp,massgrid,gij,
			    						tabipiflux,tabK1,tabK2,
										flux,tabdflux,tabdtCFL);
				// printf("dtCFLsub=%.15e \n",dtCFLsub);
				 
		
			}
		
			//add last coag timestep to reach hydro timestep
			dtlast = dthydro-dtsub;

			solver_kconst(i,nbins, kflux,kp,massgrid,massbins,gij,gij1,gij2,
                            tabipiflux,tabipiAintflux,tabipiBintflux,
                            tabK1,tabK2,
                            flux,intflux,coeff_Leg,coeff_gh,tabminvalgh,tabgamma,Lk,Lk_1,Lk_2,dtlast);

		}else {
			//run coag with hydro timestep
			solver_kconst(i,nbins, kflux,kp,massgrid,massbins,gij,gij1,gij2,
                            tabipiflux,tabipiAintflux,tabipiBintflux,
                            tabK1,tabK2,
                            flux,intflux,coeff_Leg,coeff_gh,tabminvalgh,tabgamma,Lk,Lk_1,Lk_2,dt);

		}


}


//-------------------------------------------
// compute coagulation solver for 1 dthydro
// additive kernel k>0
//-------------------------------------------
void compute_coag_kadd(u32 i,u32 nbins, u32 kflux, u32 kp,const accfltr_t massgrid,const accfltr_t massbins,
						const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,
                        const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                        const accfltr_t tabK1F1,const accfltr_t tabK2F1,const accfltr_t tabK1F2,const accfltr_t tabK2F2,
                        const accfltrw_t flux,const accfltrw_t intflux, const accfltrw_t tabdflux, const accfltrw_t tabdtCFL,
                        const accfltrw_t coeff_Leg,const accfltrw_t coeff_gh,const accfltrw_t tabminvalgh,const accfltrw_t tabgamma,
                        const accfltrw_t Lk,const accfltrw_t Lk_1,const accfltrw_t Lk_2,
                        flt dthydro){

	flt dtCFLsub;
	flt dtsub;
	flt dtlast;
	flt dt;

	//get CFL condition
    dtCFLsub = CFL_kadd(i,nbins,kflux,kp,massgrid,gij,
            			tabipiflux,tabK1F1,tabK2F1,tabK1F2,tabK2F2,
						flux,tabdflux,tabdtCFL);
	dt = fmin(dtCFLsub,dthydro);
	// dt = dthydro;

	// printf("dtCFL=%.15e\n",dtCFL);

    if (dt < dthydro){
			dtsub = ((flt)0);
            //loop for subcyling coagulation time-step if required
			while (dtsub<dthydro && dthydro-dtsub>dtCFLsub){
				dtsub=dtsub+dtCFLsub;

				solver_kadd(i,nbins,kflux,kp,massgrid,massbins,gij,gij1,gij2,
                            tabipiflux,tabipiAintflux,tabipiBintflux,
                            tabK1F1,tabK2F1,tabK1F2,tabK2F2,
                            flux,intflux,coeff_Leg,coeff_gh,tabminvalgh,tabgamma,Lk,Lk_1,Lk_2,dtCFLsub);
				
				dtCFLsub = CFL_kadd(i,nbins,kflux,kp,massgrid,gij,
			            			tabipiflux,tabK1F1,tabK2F1,tabK1F2,tabK2F2,
									flux,tabdflux,tabdtCFL);
				// printf("dtCFLsub=%.15e \n",dtCFLsub);
				 
		
			}

			//add last coag timestep to reach hydro timestep
			dtlast = dthydro-dtsub;

			solver_kadd(i,nbins,kflux,kp,massgrid,massbins,gij,gij1,gij2,
                            tabipiflux,tabipiAintflux,tabipiBintflux,
                            tabK1F1,tabK2F1,tabK1F2,tabK2F2,
                            flux,intflux,coeff_Leg,coeff_gh,tabminvalgh,tabgamma,Lk,Lk_1,Lk_2,dtlast);
		
		}else {
			//run coag with hydro timestep
			solver_kadd(i,nbins,kflux,kp,massgrid,massbins,gij,gij1,gij2,
                            tabipiflux,tabipiAintflux,tabipiBintflux,
                            tabK1F1,tabK2F1,tabK1F2,tabK2F2,
                            flux,intflux,coeff_Leg,coeff_gh,tabminvalgh,tabgamma,Lk,Lk_1,Lk_2,dt);
			
		}


}


//-------------------------------------------
// compute coagulation solver for 1 dthydro
// multiplicative kernel k>0
//-------------------------------------------
void compute_coag_kmul(u32 i,u32 nbins, u32 kflux, u32 kp,const accfltr_t massgrid,const accfltr_t massbins,
						const accfltrw_t gij,const accfltrw_t gij1,const accfltrw_t gij2,
                        const accfltr_t tabipiflux,const accfltr_t tabipiAintflux,const accfltr_t tabipiBintflux,
                        const accfltr_t tabK1,const accfltr_t tabK2,
                        const accfltrw_t flux,const accfltrw_t intflux,const accfltrw_t tabdflux, const accfltrw_t tabdtCFL,
                        const accfltrw_t coeff_Leg,const accfltrw_t coeff_gh,const accfltrw_t tabminvalgh,const accfltrw_t tabgamma,
                        const accfltrw_t Lk,const accfltrw_t Lk_1,const accfltrw_t Lk_2,
                        flt dthydro){

	flt dtCFLsub;
	flt dtsub;
	flt dtlast;
	flt dt;

	//get CFL condition
    dtCFLsub = CFL_kmul(i,nbins,kflux,kp,massgrid,gij,
            			tabipiflux,
            			tabK1,tabK2,
						flux,tabdflux,tabdtCFL);
	dt = fmin(dtCFLsub,dthydro);
	// dt = dthydro;

	// printf("dtCFLsub=%.15e \n",dtCFLsub);

    if (dt < dthydro){
			dtsub = ((flt)0);
            //loop for subcyling coagulation time-step if required
			while (dtsub<dthydro && dthydro-dtsub>dtCFLsub){
				dtsub=dtsub+dtCFLsub;


				solver_kmul(i,nbins,kflux,kp,massgrid,massbins,gij,gij1,gij2,
                            tabipiflux,tabipiAintflux,tabipiBintflux,
                            tabK1,tabK2,
                            flux,intflux,coeff_Leg,coeff_gh,tabminvalgh,tabgamma,Lk,Lk_1,Lk_2,dtCFLsub);
	
				dtCFLsub = CFL_kmul(i,nbins,kflux,kp,massgrid,gij,
			            			tabipiflux,
			            			tabK1,tabK2,
									flux,tabdflux,tabdtCFL);
				// printf("dtCFLsub=%.15e \n",dtCFLsub);
				// printf("dtsub=%.15e \n",dtsub);
				 
		
			}
		
			//add last coag timestep to reach hydro timestep
			dtlast = dthydro-dtsub;

			solver_kmul(i,nbins,kflux,kp,massgrid,massbins,gij,gij1,gij2,
                        tabipiflux,tabipiAintflux,tabipiBintflux,
                        tabK1,tabK2,
                        flux,intflux,coeff_Leg,coeff_gh,tabminvalgh,tabgamma,Lk,Lk_1,Lk_2,dtlast);
		
		}else {

			//run coag with hydro timestep
			solver_kmul(i,nbins,kflux,kp,massgrid,massbins,gij,gij1,gij2,
                        tabipiflux,tabipiAintflux,tabipiBintflux,
                        tabK1,tabK2,
                        flux,intflux,coeff_Leg,coeff_gh,tabminvalgh,tabgamma,Lk,Lk_1,Lk_2,dt);
			
		}


}



