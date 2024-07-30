//-------------------------------------------
// Functions for Legendre polynomials
// Equations refere to Lombart & Laibe (2020) (doi:10.1093/mnras/staa3682)
//-------------------------------------------
#include "options.hpp"
#include "polynomials_legendre.hpp"
#include <math.h>
#include <iostream>

//-------------------------------------------
// Normalisation coefficient for Legendre polynomials in Eq.15
//-------------------------------------------
flt coeffnorm(u32 i){
	return ((flt)2)/(((flt)2)*i+((flt)1));
}

//-------------------------------------------
// Compute coefficients for Legendre polynomials up to order 3
//-------------------------------------------
void compute_coeff_Leg(u32 k,flt coeff_Leg[]){
	if (k==0){
		coeff_Leg[0]=((flt)1);
		coeff_Leg[1]=((flt)0);
		coeff_Leg[2]=((flt)0);
		coeff_Leg[3]=((flt)0);
	}else if (k==1){
		coeff_Leg[0]=((flt)0);
		coeff_Leg[1]=((flt)1);
		coeff_Leg[2]=((flt)0);
		coeff_Leg[3]=((flt)0);
	}else if (k==2){
		coeff_Leg[0]=-((flt)5e-1);
		coeff_Leg[1]=((flt)0);
		coeff_Leg[2]=((flt)15e-1);
		coeff_Leg[3]=((flt)0);
	}else if (k==3){
		coeff_Leg[0]=((flt)0);
		coeff_Leg[1]=-((flt)15e-1);
		coeff_Leg[2]=((flt)0);
		coeff_Leg[3]=((flt)25e-1);
	}else{
		// printf("polynomials_legendre.c ->  coeff_Leg -> Wrong order \n");
		// exit(-1);
	}
		
}

//-------------------------------------------
// Compute coefficients for Legendre polynomials up to order 3
// sycl kernel
//-------------------------------------------
void compute_coeff_Leg_sycl(u32 i,u32 k,const accfltrw_t coeff_Leg){
	if (k==0){
		coeff_Leg[4*i+0]=((flt)1);
		coeff_Leg[4*i+1]=((flt)0);
		coeff_Leg[4*i+2]=((flt)0);
		coeff_Leg[4*i+3]=((flt)0);
	}else if (k==1){
		coeff_Leg[4*i+0]=((flt)0);
		coeff_Leg[4*i+1]=((flt)1);
		coeff_Leg[4*i+2]=((flt)0);
		coeff_Leg[4*i+3]=((flt)0);
	}else if (k==2){
		coeff_Leg[4*i+0]=-((flt)5e-1);
		coeff_Leg[4*i+1]=((flt)0);
		coeff_Leg[4*i+2]=((flt)15e-1);
		coeff_Leg[4*i+3]=((flt)0);
	}else if (k==3){
		coeff_Leg[4*i+0]=((flt)0);
		coeff_Leg[4*i+1]=-((flt)15e-1);
		coeff_Leg[4*i+2]=((flt)0);
		coeff_Leg[4*i+3]=((flt)25e-1);
	}else{
		
		// printf("polynomials_legendre.c ->  coeff_Leg -> Wrong order \n");
		// exit(-1);
	}
		
}





//-------------------------------------------
// Compute  Legendre polynomials up to order 3 evaluated in x
//-------------------------------------------
flt Leg_P(u32 k,flt x){
	flt coeff_Leg[4];
	compute_coeff_Leg(k,coeff_Leg);

	if (k==0){
		return coeff_Leg[0];
	}else if (k==1){
		return coeff_Leg[0] + coeff_Leg[1]*x;
	}else if (k==2){
		return coeff_Leg[0] + coeff_Leg[1]*x + coeff_Leg[2]*pow(x,((flt)2));
	}else if (k==3){
		return coeff_Leg[0] + coeff_Leg[1]*x + coeff_Leg[2]*pow(x,((flt)2)) + coeff_Leg[3]*pow(x,((flt)3));
	}else{
		printf("polynomials_legendre.cpp ->  Leg_P -> Wrong order \n");
		// exit(-1);
	}
	
	return 0;
}

//-------------------------------------------
// Compute  Legendre polynomials up to order 3 evaluated in x
//-------------------------------------------
flt Leg_P_sycl(u32 i, u32 k,const accfltrw_t coeff_Leg,flt x){
	flt coeff_Leg_sycl[4];
	compute_coeff_Leg_sycl(i,k,coeff_Leg);

	if (k==0){
		return coeff_Leg[4*i+0];
	}else if (k==1){
		return coeff_Leg[4*i+0] + coeff_Leg[4*i+1]*x;
	}else if (k==2){
		return coeff_Leg[4*i+0] + coeff_Leg[4*i+1]*x + coeff_Leg[4*i+2]*sycl::pow(x,((flt)2));
	}else if (k==3){
		return coeff_Leg[4*i+0] + coeff_Leg[4*i+1]*x + coeff_Leg[4*i+2]*sycl::pow(x,((flt)2)) + coeff_Leg[4*i+3]*sycl::pow(x,((flt)3));
	}else{
		// printf("polynomials_legendre.cpp ->  Leg_P -> Wrong order \n");
		// exit(-1);
	}
	
	return 0;
}
