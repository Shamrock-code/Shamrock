//-------------------------------------------
// Functions for polynomials gh Eq.14
// Equations refere to Lombart & Laibe (2020) (doi:10.1093/mnras/staa3682)
//-------------------------------------------
#include "options.hpp"
#include <math.h>
#include "polynomials_legendre.hpp"
#include "reconstruction_gh.hpp"
#include <iostream>




//-------------------------------------------
//+
//  Reconstruction of polynomial gh for given x in bin j, Eq.14
//+
//-------------------------------------------
flt ghscalar(u32 i,u32 kflux,u32 nbins,const accfltr_t massgrid,const accfltr_t massbins,const accfltrw_t gij,u32 j,flt x){

   flt xjgridl = massgrid[j];
   flt xjgridr = massgrid[j+1];
   flt hj = xjgridr-xjgridl;
   flt xj = massbins[j];
   flt xij = ((flt)2)*(x-xj)/hj;

   flt sum = 0;
   for (u32 k=0;k<=kflux;k++){
      flt LegP = Leg_P(k,xij);     
      sum += gij[k+j*(kflux+1)+i*nbins*(kflux+1)]*LegP;
	}
	return sum;
}


//-------------------------------------------
//+
// Coefficient of polynomials gh in bin j up to order 3
//+
//-------------------------------------------
void compute_coeff_gh(u32 i,u32 kflux,u32 nbins,const accfltr_t massgrid,const accfltr_t massbins,const accfltrw_t gij, const accfltrw_t coeff_Leg, const accfltrw_t coeff_gh){
	flt a00; flt a01; flt a02; flt a03;
	flt a10; flt a11; flt a12; flt a13;
	flt a20; flt a21; flt a22; flt a23;
	flt a30; flt a31; flt a32; flt a33;
	flt g0j; flt g1j; flt g2j; flt g3j;

   for (u32 j=0;j<=nbins-1;j++){
      flt hj = massgrid[j+1]-massgrid[j];
      flt xj = massbins[j];
  
      if (kflux==0){
			g0j = gij[0+j*(kflux+1)+i*nbins*(kflux+1)];  g1j = ((flt)0);  g2j = ((flt)0);  g3j = ((flt)0);
	
			compute_coeff_Leg_sycl(i,0,coeff_Leg);
			a00 = coeff_Leg[0+i*4];  a01 = coeff_Leg[1+i*4];  a02 = coeff_Leg[2+i*4];  a03 = coeff_Leg[3+i*4];
			a10=((flt)0);  a11=((flt)0);  a12=((flt)0);  a13=((flt)0);
			a20=((flt)0);  a21=((flt)0);  a22=((flt)0);  a23=((flt)0);
			a30=((flt)0);  a31=((flt)0);  a32=((flt)0);  a33=((flt)0);
		}else if (kflux==1){	
			g0j = gij[0+j*(kflux+1)+i*nbins*(kflux+1)]; g1j = gij[1+j*(kflux+1)+i*nbins*(kflux+1)]; g2j = ((flt)0); g3j = ((flt)0);
			

			compute_coeff_Leg_sycl(i,0,coeff_Leg);
			a00 = coeff_Leg[0+i*4];  a01 = coeff_Leg[1+i*4];  a02 = coeff_Leg[2+i*4];  a03 = coeff_Leg[3+i*4];
			compute_coeff_Leg_sycl(i,1,coeff_Leg);
			a10 = coeff_Leg[0+i*4];  a11 = coeff_Leg[1+i*4];  a12 = coeff_Leg[2+i*4];  a13 = coeff_Leg[3+i*4];
			a20=((flt)0);  a21=((flt)0);  a22=((flt)0);  a23=((flt)0);
			a30=((flt)0);  a31=((flt)0);  a32=((flt)0);  a33=((flt)0);
		}else if (kflux==2){
			g0j = gij[0+j*(kflux+1)+i*nbins*(kflux+1)];  g1j = gij[1+j*(kflux+1)+i*nbins*(kflux+1)];  g2j = gij[2+j*(kflux+1)+i*nbins*(kflux+1)];  g3j = ((flt)0);
			
			compute_coeff_Leg_sycl(i,0,coeff_Leg);
			a00 = coeff_Leg[0+i*4];  a01 = coeff_Leg[1+i*4];  a02 = coeff_Leg[2+i*4];  a03 = coeff_Leg[3+i*4];
			compute_coeff_Leg_sycl(i,1,coeff_Leg);
			a10 = coeff_Leg[0+i*4];  a11 = coeff_Leg[1+i*4];  a12 = coeff_Leg[2+i*4];  a13 = coeff_Leg[3+i*4];
			compute_coeff_Leg_sycl(i,2,coeff_Leg);
			a20 = coeff_Leg[0+i*4];  a21 = coeff_Leg[1+i*4];  a22 = coeff_Leg[2+i*4];  a23 = coeff_Leg[3+i*4];
			a30=((flt)0);  a31=((flt)0);  a32=((flt)0);  a33=((flt)0);
		}else if (kflux==3){
			g0j = gij[0+j*(kflux+1)+i*nbins*(kflux+1)];  g1j = gij[1+j*(kflux+1)+i*nbins*(kflux+1)];  g2j = gij[2+j*(kflux+1)+i*nbins*(kflux+1)];  g3j = gij[3+j*(kflux+1)+i*nbins*(kflux+1)];
			
			compute_coeff_Leg_sycl(i,0,coeff_Leg);
			a00 = coeff_Leg[0+i*4];  a01 = coeff_Leg[1+i*4];  a02 = coeff_Leg[2+i*4];  a03 = coeff_Leg[3+i*4];
			compute_coeff_Leg_sycl(i,1,coeff_Leg);
			a10 = coeff_Leg[0+i*4];  a11 = coeff_Leg[1+i*4];  a12 = coeff_Leg[2+i*4];  a13 = coeff_Leg[3+i*4];
			compute_coeff_Leg_sycl(i,2,coeff_Leg);
			a20 = coeff_Leg[0+i*4];  a21 = coeff_Leg[1+i*4];  a22 = coeff_Leg[2+i*4];  a23 = coeff_Leg[3+i*4];
			compute_coeff_Leg_sycl(i,3,coeff_Leg);
			a30 = coeff_Leg[0+i*4];  a31 = coeff_Leg[1+i*4];  a32 = coeff_Leg[2+i*4];  a33 = coeff_Leg[3+i*4];
		}else{
			// printf("limiter.c -> coeff_gh -> Wrong order \n");
			// exit(-1);
		}
      
      
    
		coeff_gh[0+4*j+i*nbins*4] = (a00*g0j*sycl::pow(hj,((flt)3)) + a10*g1j*sycl::pow(hj,((flt)3)) + a20*g2j*sycl::pow(hj,((flt)3)) + a30*g3j*sycl::pow(hj,((flt)3)) - ((flt)2)*a01*g0j*sycl::pow(hj,((flt)2))*xj - ((flt)2)*a11*g1j*sycl::pow(hj,((flt)2))*xj - ((flt)2)*a21*g2j*sycl::pow(hj,((flt)2))*xj - ((flt)2)*a31*g3j*sycl::pow(hj,((flt)2))*xj + ((flt)4)*a02*g0j*hj*sycl::pow(xj,((flt)2)) + ((flt)4)*a12*g1j*hj*sycl::pow(xj,((flt)2)) + ((flt)4)*a22*g2j*hj*sycl::pow(xj,((flt)2)) + ((flt)4)*a32*g3j*hj*sycl::pow(xj,((flt)2)) - ((flt)8)*a03*g0j*sycl::pow(xj,((flt)3)) - ((flt)8)*a13*g1j*sycl::pow(xj,((flt)3)) - ((flt)8)*a23*g2j*sycl::pow(xj,((flt)3)) - ((flt)8)*a33*g3j*sycl::pow(xj,((flt)3)))/sycl::pow(hj,((flt)3));

		coeff_gh[1+4*j+i*nbins*4] = (((flt)2)*a01*g0j*sycl::pow(hj,((flt)2)) + ((flt)2)*a11*g1j*sycl::pow(hj,((flt)2)) + ((flt)2)*a21*g2j*sycl::pow(hj,((flt)2)) + ((flt)2)*a31*g3j*sycl::pow(hj,((flt)2)) - ((flt)8)*a02*g0j*hj*xj - ((flt)8)*a12*g1j*hj*xj - ((flt)8)*a22*g2j*hj*xj - ((flt)8)*a32*g3j*hj*xj + ((flt)24)*a03*g0j*sycl::pow(xj,((flt)2)) + ((flt)24)*a13*g1j*sycl::pow(xj,((flt)2)) + ((flt)24)*a23*g2j*sycl::pow(xj,((flt)2)) + ((flt)24)*a33*g3j*sycl::pow(xj,((flt)2)))/sycl::pow(hj,((flt)3));

		coeff_gh[2+4*j+i*nbins*4] = (((flt)4)*a02*g0j*hj + ((flt)4)*a12*g1j*hj + ((flt)4)*a22*g2j*hj + ((flt)4)*a32*g3j*hj - ((flt)24)*a03*g0j*xj - ((flt)24)*a13*g1j*xj - ((flt)24)*a23*g2j*xj - ((flt)24)*a33*g3j*xj)/sycl::pow(hj,((flt)3));
	
		coeff_gh[3+4*j+i*nbins*4] = (((flt)8)*a03*g0j + ((flt)8)*a13*g1j + ((flt)8)*a23*g2j + ((flt)8)*a33*g3j)/sycl::pow(hj,((flt)3));
	}

}



