//-------------------------------------------
// L2 projection on Legendre polynomials basis 
// for initial condition in DG scheme Eq.19 with gauss legendre quadrature
// Equations refere to Lombart & Laibe (2020) (doi:10.1093/mnras/staa3682)
//-------------------------------------------

#include <math.h>
#include "options.hpp"
#include "polynomials_legendre.hpp"
#include "L2proj_GQ.hpp"
#include <iostream>

flt xhat(flt hj,flt xj,flt node){

	return xj + hj*node/((flt)2);

}

void L2proj_gij_GQ(u32 nbins, u32 kflux, flt* massgrid, flt* massbins, u32 Q,flt* vecnodes,flt* vecweights, flt* gij){
	
	flt xj;
	flt hj;
	flt c;
	flt sum;
	flt xjalpha;
	flt LegP;

	for (u32 j=0; j<nbins; j++){
		xj = massbins[j];
		hj = massgrid[j+1]-massgrid[j];

		for (u32 k=0; k <= kflux; k++){
			c = coeffnorm(k);
			sum = ((flt)0);

			for (u32 alpha=0;alpha<Q;alpha++){
				xjalpha = xhat(hj,xj,vecnodes[alpha]);
				LegP = Leg_P(k,vecnodes[alpha]);
				sum = sum + vecweights[alpha]*xjalpha*exp(-xjalpha)*LegP;
			
			}

			gij[k+j*(kflux+1)] = sum/c;
		}

		
	}

}

void L2proj_gij_GQ_kmul(u32 nbins, u32 kflux, flt* massgrid, flt* massbins, u32 Q,flt* vecnodes,flt* vecweights, flt* gij){

	flt xj;
	flt hj;
	flt c;
	flt sum;
	flt xjalpha;
	flt LegP;

	for (u32 j=0; j<nbins; j++){
		xj = massbins[j];
		hj = massgrid[j+1]-massgrid[j];

		for (u32 k=0; k <= kflux; k++){
			c = coeffnorm(k);
			sum = ((flt)0);

			for (u32 alpha=0;alpha<Q;alpha++){
				xjalpha = xhat(hj,xj,vecnodes[alpha]);
				LegP = Leg_P(k,vecnodes[alpha]);
				sum = sum + vecweights[alpha]*exp(-xjalpha)*LegP;
			
			}

			gij[k+j*(kflux+1)] = sum/c;
		}
	}
}

void L2proj_kadd_GQ(u32 nbins, u32 kp, flt* massgrid, flt* massbins, u32 Q,flt* vecnodes,flt* vecweights,
					flt* tabK1F1, flt* tabK2F1, flt* tabK1F2, flt* tabK2F2){
	
	flt xj;
	flt hj;
	flt c;
	flt sumK1F1; flt sumK2F2;

	flt xjalpha;
	flt LegP;

	//initialisation
	for (u32 j=0; j<nbins; j++){
		for (u32 p=0; p <= kp; p++){
			tabK1F1[p+j*(kp+1)] = ((flt)0);
			tabK2F1[p+j*(kp+1)] = ((flt)0);
			tabK1F2[p+j*(kp+1)] = ((flt)0);
			tabK2F2[p+j*(kp+1)] = ((flt)0);
		}
	}

	for (u32 j=0; j<nbins; j++){
		xj = massbins[j];
		hj = massgrid[j+1]-massgrid[j];

		for (u32 p=0; p <= kp; p++){
			c = coeffnorm(p);
			sumK1F1 = ((flt)0);
			sumK2F2 = ((flt)0);

			for (u32 alpha=0;alpha<Q;alpha++){
				xjalpha = xhat(hj,xj,vecnodes[alpha]);
				LegP = Leg_P(p,vecnodes[alpha]);
				sumK1F1 = sumK1F1 + vecweights[alpha]*xjalpha*LegP;
			
			}

			sumK2F2 = sumK1F1;

			tabK1F1[p+j*(kp+1)] = sumK1F1/c;
			tabK2F2[p+j*(kp+1)] = sumK2F2/c;
		}

		tabK2F1[0+j*(kp+1)] = ((flt)1);
		tabK1F2[0+j*(kp+1)] = ((flt)1);

		
	}


}

