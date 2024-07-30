//--------------------------
// Function to generate tabflux for scheme semi analytic
// ballistic kernel
// Equations refere to Lombart & Laibe (2020) (doi:10.1093/mnras/staa3682)
//--------------------------
#include "../../options.hpp"
#include "../../polynomials_legendre.hpp"
#include "flux_components.hpp"
#include "flux_function.hpp"
#include <iostream>




//------------------------------
// function to generate term T for conservative flux, similar to Eq.25
//------------------------------
flt fluxfunction(unsigned int nbins,unsigned int kflux,flt* massgrid,flt* massbins,unsigned int j, unsigned int lp,unsigned int l,unsigned int ip, unsigned int i,unsigned int pp,unsigned int p){
	flt xlgridl = massgrid[l];
	flt xlgridr = massgrid[l+1];
	flt xlpgridl = massgrid[lp];
	flt xlpgridr = massgrid[lp+1];
    flt hlp = xlpgridr - xlpgridl;
	flt hl = xlgridr - xlgridl;
	flt xl = massbins[l];
	flt xlp = massbins[lp];
    flt xjgridr = massgrid[j+1];
	
	flt xmin = massgrid[0];
	flt xmax = massgrid[nbins];
    
    
    flt aip0; flt aip1; flt aip2; flt aip3;
    flt ai0; flt ai1; flt ai2; flt ai3;
    flt app0; flt app1; flt app2; flt app3;
    flt ap0; flt ap1; flt ap2; flt ap3;

    //coeff kernel approx
    flt coeff_Leg[4];
    compute_coeff_Leg(pp,coeff_Leg);
    app0 = coeff_Leg[0];app1 = coeff_Leg[1];app2 = coeff_Leg[2];app3 = coeff_Leg[3];
    compute_coeff_Leg(p,coeff_Leg);
    ap0 = coeff_Leg[0];ap1 = coeff_Leg[1];ap2 = coeff_Leg[2];ap3 = coeff_Leg[3];



    if (kflux>0){
        compute_coeff_Leg(ip,coeff_Leg);
        aip0 = coeff_Leg[0];aip1 = coeff_Leg[1];aip2 = coeff_Leg[2];aip3 = coeff_Leg[3];
        compute_coeff_Leg(i,coeff_Leg);
        ai0 = coeff_Leg[0];ai1 = coeff_Leg[1];ai2 = coeff_Leg[2];ai3 = coeff_Leg[3];
    }
	
    // first term res1
	flt res1 = 0;
	if (xmax > xlgridr + xlpgridr - xmin){
        if (xjgridr > xlgridr + xlpgridr - xmin){
            res1 = 0;
        }else{
            switch (kflux) {
                case 0:
                    res1 = dbintT1mixT3mix_k0(ap0,ap1,app0,app1,xl,hl);
                    break;
                case 1: 
                    res1 = dbintipiT1mixT3mix_k1(ap0,ap1,app0,app1,ai0,ai1,aip0,aip1,xl,hl);
                    break;
                    
                default :
                    printf("missing dbintipiT1mixT3mix for kflux > 1 \n");
                
            }
        }

    }else if (xmax <= xlgridr + xlpgridr - xmin && xmax > xlgridl + xlpgridr - xmin && xmax > xlgridr + xlpgridl - xmin) {
        switch (kflux) {
            case 0:
                res1 = dbintT1mix_k0(ap0,ap1,app0,app1,xlp,hlp,xl,hl,xmin,xmax)-dbintT3_k0(ap0,ap1,app0,app1,xlp,hlp,xl,hl,xlgridr,xmin,xmax);
                break;
            case 1:
                res1 = dbintipiT1mix_k1(ap0,ap1,app0,app1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xmin,xmax) - dbintipiT3_k1(ap0,ap1,app0,app1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridr,xmin,xmax);
                break;
                
            default :
                printf("missing dbintipiT1mix-dbintipiT3 for kflux > 1 \n");
        }

    }else if (xmax <= xlgridr + xlpgridr - xmin && xmax <= xlgridl + xlpgridr - xmin && xmax > xlgridr + xlpgridl - xmin) {
        switch (kflux) {
            case 0:
                res1 = dbintT1_k0(ap0,ap1,app0,app1,xlp,hlp,xl,hl,xlgridl,xmin,xmax)-dbintT3_k0(ap0,ap1,app0,app1,xlp,hlp,xl,hl,xlgridr,xmin,xmax);
                break;
            case 1:
                res1 = dbintipiT1_k1(ap0,ap1,app0,app1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridl,xmin,xmax) - dbintipiT3_k1(ap0,ap1,app0,app1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridr,xmin,xmax);
                break;

            default :
                printf("missing dbintipiT1-dbintipiT3 for kflux > 1 \n");
        }

    }else if (xmax <= xlgridr + xlpgridr - xmin && xmax > xlgridl + xlpgridr - xmin && xmax <= xlgridr + xlpgridl - xmin) {
        switch (kflux) {
            case 0:
                res1 = dbintT1mix_k0(ap0,ap1,app0,app1,xlp,hlp,xl,hl,xmin,xmax);
                break;
            case 1:
                res1 = dbintipiT1mix_k1(ap0,ap1,app0,app1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xmin,xmax);
                break;

            default :
                printf("missing dbintipiT1mix for kflux > 1 \n");
            
        }

    }else if (xmax <= xlgridr + xlpgridr - xmin && xmax <= xlgridl + xlpgridr - xmin && xmax <= xlgridr + xlpgridl - xmin && xmax > xlgridl + xlpgridl - xmin) {
        switch (kflux) {
            case 0: 
                res1 = dbintT1_k0(ap0,ap1,app0,app1,xlp,hlp,xl,hl,xlgridl,xmin,xmax);
                break;
            case 1:
                res1 = dbintipiT1_k1(ap0,ap1,app0,app1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridl,xmin,xmax);
                break;

            default :
                printf("missing dbintipiT1 for kflux > 1 \n");

        }
        
    }else{
        res1 = 0;
    }

    

    //second term res2
    flt res2 = 0;
    if (xjgridr > xlgridr + xlpgridr - xmin) {
        res2 = 0;

    }else if (xjgridr <= xlgridr + xlpgridr - xmin && xjgridr > xlgridl + xlpgridr - xmin && xjgridr > xlgridr + xlpgridl - xmin) {
        switch (kflux){
            case 0:
                res2 = dbintT2mix_k0(ap0,ap1,app0,app1,xlp,hlp,xl,hl,xjgridr,xmin)-dbintT4_k0(ap0,ap1,app0,app1,xlp,hlp,xl,hl,xlgridr,xjgridr,xmin);
                break;
            case 1:
                res2 = dbintipiT2mix_k1(ap0,ap1,app0,app1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xjgridr,xmin) - dbintipiT4_k1(ap0,ap1,app0,app1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridr,xjgridr,xmin);
                break;

            default :
                printf("missing dbintipiT2mix-dbintipiT4 for kflux > 1 \n");
        }


    }else if (xjgridr <= xlgridr + xlpgridr - xmin && xjgridr <= xlgridl + xlpgridr - xmin && xjgridr > xlgridr + xlpgridl - xmin) {
        switch (kflux) {
            case 0:
                res2 = dbintT2_k0(ap0,ap1,app0,app1,xlp,hlp,xl,hl,xlgridl,xjgridr,xmin)-dbintT4_k0(ap0,ap1,app0,app1,xlp,hlp,xl,hl,xlgridr,xjgridr,xmin);
                break;
            case 1:
                res2 = dbintipiT2_k1(ap0,ap1,app0,app1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridl,xjgridr,xmin) - dbintipiT4_k1(ap0,ap1,app0,app1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridr,xjgridr,xmin);
                break;

            default :
                printf("missing dbintipiT2-dbintipiT4 for kflux > 1 \n");

        }

    }else if (xjgridr <= xlgridr + xlpgridr - xmin && xjgridr > xlgridl + xlpgridr - xmin && xjgridr <= xlgridr + xlpgridl - xmin) {
        switch (kflux) {
            case 0: 
                res2 = dbintT2mix_k0(ap0,ap1,app0,app1,xlp,hlp,xl,hl,xjgridr,xmin);
                break;
            case 1:
                res2 = dbintipiT2mix_k1(ap0,ap1,app0,app1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xjgridr,xmin);
                break;

            default :
                printf("missing dbintipiT2mix for kflux > 1 \n");
                // exit(-1);
            
        }

    }else if (xjgridr <= xlgridr + xlpgridr - xmin && xjgridr <= xlgridl + xlpgridr - xmin && xjgridr <= xlgridr + xlpgridl - xmin && xjgridr > xlgridl + xlpgridl - xmin) {
        switch (kflux) {
            case 0:
                res2 = dbintT2_k0(ap0,ap1,app0,app1,xlp,hlp,xl,hl,xlgridl,xjgridr,xmin);
                break;
            case 1: 
                res2 = dbintipiT2_k1(ap0,ap1,app0,app1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridl,xjgridr,xmin);
                break;

            default :
                printf("missing dbintipiT2 for kflux > 1\n");
                // exit(-1);
        
        }

    }else {
        res2 = 0;
    }

    return ((res1+res2)*hl*hlp)/((flt)4);

}





