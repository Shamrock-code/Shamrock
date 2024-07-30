//--------------------------
// Function to generate tabintflux for scheme semi analytic (kernel not integrated)
// in Eq.26 second integral on [xmin,x] is split into 2 integrals: termA for [xmin,x_{j-1/2}] and termB for [x_{j-1/2},x]
// Equations refere to Lombart & Laibe (2020) (doi:10.1093/mnras/staa3682)
//--------------------------
#include "../../../options.hpp"
#include "../../../polynomials_legendre.hpp"
#include "termB_intflux_components.hpp"
#include "termB_intflux_function.hpp"
#include <iostream>

flt termB_intfluxfunction(unsigned int nbins,unsigned int kflux,flt* massgrid,flt* massbins,unsigned int j,unsigned int k,unsigned int l,unsigned int ip, unsigned int i,unsigned int pp,unsigned int p){
    flt xlgridl = massgrid[l];
	flt xlgridr = massgrid[l+1];
	flt hl = xlgridr - xlgridl;
	flt xl = massbins[l];
    flt xjgridr = massgrid[j+1];
    flt xjgridl = massgrid[j];
    flt hj = xjgridr - xjgridl;
	flt xj = massbins[j];
	
	flt xmin = massgrid[0];
	flt xmax = massgrid[nbins];
    
    
    flt aip0; flt aip1; flt aip2; flt aip3;
    flt ai0; flt ai1; flt ai2; flt ai3;
    flt ak0; flt ak1; flt ak2; flt ak3;

    flt app0; flt app1; flt app2; flt app3;
    flt ap0; flt ap1; flt ap2; flt ap3;

    //coeff kernel approx
    flt coeff_Leg[4];
    compute_coeff_Leg(pp,coeff_Leg);
    app0 = coeff_Leg[0];app1 = coeff_Leg[1];app2 = coeff_Leg[2];app3 = coeff_Leg[3];
    compute_coeff_Leg(p,coeff_Leg);
    ap0 = coeff_Leg[0];ap1 = coeff_Leg[1];ap2 = coeff_Leg[2];ap3 = coeff_Leg[3];

    compute_coeff_Leg(ip,coeff_Leg);
    aip0 = coeff_Leg[0];aip1 = coeff_Leg[1];aip2 = coeff_Leg[2];aip3 = coeff_Leg[3];
    compute_coeff_Leg(i,coeff_Leg);
    ai0 = coeff_Leg[0];ai1 = coeff_Leg[1];ai2 = coeff_Leg[2];ai3 = coeff_Leg[3];
    compute_coeff_Leg(k,coeff_Leg);
    ak0 = coeff_Leg[0];ak1 = coeff_Leg[1];ak2 = coeff_Leg[2];ak3 = coeff_Leg[3];

    flt res = 0;
    
    flt res1 = 0;
    if (xmax > xlgridr + xjgridr - xmin){
        switch (kflux) {
            case 1:
                res1 = BtpintipiT1T4mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl);
                break;

            default :
                printf("missing BtpintipiT1T4mix for kflux > 1 \n");
                // exit(-1);
            
        }

    }else if (xmax <= xlgridr + xjgridr - xmin && xmax > xlgridl + xjgridr - xmin && xmax > xlgridr + xjgridl - xmin){
        switch (kflux) {
            case 1:
                res1 = BtpintipiT12mixT11_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl,xj,hj,xmin,xmax)
                            - BtpintipiT41_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl,xlgridr,xj,hj,xmin,xmax)
                            - BtpintipiT42_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl,xlgridr,xj,hj,xmin,xmax);
                break;

            default :
                printf("missing BtpintipiT12mixT11-BtpintipiT41-BtpintipiT42 for kflux > 1 \n");
                // exit(-1);
            
        }

    }else if (xmax <= xlgridr + xjgridr - xmin && xmax <= xlgridl + xjgridr - xmin && xmax > xlgridr + xjgridl - xmin){
        switch (kflux) {
            case 1:
                res1 = BtpintipiT11_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl,xlgridl,xj,hj,xmin,xmax)
                            + BtpintipiT12_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl,xlgridl,xj,hj,xmin,xmax)
                            - BtpintipiT41_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl,xlgridr,xj,hj,xmin,xmax)
                            - BtpintipiT42_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl,xlgridr,xj,hj,xmin,xmax);
                break;
            

            default :
                printf("missing BtpintipiT11+BtpintipiT12-BtpintipiT41-BtpintipiT42 for kflux > 1 \n");
                // exit(-1);
            
        }

    }else if (xmax <= xlgridr + xjgridr - xmin && xmax > xlgridl + xjgridr - xmin && xmax <= xlgridr + xjgridl - xmin){
        switch (kflux) {
            case 1:
                res1 = BtpintipiT12mixT11_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl,xj,hj,xmin,xmax);
                break;
            

            default :
                printf("missing BtpintipiT12mixT11 for kflux > 1 \n");
                // exit(-1);
            
        }

    }else if (xmax <= xlgridr + xjgridr - xmin && xmax <= xlgridl + xjgridr - xmin && xmax <= xlgridr + xjgridl - xmin && xmax > xlgridl + xjgridl - xmin){
        switch (kflux) {
            case 1:
                res1 = BtpintipiT11_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl,xlgridl,xj,hj,xmin,xmax)
                            + BtpintipiT12_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl,xlgridl,xj,hj,xmin,xmax);
                break;

            default :
                printf("missing BtpintipiT11+BtpintipiT12 for kflux > 1 \n");
                // exit(-1);
            
        }

    }else{
        res1 = 0;
    }


    flt res2 = 0;
    if (xjgridr > xlgridr + xjgridl - xmin){
        switch (kflux) {
            case 1:
                res2 = BtpintipiT2_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl,xlgridl,xj,hj,xjgridl,xmin)
                            - BtpintipiT5_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl,xlgridr,xj,hj,xjgridl,xmin);

                break;

            default :
                printf("missing BtpintipiT2-BtpintipiT5 for kflux > 1 \n");
                // exit(-1);
            
        }

    }else if (xjgridr <= xlgridr + xjgridl - xmin && xjgridr > xlgridl + xjgridl - xmin){
        switch (kflux) {
            case 1: //tpint
                res2 = BtpintipiT2_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl,xlgridl,xj,hj,xjgridl,xmin);
                break;

            default :
                 printf("missing BtpintipiT2 for kflux > 1 \n");
                // exit(-1);
            
        }
    }else{
        res2 = 0;
    }

    return hj*hl*(res1+res2)/((flt)4);


}


