//--------------------------
// Function to generate tabintflux for scheme semi analytic (kernel not integrated)
// in Eq.26 second integral on [xmin,x] is split into 2 integrals: termA for [xmin,x_{j-1/2}] and termB for [x_{j-1/2},x]
// Equations refere to Lombart & Laibe (2020) (doi:10.1093/mnras/staa3682)
//--------------------------
#include "../../../options.hpp"
#include "../../../polynomials_legendre.hpp"
#include "termA_intflux_components.hpp"
#include "termA_intflux_function.hpp"
#include <iostream>

//------------------------------
// function to generate term T for conservative flux, similar to Eq.26
//------------------------------
flt termA_intfluxfunction(unsigned int nbins,unsigned int kflux,flt* massgrid,flt* massbins,unsigned int j,unsigned int k,unsigned int lp,unsigned int l,unsigned int ip, unsigned int i,unsigned int pp,unsigned int p){
	flt xlgridl = massgrid[l];
	flt xlgridr = massgrid[l+1];
	flt xlpgridl = massgrid[lp];
	flt xlpgridr = massgrid[lp+1];
    flt hlp = xlpgridr - xlpgridl;
	flt hl = xlgridr - xlgridl;
	flt xl = massbins[l];
	flt xlp = massbins[lp];
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
    if (xmax > xlgridr + xlpgridr -xmin){
        if (xjgridr > xlgridr + xlpgridr - xmin && xjgridl > xlgridr + xlpgridr - xmin){
            res1 = 0;
        }else{
            switch (kflux) {
                case 1:
                    res1 = AtpintipiT1mixT4mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xl,hl);
                    break;
                
                default :
                    printf("missing AtpintiT1mixT4mix for kflux > 1 \n");
                    // exit(-1);
                
            }
        }

    }else if (xmax <= xlgridr + xlpgridr - xmin && xmax > xlgridl + xlpgridr - xmin && xmax > xlgridr + xlpgridl - xmin){
        switch (kflux) {
            case 1:
                res1 = AtpintipiT1mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xmin,xmax) - AtpintipiT4_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridr,xmin,xmax);
                break;

            default :
               printf("missing AtpintipiT1mix-AtpintipiT4 for kflux > 1 \n");
                // exit(-1);
            
        }

    }else if (xmax <= xlgridr + xlpgridr - xmin && xmax <= xlgridl + xlpgridr - xmin && xmax > xlgridr + xlpgridl - xmin){
        switch (kflux) {
            case 1:
                res1 = AtpintipiT1_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridl,xmin,xmax) - AtpintipiT4_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridr,xmin,xmax);
                break;
            
            default :
                printf("missing AtpintipiT1-AtpintipiT4 for kflux > 1 \n");
                // exit(-1);
            
        }

    }else if (xmax <= xlgridr + xlpgridr - xmin && xmax > xlgridl + xlpgridr - xmin && xmax <= xlgridr + xlpgridl - xmin){
        switch (kflux) {
            case 1:
                res1 = AtpintipiT1mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xmin,xmax);
                break;
            
            default :
                printf("missing AtpintipiT1mix for kflux > 1 \n");
                // exit(-1);
            
        }

    }else if (xmax <= xlgridr + xlpgridr - xmin && xmax <= xlgridl + xlpgridr - xmin && xmax <= xlgridr + xlpgridl - xmin && xmax > xlgridl + xlpgridl - xmin){
        switch (kflux) {
            case 1:
                res1 = AtpintipiT1_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridl,xmin,xmax);
                break;
            

            default :
                printf("missing AtpintipiT1 for kflux > 1 \n");
                // exit(-1);
            
        }

    }else{
        res1 = 0;
    }


    flt res2 = 0;
    if (xjgridr > xlgridr + xlpgridr - xmin){
        if (xjgridl > xlgridr + xlpgridr - xmin){
            res2 = 0;

        }else if (xjgridl <= xlgridr + xlpgridr - xmin && xjgridl > xlgridl + xlpgridr - xmin && xjgridl > xlgridr + xlpgridl - xmin){
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2mixT3mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xj,hj,xmin)
                                - AtpintipiT5mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridr,xj,hj,xmin)
                                - AtpintipiT6_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridr,xl,hl,xlgridr,xj,hj,xmin);
                    break;
                

                default :
                    printf("missing AtpintipiT2mixT3mix-AtpintipiT5mix-AtpintipiT6 for kflux > 3 \n");
                    // exit(-1);
                
            }

        }else if (xjgridl <= xlgridr + xlpgridr - xmin && xjgridl <= xlgridl + xlpgridr - xmin && xjgridl > xlgridr + xlpgridl - xmin){
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridl,xj,hj,xmin)
                                + AtpintipiT3_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridr,xl,hl,xlgridl,xj,hj,xmin)
                                - AtpintipiT5mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridr,xj,hj,xmin)
                                - AtpintipiT6_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridr,xl,hl,xlgridr,xj,hj,xmin);
                    break;
                

                default :
                    printf("missing AtpintipiT2mix+AtpintipiT3-AtpintipiT5mix-AtpintipiT6 for kflux > 1 \n");
                    // exit(-1);
                
            }

        }else if (xjgridl <= xlgridr + xlpgridr - xmin && xjgridl > xlgridl + xlpgridr - xmin && xjgridl <= xlgridr + xlpgridl - xmin){
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2mixT3mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xj,hj,xmin)
                                - AtpintipiT5_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridl,xl,hl,xlgridr,xj,hj,xmin)
                                - AtpintipiT6_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridr,xl,hl,xlgridr,xj,hj,xmin);
                    break;
                

                default :
                    printf("missing AtpintipiT2mixT3mix-AtpintipiT5-AtpintipiT6 for kflux > 1 \n");
                    // exit(-1);
                
            }

        }else if (xjgridl <= xlgridr + xlpgridr - xmin && xjgridl <= xlgridl + xlpgridr - xmin && xjgridl <= xlgridr + xlpgridl - xmin && xjgridl > xlgridl + xlpgridl - xmin){
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridl,xj,hj,xmin)
                                + AtpintipiT3_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridr,xl,hl,xlgridl,xj,hj,xmin)
                                - AtpintipiT5_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridl,xl,hl,xlgridr,xj,hj,xmin)
                                - AtpintipiT6_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridr,xl,hl,xlgridr,xj,hj,xmin);
                    break;
                
                default :
                    printf("missing AtpintipiT2mix+AtpintipiT3-AtpintipiT5-AtpintipiT6 for kflux > 1 \n");
                    // exit(-1);
                
            }

        }else{
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridl,xl,hl,xlgridl,xj,hj,xmin)
                                + AtpintipiT3_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridr,xl,hl,xlgridl,xj,hj,xmin)
                                - AtpintipiT5_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridl,xl,hl,xlgridr,xj,hj,xmin)
                                - AtpintipiT6_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridr,xl,hl,xlgridr,xj,hj,xmin);
                    break;
                

                default :
                    printf("missing AtpintipiT2+AtpintipiT3-AtpintipiT5-AtpintipiT6 for kflux > 1 \n");
                    // exit(-1);
                
            }

        }

    }else if (xjgridr <= xlgridr + xlpgridr - xmin && xjgridr > xlgridl + xlpgridr - xmin && xjgridr > xlgridr + xlpgridl - xmin){
        if (xjgridl > xlgridl + xlpgridr - xmin && xjgridl > xlgridr + xlpgridl - xmin){
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2mixT3mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xj,hj,xmin)
                                - AtpintipiT5mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridr,xj,hj,xmin);
                    break;
                

                default :
                    printf("missing AtpintipiT2mixT3mix-AtpintipiT5mix for kflux > 1 \n");
                    // exit(-1);
                
            }  

        }else if (xjgridl <= xlgridl + xlpgridr - xmin && xjgridl > xlgridr + xlpgridl - xmin){
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridl,xj,hj,xmin)
                                + AtpintipiT3_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridr,xl,hl,xlgridl,xj,hj,xmin)
                                - AtpintipiT5mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridr,xj,hj,xmin);
                    break;
                

                default :
                    printf("missing AtpintipiT2mix+AtpintipiT3-AtpintipiT5mix for kflux > 1 \n");
                    // exit(-1);
                
            }

        }else if (xjgridl > xlgridl + xlpgridr - xmin && xjgridl <= xlgridr + xlpgridl - xmin){
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2mixT3mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xj,hj,xmin)
                                - AtpintipiT5_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridl,xl,hl,xlgridr,xj,hj,xmin);
                    break;
                
                default :
                    printf("missing AtpintipiT2mixT3mix-AtpintipiT5 for kflux > 1 \n");
                    // exit(-1);
                
            }

        }else if (xjgridl <= xlgridl + xlpgridr - xmin && xjgridl <= xlgridr + xlpgridl - xmin && xjgridl > xlgridl + xlpgridl - xmin){
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridl,xj,hj,xmin)
                                + AtpintipiT3_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridr,xl,hl,xlgridl,xj,hj,xmin)
                                - AtpintipiT5_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridl,xl,hl,xlgridr,xj,hj,xmin);
                    break;
                

                default :
                    printf("missing AtpintipiT2mix+AtpintipiT3-AtpintipiT5 for kflux > 1 \n");
                    // exit(-1);
                
            }

        }else{
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridl,xl,hl,xlgridl,xj,hj,xmin)
                                + AtpintipiT3_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridr,xl,hl,xlgridl,xj,hj,xmin)
                                - AtpintipiT5_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridl,xl,hl,xlgridr,xj,hj,xmin);
                    break;
                
                default :
                    printf("missing AtpintipiT2+AtpintipiT3-AtpintipiT5 for kflux > 1 \n");
                    // exit(-1);
                
            }

        }

    }else if (xjgridr <= xlgridr + xlpgridr - xmin && xjgridr <= xlgridl + xlpgridr - xmin && xjgridr > xlgridr + xlpgridl - xmin){
        if (xjgridl > xlgridr + xlpgridl - xmin){
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridl,xj,hj,xmin)
                                - AtpintipiT5mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridr,xj,hj,xmin);
                    break;
                
                default :
                    printf("missing AtpintipiT2mix-AtpintipiT5mix for kflux > 1 \n");
                    // exit(-1);
                
            }

        }else if (xjgridl <= xlgridr + xlpgridl - xmin && xjgridl > xlgridl + xlpgridl - xmin){
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridl,xj,hj,xmin)
                                - AtpintipiT5_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridl,xl,hl,xlgridr,xj,hj,xmin);
                    break;
                

                default :
                    printf("missing AtpintipiT2mix-AtpintipiT5 for kflux > 1 \n");
                    // exit(-1);
                
            }

        }else {
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridl,xl,hl,xlgridl,xj,hj,xmin)
                                - AtpintipiT5_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridl,xl,hl,xlgridr,xj,hj,xmin);
                    break;
                

                default :
                    printf("missing AtpintipiT2-AtpintipiT5 for kflux > 1 \n");
                    // exit(-1);
                
            }
        }

    }else if (xjgridr <= xlgridr + xlpgridr - xmin && xjgridr > xlgridl + xlpgridr - xmin && xjgridr <= xlgridr + xlpgridl - xmin){
        if (xjgridl > xlgridl + xlpgridr - xmin){
            switch (kflux) {
                case 1: 
                    res2 = AtpintipiT2mixT3mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xj,hj,xmin);
                    break;
                

                default :
                    printf("missing AtpintipiT2mixT3mix for kflux > 1 \n");
                    // exit(-1);
                
            } 

        }else if (xjgridl <= xlgridl + xlpgridr - xmin && xjgridl > xlgridl + xlpgridl - xmin){
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridl,xj,hj,xmin)
                                + AtpintipiT3_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridr,xl,hl,xlgridl,xj,hj,xmin);
                    break;
                
                default :
                    printf("missing AtpintipiT2mix+AtpintipiT3 for kflux > 1 \n");
                    // exit(-1);
                
            }

        }else{
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridl,xl,hl,xlgridl,xj,hj,xmin)
                                + AtpintipiT3_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridr,xl,hl,xlgridl,xj,hj,xmin);
                    break;
                

                default :
                    printf("missing AtpintipiT2+AtpintipiT3 for kflux > 1 \n");
                    // exit(-1);
                
            }
        }

    }else if (xjgridr <= xlgridr + xlpgridr - xmin && xjgridr <= xlgridl + xlpgridr - xmin && xjgridr <= xlgridr + xlpgridl - xmin && xjgridr > xlgridl + xlpgridl - xmin){
        if (xjgridl > xlgridl + xlpgridl - xmin){
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2mix_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xl,hl,xlgridl,xj,hj,xmin);
                    break;
                

                default :
                    printf("missing AtpintipiT2mix for kflux > 1 \n");
                    // exit(-1);
                
            }

        }else{
            switch (kflux) {
                case 1:
                    res2 = AtpintipiT2_k1(ap0,ap1,app0,app1,ak1,ai0,ai1,aip0,aip1,xlp,hlp,xlpgridl,xl,hl,xlgridl,xj,hj,xmin);
                    break;
                

                default :
                    printf("missing AtpintipiT2 for kflux > 1 \n");
                    // exit(-1);
                
            }
        }
    }else{
        res2 = 0;
    }

    return hl*hlp*(res1+res2)/((flt) 4);




}
    
	