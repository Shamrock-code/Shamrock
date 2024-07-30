//-------------------------------------------
// Function for Limiter for DG scheme (Zhang,Shu 2010)
// To compute gamma coefficient in Eq.28
// Equations refere to Lombart & Laibe (2020) (doi:10.1093/mnras/staa3682)
//-------------------------------------------
#include "options.hpp"
#include <math.h>
#include "polynomials_legendre.hpp"
#include "reconstruction_gh.hpp"
#include "limiter.hpp"
#include <iostream>


//-------------------------------------------
//+
//  Minimum value of a general polynomials in interval [xmin,xmax]
//  which depends only on the coefficient of the polynomials
//  ONLY for polynomials until order 1
//  Generated from Mathematica
//+
//-------------------------------------------
flt minvalpolk1(flt a,flt b,flt xmin,flt xmax){

    if (xmin > ((flt)0) && xmax > xmin && b <= ((flt)0)) {
        return a+b*xmax;

    }else if (xmin > ((flt)0) && xmax > xmin && b > ((flt)0)) {
        return a+b*xmin;

    }else{
        return 0;
    }


}

//-------------------------------------------
//+
//  Minimum value of a general polynomials in interval [xmin,xmax]
//  which depends only on the coefficient of the polynomials
//  ONLY for polynomials until order 2
//  Generated from Mathematica
//+
//-------------------------------------------
flt minvalpolk2(flt a,flt b,flt c,flt xmin,flt xmax){

    if ((xmin > ((flt)0) && xmax == ((flt)2)*xmin && b == ((flt)0) && c == ((flt)0)) || 
        (xmin > ((flt)0) && xmax == ((flt)2)*xmin && b > ((flt)0) && c == -(b/(((flt)2)*xmin))) || 
        (xmin > ((flt)0) && xmax == ((flt)2)*xmin && b < ((flt)0) && c == -(b/xmin)) || 
        (xmin > ((flt)0) && xmax > ((flt)2)*xmin && b == ((flt)0) && c == ((flt)0)) || 
        (xmin > ((flt)0) && xmax > ((flt)2)*xmin && b > ((flt)0) && c == -(b/xmax)) || 
        (xmin > ((flt)0) && xmax > ((flt)2)*xmin && b < ((flt)0) && c == -(b/xmin)) || 
        (xmin > ((flt)0) && xmin < xmax && xmax < ((flt)2)*xmin && b > ((flt)0) && c == -(b/xmax))) {

        return a;

    }else if (  (xmin > ((flt)0) && xmax == ((flt)2)*xmin && b < ((flt)0) && -(b/(((flt)2)*xmax)) < c && c < -(b/(((flt)2)*xmin))) || 
                (xmin > ((flt)0) && xmax > ((flt)2)*xmin && b < ((flt)0) && -(b/(((flt)2)*xmax)) < c && c < -(b/(((flt)2)*xmin))) || 
                (xmin > ((flt)0) && xmin < xmax && xmax < ((flt)2)*xmin && b < ((flt)0) && -(b/(((flt)2)*xmax)) < c && c <= -(b/(((flt)2)*xmin)))) {
        return (((flt)4)*a*c-pow(b,2))/(((flt)4)*c);

    }else if ((xmin > ((flt)0) && xmax == ((flt)2)*xmin && b == ((flt)0) && c < ((flt)0)) || 
              (xmin > ((flt)0) && xmax > ((flt)2)*xmin && b == ((flt)0) && c < ((flt)0)) || 
              (xmin > ((flt)0) && xmin < xmax && xmax < ((flt)2)*xmin && b == ((flt)0) && c < ((flt)0))){

        return a+c*pow(xmax,2);
        
    }else if (  (xmin > ((flt)0) && xmax == ((flt)2)*xmin && b > ((flt)0) && -(b/(((flt)2)*xmin)) < c && c < -(b/(xmax + xmin))) || 
                (xmin > ((flt)0) && xmax == ((flt)2)*xmin && b > ((flt)0) && c < -(b/(((flt)2)*xmin))) || 
                (xmin > ((flt)0) && xmax == ((flt)2)*xmin && b < ((flt)0) && c <= -(b/(((flt)2)*xmax))) || 
                (xmin > ((flt)0) && xmax > ((flt)2)*xmin && b > ((flt)0) && -(b/xmax) < c && c < -(b/(xmax + xmin))) || 
                (xmin > ((flt)0) && xmax > ((flt)2)*xmin && b > ((flt)0) && c < -(b/xmax)) || 
                (xmin > ((flt)0) && xmax > ((flt)2)*xmin && b < ((flt)0) && c <= -(b/(((flt)2)*xmax))) || 
                (xmin > ((flt)0) && xmin < xmax && xmax < ((flt)2)*xmin && b > ((flt)0) && -(b/xmax) < c && c < -(b/(xmax + xmin))) || 
                (xmin > ((flt)0) && xmin < xmax && xmax < ((flt)2)*xmin && b > ((flt)0) && c < -(b/xmax)) || 
                (xmin > ((flt)0) && xmin < xmax && xmax < ((flt)2)*xmin && b < ((flt)0) && c <= -(b/(((flt)2)*xmax)))) {
        
        return a + b*xmax + c*pow(xmax,2);

    }else if ((xmin > ((flt)0) && xmax == ((flt)2)*xmin && b == ((flt)0) && c > ((flt)0)) || 
              (xmin > ((flt)0) && xmax > ((flt)2)*xmin && b == ((flt)0) && c > ((flt)0)) || 
              (xmin > ((flt)0) && xmin < xmax && xmax < ((flt)2)*xmin && b == ((flt)0) && c >= 0)) {

        return a + c*pow(xmin,2);

    }else if ((xmin > ((flt)0) && xmax == ((flt)2)*xmin && b > ((flt)0) && c >= -(b/(xmax + xmin))) || 
              (xmin > ((flt)0) && xmax == ((flt)2)*xmin && b < ((flt)0) && c > -(b/xmin)) || 
              (xmin > ((flt)0) && xmax == ((flt)2)*xmin && b < ((flt)0) && -(b/(((flt)2)*xmin)) <= c && c < -(b/xmin)) || 
              (xmin > ((flt)0) && xmax > ((flt)2)*xmin && b > ((flt)0) && c >= -(b/(xmax + xmin))) || 
              (xmin > ((flt)0) && xmax > ((flt)2)*xmin && b < ((flt)0) && c > -(b/xmin)) || 
              (xmin > ((flt)0) && xmax > ((flt)2)*xmin && b < ((flt)0) && -(b/(((flt)2)*xmin)) <= c && c < -(b/xmin)) || 
              (xmin > ((flt)0) && xmin < xmax && xmax < ((flt)2)*xmin && b > ((flt)0) && c >= -(b/(xmax + xmin))) || 
              (xmin > ((flt)0) && xmin < xmax && xmax < ((flt)2)*xmin && b < ((flt)0) && c > -(b/(((flt)2)*xmin)))) {

        return a + b*xmin + c*pow(xmin,2);

    }else{
        return 0;
    }
}

//-------------------------------------------
//+
//  Minimum value of a general polynomials in interval [xmin,xmax]
//  which depends only on the coefficient of the polynomials
//  ONLY for polynomials until order 3
//  Generated from Mathematica
//+
//-------------------------------------------
flt minvalpolk3(flt a,flt b,flt c,flt d,flt xmin,flt xmax){

    if ((d == ((flt)0) && c == ((flt)0) && b == ((flt)0) && xmin > ((flt)0) && xmax > xmin) || 
        (d == ((flt)0) && c > ((flt)0) && b < ((flt)0) && xmin == -(b/c) && xmax > xmin) || 
        (d == ((flt)0) && c < ((flt)0) && b > ((flt)0) && xmin == -(b/(((flt)2)*c)) && xmax == -(b/c)) || 
        (d == ((flt)0) && c < ((flt)0) && b > ((flt)0) && 0 < xmin && xmin < -(b/(((flt)2)*c)) && xmax == -(b/c)) || 
        (d == ((flt)0) && c < ((flt)0) && b > ((flt)0) && -(b/(((flt)2)*c)) < xmin && xmin < -(b/c) && xmax == -(b/c)) || 
        (d > ((flt)0) && c >= 0 && b < ((flt)0) && xmin == -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > xmin) || 
        (d > ((flt)0) && c < ((flt)0) && b == ((flt)0) && xmin == (((flt)2)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) && xmax > xmin) || 
        (d > ((flt)0) && c < ((flt)0) && b == pow(c,2)/(((flt)4)*d) && xmin == -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > xmin) || 
        (d > ((flt)0) && c < ((flt)0) && b == pow(c,2)/(((flt)4)*d) && 0 < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax >= -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
        (d > ((flt)0) && c < ((flt)0) && b == pow(c,2)/(((flt)4)*d) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax >= -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
        (d > ((flt)0) && c < ((flt)0) && 0 < b && b < pow(c,2)/(((flt)4)*d) && xmin == -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > xmin) || 
        (d > ((flt)0) && c < ((flt)0) && 0 < b && b < pow(c,2)/(((flt)4)*d) && 0 < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax == -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
        (d > ((flt)0) && c < ((flt)0) && 0 < b && b < pow(c,2)/(((flt)4)*d) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) &&  xmax == -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
        (d > ((flt)0) && c < ((flt)0) && b < ((flt)0) && xmin == -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > xmin) || 
        (d < ((flt)0) && c > ((flt)0) && b == ((flt)0) && 0 < xmin && xmin < (((flt)1)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) && xmax == (((flt)2)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d)) || 
        (d < ((flt)0) && c > ((flt)0) && b == ((flt)0) && (((flt)1)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) <= xmin && xmin < (((flt)2)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) && xmax == (((flt)2)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d)) || 
        (d < ((flt)0) && c > ((flt)0) && b > ((flt)0) && 0 < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax == -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
        (d < ((flt)0) && c > ((flt)0) && b > ((flt)0) && -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax == -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
        (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && xmin == -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmin < xmax && xmax <= -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
        (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax == -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
        (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax == -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
        (d < ((flt)0) && c <= 0 && b > ((flt)0) && 0 < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax == -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
        (d < ((flt)0) && c <= 0 && b > ((flt)0) && -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax == -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) ){
        return a;

    }else if (d == ((flt)0) && c > ((flt)0) && b < ((flt)0) && ((flt)0) < xmin && xmin < -(b/(((flt)2)*c)) && xmax > -(b/(((flt)2)*c))) {
        return (-pow(b,2) + ((flt)4)*a*c)/(((flt)4)*c);

    }else if (d > ((flt)0) && c < ((flt)0) && b == ((flt)0) && ((flt)0) < xmin && xmin < sqrt(pow(c,2)/pow(d,2))/3. - c/(((flt)3)*d) && xmax > sqrt(pow(c,2)/pow(d,2))/3. - c/(((flt)3)*d)) {
        return (((flt)2)*pow(c,3) + ((flt)27)*a*pow(d,2) - ((flt)2)*sqrt(pow(c,6)/pow(d,4))*pow(d,2))/(((flt)27)*pow(d,2));

    }else if (  (d > ((flt)0) && c >= 0 && b < ((flt)0) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && ((flt)0) < b && b < pow(c,2)/(((flt)4)*d) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && ((flt)0) < b && b < pow(c,2)/(((flt)4)*d) && -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && ((flt)0) < b && b < pow(c,2)/(((flt)4)*d) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < pow(c,2)/(((flt)3)*d) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < pow(c,2)/(((flt)3)*d) && -(c/(((flt)3)*d)) - (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2))) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && b < ((flt)0) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && b == pow(c,2)/(((flt)4)*d) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) < xmax && xmax < -(c/(((flt)3)*d)) + (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2)))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) < xmax && xmax < -(c/(((flt)3)*d)) + (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2)))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)3)*d) < b && b < pow(c,2)/(((flt)4)*d) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) < xmax && xmax < -(c/(((flt)3)*d)) + (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2))))) {

        return (((flt)2)*pow(c,3) - ((flt)9)*b*c*d + ((flt)27)*a*pow(d,2) - ((flt)2)*pow(d,2)*sqrt(pow((pow(c,2) - ((flt)3)*b*d),3)/pow(d,4)))/(((flt)27)*pow(d,2));

    }else if (d == ((flt)0) && c == ((flt)0) && b < ((flt)0) && xmin > ((flt)0) && xmax > xmin) {
        return a+b*xmax;

    }else if (  (d == ((flt)0) && c > ((flt)0) && b < ((flt)0) && ((flt)0) < xmin && xmin < -(b/(((flt)2)*c)) && xmin < xmax && xmax <= -(b/(((flt)2)*c))) || 
                (d == ((flt)0) && c < ((flt)0) && b > ((flt)0) && xmin == -(b/(((flt)2)*c)) && xmax > -(b/c)) || 
                (d == ((flt)0) && c < ((flt)0) && b > ((flt)0) && xmin == -(b/(((flt)2)*c)) && (-b - c*xmin)/c < xmax && xmax < -(b/c)) || 
                (d == ((flt)0) && c < ((flt)0) && b > ((flt)0) && xmin >= -(b/c) && xmax > xmin) || 
                (d == ((flt)0) && c < ((flt)0) && b > ((flt)0) && ((flt)0) < xmin && xmin < -(b/(((flt)2)*c)) && xmax > -(b/c)) || 
                (d == ((flt)0) && c < ((flt)0) && b > ((flt)0) && ((flt)0) < xmin && xmin < -(b/(((flt)2)*c)) && (-b - c*xmin)/c < xmax && xmax < -(b/c)) || 
                (d == ((flt)0) && c < ((flt)0) && b > ((flt)0) && -(b/(((flt)2)*c)) < xmin && xmin < -(b/c) && xmax > -(b/c)) || 
                (d == ((flt)0) && c < ((flt)0) && b > ((flt)0) && -(b/(((flt)2)*c)) < xmin && xmin < -(b/c) && xmin < xmax && xmax < -(b/c)) || 
                (d == ((flt)0) && c < ((flt)0) && b <= 0 && xmin > ((flt)0) && xmax > xmin)) {

        return a+b*xmax+c*pow(xmax,2);

    }else if ((d > ((flt)0) && c < ((flt)0) && b == ((flt)0) && ((flt)0) < xmin && xmin < (((flt)1)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) && xmin < xmax && xmax <= (((flt)1)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d)) || 
              (d < ((flt)0) && c > ((flt)0) && b == ((flt)0) && xmin >= (((flt)2)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) && xmax > xmin) || 
              (d < ((flt)0) && c > ((flt)0) && b == ((flt)0) && ((flt)0) < xmin && xmin < (((flt)1)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) && xmax > (((flt)2)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d)) || 
              (d < ((flt)0) && c > ((flt)0) && b == ((flt)0) && ((flt)0) < xmin && xmin < (((flt)1)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) && (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2)) <= xmax && xmax < (((flt)2)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d)) || 
              (d < ((flt)0) && c > ((flt)0) && b == ((flt)0) && (((flt)1)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) <= xmin && xmin < (((flt)2)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) && xmax > (((flt)2)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d)) || 
              (d < ((flt)0) && c > ((flt)0) && b == ((flt)0) && (((flt)1)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) <= xmin && xmin < (((flt)2)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) && xmin < xmax && xmax < (((flt)2)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d))) {
        
        return a +c*pow(xmax,2) + d*pow(xmax,3);

    }else if (  (d > ((flt)0) && c >= 0 && b < ((flt)0) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax <= -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && b == pow(c,2)/(((flt)4)*d) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && (-c - d*xmin)/(((flt)2)*d) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2)) <= xmax && xmax < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && b == pow(c,2)/(((flt)4)*d) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && ((flt)0) < b && b < pow(c,2)/(((flt)4)*d) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) < xmax && xmax <= -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && ((flt)0) < b && b < pow(c,2)/(((flt)4)*d) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && (-c - d*xmin)/(((flt)2)*d) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2)) <= xmax && xmax < -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && ((flt)0) < b && b < pow(c,2)/(((flt)4)*d) && -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax <= -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && ((flt)0) < b && b < pow(c,2)/(((flt)4)*d) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) < xmax && xmax <= -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && ((flt)0) < b && b < pow(c,2)/(((flt)4)*d) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmin < xmax && xmax < -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < pow(c,2)/(((flt)3)*d) && xmin == -(c/(((flt)3)*d)) - (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2))) && xmax == -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < pow(c,2)/(((flt)3)*d) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax <= -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < pow(c,2)/(((flt)3)*d) && -(c/(((flt)3)*d)) - (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2))) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && (-c - d*xmin)/(((flt)2)*d) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2)) <= xmax && xmax <= -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && b < ((flt)0) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax <= -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && b == pow(c,2)/(((flt)4)*d) && xmin == -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && b == pow(c,2)/(((flt)4)*d) && xmin >= -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > xmin) || 
                (d < ((flt)0) && c > ((flt)0) && b == pow(c,2)/(((flt)4)*d) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax >= -(c/(((flt)3)*d)) + (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2)))) || 
                (d < ((flt)0) && c > ((flt)0) && b == pow(c,2)/(((flt)4)*d) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax <= -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && b == pow(c,2)/(((flt)4)*d) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax >= (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && b > ((flt)0) && xmin >= -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > xmin) || 
                (d < ((flt)0) && c > ((flt)0) && b > ((flt)0) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && b > ((flt)0) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2)) <= xmax && xmax < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && b > ((flt)0) && -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && b > ((flt)0) && -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmin < xmax && xmax < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && xmin == -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && xmin == -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax >= -(c/(((flt)3)*d)) + (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2)))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && xmin >= -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > xmin) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax >= -(c/(((flt)3)*d)) + (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2)))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax <= -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2)) <= xmax && xmax < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) < xmin && xmin < -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax >= (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmin < xmax && xmax < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)3)*d) < b && b < pow(c,2)/(((flt)4)*d) && xmin == -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax >= -(c/(((flt)3)*d)) + (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2)))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)3)*d) < b && b < pow(c,2)/(((flt)4)*d) && xmin >= -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > xmin) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)3)*d) < b && b < pow(c,2)/(((flt)4)*d) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax >= -(c/(((flt)3)*d)) + (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2)))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)3)*d) < b && b < pow(c,2)/(((flt)4)*d) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax <= -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)3)*d) < b && b < pow(c,2)/(((flt)4)*d) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax >= (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && b <= pow(c,2)/(((flt)3)*d) && xmin > ((flt)0) && xmax > xmin) || 
                (d < ((flt)0) && c <= 0 && b > ((flt)0) && xmin >= -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > xmin) || 
                (d < ((flt)0) && c <= 0 && b > ((flt)0) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c <= 0 && b > ((flt)0) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2)) <= xmax && xmax < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c <= 0 && b > ((flt)0) && -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c <= 0 && b > ((flt)0) && -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmin < xmax && xmax < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2))) || 
                (d < ((flt)0) && c <= 0 && b <= 0 && xmin > ((flt)0) && xmax > xmin)) {

        return a + b*xmax + c*pow(xmax,2) + d*pow(xmax,3);

    }else if (d == ((flt)0) && c == ((flt)0) && b > ((flt)0) && xmin > ((flt)0) && xmax > xmin) {
        return a+b*xmin;

    }else if (  (d == ((flt)0) && c > ((flt)0) && b >= 0 && xmin > ((flt)0) && xmax > xmin) || 
                (d == ((flt)0) && c > ((flt)0) && b < ((flt)0) && xmin == -(b/(((flt)2)*c)) && xmax > (-b - c*xmin)/c) || 
                (d == ((flt)0) && c > ((flt)0) && b < ((flt)0) && xmin > -(b/c) && xmax > xmin) || 
                (d == ((flt)0) && c > ((flt)0) && b < ((flt)0) && -(b/(((flt)2)*c)) < xmin && xmin < -(b/c) && xmax > xmin) || 
                (d == ((flt)0) && c < ((flt)0) && b > ((flt)0) && ((flt)0) < xmin && xmin < -(b/(((flt)2)*c)) && xmin < xmax && xmax <= (-b - c*xmin)/c)) {
        
        return a + b*xmin + c*pow(xmin,2);


    }else if (  (d > ((flt)0) && c < ((flt)0) && b == ((flt)0) && xmin > (((flt)2)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) && xmax > xmin) || 
                (d > ((flt)0) && c < ((flt)0) && b == ((flt)0) && (((flt)1)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) <= xmin && xmin < (((flt)2)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) && xmax > xmin) || 
                (d < ((flt)0) && c > ((flt)0) && b == ((flt)0) && ((flt)0) < xmin && xmin < (((flt)1)/((flt)3))*sqrt(pow(c,2)/pow(d,2)) - c/(((flt)3)*d) && xmin < xmax && xmax < (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2)))) {

        return a + c*pow(xmin,2) + d*pow(xmin,3);

    }else if (  (d > ((flt)0) && c >= 0 && b >= 0 && xmin > ((flt)0) && xmax > xmin) || 
                (d > ((flt)0) && c >= 0 && b < ((flt)0) && xmin > -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > xmin) || 
                (d > ((flt)0) && c >= 0 && b < ((flt)0) && -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > xmin) || 
                (d > ((flt)0) && c < ((flt)0) && b == pow(c,2)/(((flt)4)*d) && xmin > -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > xmin) || 
                (d > ((flt)0) && c < ((flt)0) && b == pow(c,2)/(((flt)4)*d) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax < (-c - d*xmin)/(((flt)2)*d) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && b >= pow(c,2)/(((flt)3)*d) && xmin > ((flt)0) && xmax > xmin) || 
                (d > ((flt)0) && c < ((flt)0) && ((flt)0) < b && b < pow(c,2)/(((flt)4)*d) && xmin > -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > xmin) || 
                (d > ((flt)0) && c < ((flt)0) && ((flt)0) < b && b < pow(c,2)/(((flt)4)*d) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax < (-c - d*xmin)/(((flt)2)*d) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && ((flt)0) < b && b < pow(c,2)/(((flt)4)*d) && -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > xmin) || 
                (d > ((flt)0) && c < ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < pow(c,2)/(((flt)3)*d) && xmin == -(c/(((flt)3)*d)) - (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2))) && xmax > -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < pow(c,2)/(((flt)3)*d) && xmin == -(c/(((flt)3)*d)) - (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2))) && xmin < xmax && xmax < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < pow(c,2)/(((flt)3)*d) && xmin >= -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmax > xmin) || 
                (d > ((flt)0) && c < ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < pow(c,2)/(((flt)3)*d) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2))) && xmax > xmin) || 
                (d > ((flt)0) && c < ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < pow(c,2)/(((flt)3)*d) && -(c/(((flt)3)*d)) - (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2))) < xmin && xmin < -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax < (-c - d*xmin)/(((flt)2)*d) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2))) || 
                (d > ((flt)0) && c < ((flt)0) && b < ((flt)0) && xmin > -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > xmin) || 
                (d > ((flt)0) && c < ((flt)0) && b < ((flt)0) && -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) <= xmin && xmin < -(c/(((flt)2)*d)) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmax > xmin) || 
                (d < ((flt)0) && c > ((flt)0) && b == pow(c,2)/(((flt)4)*d) && xmin == -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax <= (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && b == pow(c,2)/(((flt)4)*d) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax < (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && b > ((flt)0) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax < (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && xmin == -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax < -(c/(((flt)3)*d)) + (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2)))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax < (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)4)*d) < b && b < ((flt)0) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) < xmin && xmin < -(c/(((flt)2)*d)) - (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d)/pow(d,2)) && xmin < xmax && xmax < (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)3)*d) < b && b < pow(c,2)/(((flt)4)*d) && xmin == -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax < -(c/(((flt)3)*d)) + (((flt)2)/((flt)3))*sqrt(-((-pow(c,2) + ((flt)3)*b*d)/pow(d,2)))) || 
                (d < ((flt)0) && c > ((flt)0) && pow(c,2)/(((flt)3)*d) < b && b < pow(c,2)/(((flt)4)*d) && -(c/(((flt)3)*d)) - (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax < (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2))) || 
                (d < ((flt)0) && c <= 0 && b > ((flt)0) && ((flt)0) < xmin && xmin < -(c/(((flt)3)*d)) + (((flt)1)/((flt)3))*sqrt((pow(c,2) - ((flt)3)*b*d)/pow(d,2)) && xmin < xmax && xmax < (-c - d*xmin)/(((flt)2)*d) + (((flt)1)/((flt)2))*sqrt((pow(c,2) - ((flt)4)*b*d - ((flt)2)*c*d*xmin - ((flt)3)*pow(d,2)*pow(xmin,2))/pow(d,2)))) {

        return a +b*xmin + c*pow(xmin,2) + d*pow(xmin,3);
    }else{
        return 0;
    }

}



//-------------------------------------------
//+
//  Minimum value for polynomial g in Eq.28
//+
//-------------------------------------------
void minvalgh(u32 i,u32 kflux,u32 nbins,const accfltr_t massgrid,const accfltr_t massbins,const accfltrw_t gij,const accfltrw_t coeff_Leg,const accfltrw_t coeff_gh,const accfltrw_t tabminvalgh){
	flt a;flt b;flt c;flt d;
   flt xjgridr; flt xjgridl;

	compute_coeff_gh(i,kflux,nbins,massgrid,massbins,gij,coeff_Leg,coeff_gh);

   for (u32 j=0;j<=nbins-1;j++){
      xjgridl = massgrid[j];
      xjgridr = massgrid[j+1];
      
      a = coeff_gh[0+j*4+i*4*nbins];
      b = coeff_gh[1+j*4+i*4*nbins];
      c = coeff_gh[2+j*4+i*4*nbins];
      d = coeff_gh[3+j*4+i*4*nbins];

      
      if (kflux==1) {
         tabminvalgh[j+i*nbins] = minvalpolk1(a,b,xjgridl,xjgridr);
      }else if(kflux==2){
         tabminvalgh[j+i*nbins] = minvalpolk2(a,b,c,xjgridl,xjgridr);
      }else if (kflux==3){
         tabminvalgh[j+i*nbins] = minvalpolk3(a,b,c,d,xjgridl,xjgridr);
      }else{
         tabminvalgh[j+i*nbins] = ((flt)0);
      }

   }
}




//-------------------------------------------
//+
// Limiter coefficient theta (Liu et al. 2019), https://doi.org/10.1137/17M1150360
// Equivalent to gamma coefficient in Eq.28 with m=0, M=infinity.
//+
//-------------------------------------------
void gammafunction(u32 i,u32 nbins,u32 kflux,const accfltr_t massgrid,const accfltr_t massbins,const accfltrw_t gij,const accfltrw_t coeff_Leg,const accfltrw_t coeff_gh,const accfltrw_t tabminvalgh,const accfltrw_t tabgamma){

   flt meangh;
   //Liu 2019
   if (kflux==0){
		for (u32 j=0;j<=nbins-1;j++){
			tabgamma[j+i*nbins] = ((flt)1);
		}
	}else if (0<kflux && kflux<=3){
		minvalgh(i,kflux,nbins,massgrid,massbins,gij,coeff_Leg,coeff_gh,tabminvalgh);
		for (u32 j=0;j<=nbins-1;j++){
			meangh = gij[0+j*(kflux+1)+i*nbins*(kflux+1)];

			tabgamma[j+i*nbins] = fmin(((flt)1),fabs(meangh/(meangh-tabminvalgh[j+i*nbins])));

		}
	}else{
		// printf("limiter.c -> gammafunction -> wrong polynomials order \n");
		exit(-1);
	}

}



