#include "coala.hpp"
#include <cmath>

namespace coala {

    /**
     * @brief init the grid mass bins
     * \todo should not use pointer
     * @param massmin 
     * @param massmax 
     * @param nbins 
     * @param massgrid 
     * @param massbins 
     * @param xmeanlog 
     */
    inline void init_grid(flt massmin, flt massmax,u32 nbins,flt* massgrid,flt* massbins, flt* xmeanlog){
        flt r = pow(massmax/massmin,((flt)1)/((flt)nbins));
        massgrid[0] = massmin;
        for (unsigned int j=0;j<nbins;j++){
            massgrid[j+1] = r*massgrid[j];
        }
        for (unsigned int j=0;j<nbins;j++){
            massbins[j] = (massgrid[j+1]+massgrid[j])/((flt)2);
            xmeanlog[j] = sqrt(massgrid[j+1]*massgrid[j]);
        }
    }

}