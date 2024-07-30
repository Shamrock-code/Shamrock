#include "options.hpp"
#include "setup.hpp"
#include "coala/init_grid.hpp"
#include "iterate_coag.hpp"
#include "GQ_legendre_nodes_weights.hpp"
#include "L2proj_GQ.hpp"
#include "generate_tabflux_tabintflux.hpp"
#include "polynomials_legendre.hpp"
#include "solver_DG.hpp"
#include <iostream>
#include <time.h>
#include <tgmath.h>
#include <string.h>

#include <SYCL/sycl.hpp>
#include "coala/coala.hpp"




//declare pointers
flt* massgrid;
flt* massbins;
flt* xmeanlog;
flt* vecnodes;
flt* vecweights;

//sycl definitions
sycl::queue* queue;

auto exception_handler = [] (sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch(sycl::exception const& e) {
            printf("Caught synchronous SYCL exception: %s\n",e.what());
        }
    }
};



//print sycl device
void init(){
    //initialisation sycl
    queue = new sycl::queue(sycl::host_selector(),exception_handler);
    // queue = new sycl::queue(sycl::cpu_selector());
    
    std::cout << "Device : "<< queue->get_device().get_info<sycl::info::device::name>() << "\n"; 
}



void main_coala()
{

    //setup
    constexpr u32 nbins = 20;     //number of dust bins
    constexpr u32 kflux = 1;      //order of polynomials for approximation of mass density
    constexpr u32 kp = 1;         //order of polynomials for approximation of collision kernel
    constexpr u32 nparts = 100;   //number of cells or particles in which COALA is run
    constexpr u32 Q = 5;          //number of points for Gauss quadrature method
    constexpr u32 ndthydro = 10;  //number of timestep ( hydro timestep)

    flt massmax = ((flt)1e6);     //maximum mass of particles to consider
    flt massmin = ((flt)1e-3);    //minimum mass of particles
    flt dthydro = ((flt)1e-2);    //hydro timestep
    // flt eps = 10e-40;

	// Using time point and system_clock
	clock_t start_tot, end_tot;
    clock_t start, end;
    flt elapsed_seconds;


	//path for saving data
    FILE *fp;
    char path_data[100];
    char path_massgrid[100];
    char path_log[100];

    init_path_files(nbins,kflux,scheme,kernel,path_data);
    strcpy(path_massgrid,path_data);
    strcat(path_massgrid,"massgrid.txt");

    strcpy(path_log,path_data);
    strcat(path_log,"log.txt");


    if(process)
    {
        printf("\033[96m>>>Setup<<<\033[0m \n");
        // if (typeid(flt).name() == typeid(float).name()){
        //     printf("Valuetype = FP32\n");
        // }else if (typeid(flt).name() == typeid(double).name()){
        //     printf("Valuetype = FP64\n");
        // }else if (typeid(flt).name() == typeid(__float128).name()){
		// 	printf("Valuetype = FP128\n");
		// }else{
        //     printf("Error in type of flt ! \n");
        //     exit(-1);
        // }

        switch (kernel) {
            case 0: //kconst
                printf("Kernel -> Kconst \n");
                break;
            case 1: //kadd
                printf("Kernel -> Kadd \n");
                break;
            case 2: //kmul
                printf("Kernel -> Kmul \n");
                break;
            case 3: //kballistic
                printf("Kernel -> Kballistic \n");
                break;
            
            default:
                printf("Need to chosse kernel in setup. \n");
                // exit(-1);
        }
        
        printf("Nbins = %d \n",nbins);
        printf("Kflux = %d \n",kflux);
        printf("NdtHydro = %d \n", ndthydro);
        printf("dtHydro = %f \n", dthydro);
        printf("MassMin = %f \n", massmin);
        printf("MassMax = %f \n", massmax);
        printf("Processes printing: %s \n",process ? "true" : "false");
        printf("Gij printing: %s \n",results ? "true" : "false");
        printf("Gij and gxmeanlog saving: %s \n",save ? "true" : "false");
        printf("\n");
    }

    

    //log simu
    if (save){
        fp = fopen(path_log,"w");
        switch (kernel) {
            case 0: //kconst
                fprintf(fp,"Kernel -> Kconst \n");
                break;
            case 1: //kadd
                fprintf(fp,"Kernel -> Kadd \n");
                break;
            case 2: //kmul
                fprintf(fp,"Kernel -> Kmul \n");
                break;
            case 3: //kballistic
                fprintf(fp,"Kernel -> Kballistic \n");
                break;
            
            default:
                printf("Need to chosse kernel in setup. \n");
                // exit(-1);
        }
        fprintf(fp,"Nbins = %d \n",nbins);
        fprintf(fp,"Kflux = %d \n",kflux);
        fprintf(fp,"dthydro = %.15e \n",dthydro);
        fprintf(fp,"ndthydro = %d \n",ndthydro);
        fprintf(fp,"nb Gauss points = %d \n",Q);
        fclose(fp);
    }

	
    start_tot = clock();

    if (process) printf("\033[96m>>>Generate arrays<<<\033[0m \n");
    //init massgrid and massbins
    start = clock();

    //allocate arrays
    massgrid = new flt[nbins+1];
    massbins = new flt[nbins];
    xmeanlog = new flt[nbins];
    coala::init_grid(massmin,massmax,nbins,massgrid,massbins,xmeanlog);

    // massgrid = new flt[nbins+1];
    // massbins = new flt[nbins];
    // xmeanlog = new flt[nbins];

    // for(unsigned int j=0;j<=nbins;j++){
    //     massgrid[j] = ((flt)massgrid_ld[j]);
    //     if (j<=nbins-1){
    //         massbins[j] = ((flt)massbins_ld[j]);
    //         xmeanlog[j] = ((flt)xmeanlog_ld[j]);

    //     }
    // }


    //Gauss-Legendre quadrature data
    vecnodes   = new flt[Q];
    vecweights = new flt[Q];
    GQLeg_nodes(Q,vecnodes);
    GQLeg_weights(Q,vecweights);


    end = clock();
    elapsed_seconds = (double)(end - start) / CLOCKS_PER_SEC;
    if (process){
        printf("Init massgrid and massbins in %.3e s \n",elapsed_seconds);
        // for (unsigned int j=0;j<nbins+1;j++){
        //     printf("j=%d, %.15e \n",j,massgrid[j]);
        // }
    }

    // printf("path_massgrid=%s \n",path_massgrid);

    if (save){
        fp = fopen(path_massgrid,"w");
        for (unsigned int j=0;j<nbins+1;j++){
            fprintf(fp,"%.15e\n",massgrid[j]);
        }
        
        fclose(fp);
    }


    //run COALA
    start = clock();
    if (kflux<=3){        
        init();
        iterate_coag(nparts,nbins,kflux,kp,Q,massgrid,massbins,vecnodes,vecweights,dthydro,ndthydro,queue);

    }else{
        printf("coala.cpp ->  need kflux<=3 \n");
        // exit(-1);
    }
    end = clock();
    elapsed_seconds = (double)(end - start) / CLOCKS_PER_SEC;
    if (process){
        printf("\n");
        printf("Iterate coag in %.3e s \n",elapsed_seconds);
    }

    end_tot = clock();
    elapsed_seconds = (double)(end - start) / CLOCKS_PER_SEC;
    if(process)
    {
        printf("\n");
        printf("\033[96m>>>Compute Time<<<\033[0m \n");
        printf("Elapsed Time : %.3e s\n", elapsed_seconds);
    }



}
