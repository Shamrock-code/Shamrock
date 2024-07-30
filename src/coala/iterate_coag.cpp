#include <tgmath.h>
#include "solver_DG.hpp"
#include "options.hpp"
#include "reconstruction_gh.hpp"
#include "generate_tabflux_tabintflux.hpp"
#include "compute_coag.hpp"
#include "L2proj_GQ.hpp"
#include <stdio.h>
#include <time.h>
#include "setup.hpp"
#include <stdlib.h>
#include "string.h"
#include "limiter.hpp"
#include "generate_gij.hpp"
#include <SYCL/sycl.hpp>



//----------------------------------------
// Function to compute coagulation for hydro timesteps
//----------------------------------------
void iterate_coag(u32 nparts, u32 nbins, u32 kflux, u32 kp, u32 Q,flt* massgrid, flt* massbins, flt* vecnodes, flt* vecweights, flt dthydro, u32 ndthydro, sycl::queue* queue){

    //null pointers
    //create pointer to buffer
    sycl::buffer<flt>* buf_massgrid = NULL; // dim = nbins+1
    sycl::buffer<flt>* buf_massbins = NULL; // dim = nbins
    sycl::buffer<flt>* buf_tabipiflux = NULL; // dim = nbins*nbins*nbins*(kflux+1)*(kflux+1)
    sycl::buffer<flt>* buf_tabipiAintflux = NULL; // dim = nbins*nbins*nbins*(kflux+1)*(kflux+1)*(kflux+1)
    sycl::buffer<flt>* buf_tabipiBintflux = NULL; // dim = nbins*nbins*(kflux+1)*(kflux+1)*(kflux+1)

    sycl::buffer<flt>* buf_tabK1F1 = NULL; // dim = nbins*(kp+1)
    sycl::buffer<flt>* buf_tabK2F1 = NULL; // dim = nbins*(kp+1)
    sycl::buffer<flt>* buf_tabK1F2 = NULL; // dim = nbins*(kp+1)
    sycl::buffer<flt>* buf_tabK2F2 = NULL; // dim = nbins*(kp+1)


    //create pointer to buffer for arrays need in solver_DG
    sycl::buffer<flt>* buf_coeff_Leg = NULL; // dim = 4*nparts
    sycl::buffer<flt>* buf_coeff_gh = NULL; // dim = nbins*4*nparts
    sycl::buffer<flt>* buf_tabminvalgh = NULL; // dim = nbins*nparts
    sycl::buffer<flt>* buf_tabgamma = NULL; // dim = nbins*nparts
    sycl::buffer<flt>* buf_tabdtCFL = NULL; // dim = nbins*nparts
    sycl::buffer<flt>* buf_tabdflux = NULL; // dim = nbins*nparts
    sycl::buffer<flt>* buf_flux = NULL; // dim = nbins*nparts
    sycl::buffer<flt>* buf_intflux = NULL; // dim = (kflux+1)*nbins*nparts
    sycl::buffer<flt>* buf_gij = NULL; //dim = nbins*(kflux+1)*nparts
    sycl::buffer<flt>* buf_gij1 = NULL; //dim = nbins*(kflux+1)*nparts
    sycl::buffer<flt>* buf_gij2 = NULL; //dim = nbins*(kflux+1)*nparts
    sycl::buffer<flt>* buf_Lk = NULL; //dim = nbins*(kflux+1)*nparts
    sycl::buffer<flt>* buf_Lk1 = NULL; //dim = nbins*(kflux+1)*nparts
    sycl::buffer<flt>* buf_Lk2 = NULL; //dim = nbins*(kflux+1)*nparts

    //host accessor
    sycl::host_accessor<flt>* hacc_gij = NULL; //dim = nbins*(kflux+1)*nparts


    //generate vectors for DG scheme
    flt* tabipiflux = new flt[nbins*nbins*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];
    flt* tabipiAintflux = new flt[nbins*nbins*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];
    flt* tabipiBintflux = new flt[nbins*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];

    compute_tabflux_tabintflux(nbins,kflux,kp,massgrid,massbins,tabipiflux,tabipiAintflux,tabipiBintflux);


    //generate vectors for collision kernel approximation (L2 projection)
    flt* tabK1F1 = new flt[nbins*(kp+1)];
    flt* tabK2F1 = new flt[nbins*(kp+1)];
    flt* tabK1F2 = new flt[nbins*(kp+1)];
    flt* tabK2F2 = new flt[nbins*(kp+1)];

    L2proj_kadd_GQ(nbins, kp, massgrid, massbins, Q,vecnodes,vecweights,
					tabK1F1, tabK2F1, tabK1F2, tabK2F2);



    //sycl with pointers
    buf_massgrid = new sycl::buffer<flt>(nbins+1);
    buf_massbins = new sycl::buffer<flt>(nbins);

    //create buffer for vectors collision kernel
    buf_tabK1F1 = new sycl::buffer<flt>(nbins*(kp+1));
    buf_tabK2F1 = new sycl::buffer<flt>(nbins*(kp+1));
    buf_tabK1F2 = new sycl::buffer<flt>(nbins*(kp+1));
    buf_tabK2F2 = new sycl::buffer<flt>(nbins*(kp+1));

    //create buffer for vectors DG scheme
    buf_tabipiflux = new sycl::buffer<flt>(nbins*nbins*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1));
    buf_tabipiAintflux = new sycl::buffer<flt>(nbins*nbins*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1));
    buf_tabipiBintflux = new sycl::buffer<flt>(nbins*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1));


    //write in buffer for all vectors
    auto massgrid_buf = buf_massgrid->get_access<sycl::access::mode::discard_write>();
    auto massbins_buf = buf_massbins->get_access<sycl::access::mode::discard_write>();

    auto tabK1F1_buf = buf_tabK1F1->get_access<sycl::access::mode::discard_write>();
    auto tabK2F1_buf = buf_tabK2F1->get_access<sycl::access::mode::discard_write>();
    auto tabK1F2_buf = buf_tabK1F2->get_access<sycl::access::mode::discard_write>();
    auto tabK2F2_buf = buf_tabK2F2->get_access<sycl::access::mode::discard_write>();

    auto tabipiflux_buf = buf_tabipiflux->get_access<sycl::access::mode::discard_write>();
    auto tabipiAintflux_buf = buf_tabipiAintflux->get_access<sycl::access::mode::discard_write>();
    auto tabipiBintflux_buf = buf_tabipiBintflux->get_access<sycl::access::mode::discard_write>();

    for (u32 j=0;j<nbins;j++){
        massgrid_buf[j] = massgrid[j];
        massbins_buf[j] = massbins[j];
        
        for (u32 lp=0;lp<=j;lp++){
            for (u32 l=0;l<nbins;l++){
                for (u32 ip=0;ip<=kflux;ip++){
                    for (u32 i=0; i<=kflux;i++){
                    	for (u32 pp=0;pp<=kp;pp++){
                        	for (u32 p=0;p<=kp;p++){
		                        tabipiflux_buf[p + pp*(kp+1) + i*(kp+1)*(kp+1) + ip*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)] 
		                        	= tabipiflux[p + pp*(kp+1) + i*(kp+1)*(kp+1) + ip*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];

		                        for (u32 k=0;k<=kflux;k++){
		                            if (j>0 && lp<j){
		                                tabipiAintflux_buf[p + pp*(kp+1) + k*(kp+1)*(kp+1) + i*(kflux+1)*(kp+1)*(kp+1) + ip*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)] 
		                                	= tabipiAintflux[p + pp*(kp+1) + k*(kp+1)*(kp+1) + i*(kflux+1)*(kp+1)*(kp+1) + ip*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + lp*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + j*nbins*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];

		                            }

		                            if (lp==0){
		                                tabipiBintflux_buf[p + pp*(kp+1) + k*(kp+1)*(kp+1) + i*(kflux+1)*(kp+1)*(kp+1) + ip*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) +  j*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)] 
		                                	= tabipiBintflux[p + pp*(kp+1) + k*(kp+1)*(kp+1) + i*(kflux+1)*(kp+1)*(kp+1) + ip*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) + l*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1) +  j*nbins*(kflux+1)*(kflux+1)*(kflux+1)*(kp+1)*(kp+1)];

		                            }


		                        }
		                    }
		                }
                    }
                }


            }
        }

        //tabs kernel
        for (u32 p=0;p<=kp;p++){
        	tabK1F1_buf[p+j*(kp+1)] = tabK1F1[p+j*(kp+1)];
        	tabK2F1_buf[p+j*(kp+1)] = tabK1F1[p+j*(kp+1)];
        	tabK1F2_buf[p+j*(kp+1)] = tabK1F1[p+j*(kp+1)];
        	tabK2F2_buf[p+j*(kp+1)] = tabK1F1[p+j*(kp+1)];
        }
    }



    //create buffer gij
    flt* gij_init = new flt[nparts*nbins*(kflux+1)];
    buf_gij = new sycl::buffer<flt>((kflux+1)*nbins*nparts);
    auto gij_buf = buf_gij->get_access<sycl::access::mode::discard_write>();
    generate_gij(nparts,nbins,kflux,Q,vecnodes,vecweights,massgrid,massbins,gij_init);

    for (u32 i=0;i<nparts;i++){
        for (u32 j=0;j<=nbins-1;j++){
            for (u32 k=0;k<=kflux;k++){
                gij_buf[k+j*(kflux+1)+i*nbins*(kflux+1)] = gij_init[k+j*(kflux+1)+i*nbins*(kflux+1)];
            }
        }
    }



    //create buffer for vectors needed in DG scheme (solver_DG.cpp)
    buf_coeff_Leg = new sycl::buffer<flt>(4*nparts);
    buf_coeff_gh = new sycl::buffer<flt>(4*nparts);
    buf_tabminvalgh = new sycl::buffer<flt>(nbins*nparts);
    buf_tabgamma = new sycl::buffer<flt>(nbins*nparts);
    buf_tabdflux = new sycl::buffer<flt>(nbins*nparts);
    buf_tabdtCFL = new sycl::buffer<flt>(nbins*nparts);
    buf_flux = new sycl::buffer<flt>(nbins*nparts);
    buf_intflux = new sycl::buffer<flt>((kflux+1)*nbins*nparts);
    buf_gij1 = new sycl::buffer<flt>((kflux+1)*nbins*nparts);
    buf_gij2 = new sycl::buffer<flt>((kflux+1)*nbins*nparts);
    buf_Lk = new sycl::buffer<flt>((kflux+1)*nbins*nparts);
    buf_Lk1 = new sycl::buffer<flt>((kflux+1)*nbins*nparts);
    buf_Lk2 = new sycl::buffer<flt>((kflux+1)*nbins*nparts);


    //in device
    sycl::range<1> range_npart{nparts};
    //iterate solver time
    flt time = ((flt)0);

    for (u32 j=0;j<ndthydro;j++){
        flt dt = dthydro;

        auto kernel_coag = [&] (sycl::handler &cgh){

            auto massgrid_k = buf_massgrid->get_access<sycl::access::mode::read>(cgh);
            auto massbins_k = buf_massbins->get_access<sycl::access::mode::read>(cgh);

            auto tabK1F1_k = buf_tabK1F1->get_access<sycl::access::mode::read>(cgh);
		    auto tabK2F1_k = buf_tabK2F1->get_access<sycl::access::mode::read>(cgh);
		    auto tabK1F2_k = buf_tabK1F2->get_access<sycl::access::mode::read>(cgh);
		    auto tabK2F2_k = buf_tabK2F2->get_access<sycl::access::mode::read>(cgh);

            auto tabipiflux_k = buf_tabipiflux->get_access<sycl::access::mode::read>(cgh);
            auto tabipiAintflux_k = buf_tabipiAintflux->get_access<sycl::access::mode::read>(cgh);
            auto tabipiBintflux_k = buf_tabipiBintflux->get_access<sycl::access::mode::read>(cgh);

            auto coeff_Leg = buf_coeff_Leg->get_access<sycl::access::mode::read_write>(cgh);
            auto coeff_gh = buf_coeff_gh->get_access<sycl::access::mode::read_write>(cgh);
            auto tabminvalgh = buf_tabminvalgh->get_access<sycl::access::mode::read_write>(cgh);
            auto tabgamma = buf_tabgamma->get_access<sycl::access::mode::read_write>(cgh);
            auto tabdtCFL = buf_tabdtCFL->get_access<sycl::access::mode::read_write>(cgh);
            auto tabdflux = buf_tabdflux->get_access<sycl::access::mode::read_write>(cgh);
            auto flux = buf_flux->get_access<sycl::access::mode::read_write>(cgh);
            auto intflux = buf_intflux->get_access<sycl::access::mode::read_write>(cgh);
            auto gij1 = buf_gij1->get_access<sycl::access::mode::read_write>(cgh);
            auto gij2 = buf_gij2->get_access<sycl::access::mode::read_write>(cgh);
            auto Lk = buf_Lk->get_access<sycl::access::mode::read_write>(cgh);
            auto Lk1 = buf_Lk1->get_access<sycl::access::mode::read_write>(cgh);
            auto Lk2 = buf_Lk2->get_access<sycl::access::mode::read_write>(cgh);

            auto gij = buf_gij->get_access<sycl::access::mode::read_write>(cgh);

            //command to display values in kernel
            // sycl::stream out(1024, 256, cgh);
            

            cgh.parallel_for<class coag>(range_npart, [=](sycl::item<1> item) {
                u32 i = item.get_id(0);

                compute_coag_kadd(i,nbins,kflux,kp,massgrid_k,massbins_k,
									gij,gij1,gij2,
			                        tabipiflux_k,tabipiAintflux_k,tabipiBintflux_k,
			                        tabK1F1_k,tabK2F1_k,tabK1F2_k,tabK2F2_k,
			                        flux,intflux,tabdflux,tabdtCFL,
			                        coeff_Leg,coeff_gh,tabminvalgh,tabgamma,
			                        Lk,Lk1,Lk2,
			                        dthydro);

                
            });
        };

        //submit task
        queue->submit(kernel_coag);

    }



    printf("Done");

    //read final data
    // hacc_gij = new sycl::host_accessor<flt>((kflux+1)*nbins*nparts);
    // auto acc_gij = hacc_gij->get_access<sycl::access::mode::read>();
    // auto h_gij = buf_gij->get_access<sycl::access::mode::read>();

    // printf("gij end p1\n");
    // for(u32 j=0;j<nbins;j++){
    //     for(u32 k=0;k<=kflux;k++){
    //          printf("%.4e, ",h_gij[k+j*(kflux+1)]);
    //     }
    //     printf("\n");
    // }


    //destroy pointers
    // _FREE(buf_massgrid);
    // _FREE(buf_massbins);

    // _FREE(buf_tabipiflux);
    // _FREE(buf_tabipiAintflux);
    // _FREE(buf_tabipiBintflux);

    // _FREE(buf_tabK1F1);
    // _FREE(buf_tabK2F1);
    // _FREE(buf_tabK1F2);
    // _FREE(buf_tabK2F2);

    // _FREE(buf_coeff_Leg);
    // _FREE(buf_coeff_gh);
    // _FREE(buf_tabminvalgh);
    // _FREE(buf_tabgamma);
    // _FREE(buf_flux);
    // _FREE(buf_intflux);
    // _FREE(buf_gij1);
    // _FREE(buf_gij2);
    // _FREE(buf_Lk);
    // _FREE(buf_Lk1);
    // _FREE(buf_Lk2);
    // _FREE(buf_gij);


    // delete queue;

}
