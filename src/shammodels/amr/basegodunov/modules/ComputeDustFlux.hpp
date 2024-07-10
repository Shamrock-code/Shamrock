// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file riemann_dust.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief This file contain states and Riemann solvers for dust
 */


 #include "shambase/sycl_utils.hpp"
#include "shammodels/amr/basegodunov/Solver.hpp"
 #include "shammath/riemann_dust.hpp"
#include <hipSYCL/sycl/handler.hpp>
#include <hipSYCL/sycl/queue.hpp>
#include <hipSYCL/sycl/sycl.hpp>

 enum Direction{
    xp = 0,
    xm = 1,
    yp = 2,
    ym = 3,
    zp = 4,
    zm = 5
 };



 using DustRiemannSolverMode =shammodels::basegodunov::DustRiemannSolverMode;

 template<class Tvec, DustRiemannSolverMode mode, Direction dir>
 class DustFluxCompute{
    public:
    using Tcons = shammath::DustConsState<Tvec>;
    using Tscal = typename Tcons::Tscal;

    inline static Tcons flux(Tcons cL, Tcons cR){
        if constexpr (mode == DustRiemannSolverMode::HB){
            if constexpr (dir == xp){
                    return shammath::huang_bai_flux_x(cL,cR);
            }
            if constexpr (dir == yp){
                    return shammath::huang_bai_flux_y(cL,cR);
            }
            if constexpr (dir == zp){
                    return shammath::huang_bai_flux_z(cL,cR);
            }

            if constexpr (dir == xm){
                    return shammath::huang_bai_flux_mx(cL,cR);
            }
            if constexpr (dir == ym){
                    return shammath::huang_bai_flux_my(cL,cR);
            }
            if constexpr (dir == zm){
                    return shammath::huang_bai_flux_mz(cL,cR);
            }
        }
        if constexpr (mode == DustRiemannSolverMode::DHLL){
            if constexpr (dir == xp){
                    return shammath::d_hll_flux_x(cL,cR);
            }
            if constexpr (dir == yp){
                    return shammath::d_hll_flux_y(cL,cR);
            }
            if constexpr (dir == zp){
                    return shammath::d_hll_flux_z(cL,cR);
            }

            if constexpr (dir == xm){
                    return shammath::d_hll_flux_mx(cL,cR);
            }
            if constexpr (dir == ym){
                    return shammath::d_hll_flux_my(cL,cR);
            }
            if constexpr (dir == zm){
                    return shammath::d_hll_flux_mz(cL,cR);
            }
        }
    }
 };

 template<DustRiemannSolverMode mode, class Tvec, class Tscal>
 void dust_compute_fluxes_xp(
     sycl::queue &q,
     u32 link_count,
     u32 nb_dust,
     sycl::buffer<std::vector<std::array<Tscal,2>>> &rho_dust_face_xp,
     sycl::buffer<std::vector<std::array<Tvec,2>>> &rhov_dust_face_xp,
     sycl::buffer<std::vector<Tscal>> &flux_rho_dust_face_xp,
     sycl::buffer<std::vector<Tvec>> &flux_rhov_dust_face_xp) {
    
    using DustFlux = DustFluxCompute<Tvec,mode, xp>;
    u32 bufs_size = link_count*nb_dust;
    sycl::buffer<std::array<Tscal,2>> rho_dust_buf(bufs_size);
    sycl::buffer<std::array<Tvec,2>> rhov_dust_buf(bufs_size);
    sycl::buffer<Tscal> flux_rho_dust_buf(bufs_size);
    sycl::buffer<Tvec> flux_rhov_dust_buf(bufs_size);

    // copy data into large buffer 
    for(u32 loc_id = 0; loc_id < nb_dust; loc_id++)
    {
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor rho_dust_face_xp_acc{rho_dust_face_xp, cgh, sycl::read_only};
            sycl::accessor rhov_dust_face_xp_acc{rhov_dust_face_xp, cgh, sycl::read_only};

            sycl::accessor rho_dust_buf_acc{rho_dust_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor rhov_dust_buf_acc{rhov_dust_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{link_count}, [=](sycl::item<1> gid){
                u64 lin_id = link_count * loc_id + gid;
                rho_dust_buf_acc[lin_id]  = rho_dust_face_xp_acc[gid][loc_id];
                rhov_dust_buf_acc[lin_id] = rhov_dust_face_xp_acc[gid][loc_id];
            });
        });
    }

    // process datas in large buffer
    q.submit([&](sycl::handler &cgh) {
        sycl::accessor rho_dust_buf_acc{rho_dust_buf, cgh, sycl::read_only};
        sycl::accessor rhov_dust_buf_acc{rhov_dust_buf, cgh, sycl::read_only};

        sycl::accessor flux_rho_dust_buf_acc{flux_rho_dust_buf, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhov_dust_buf_acc{flux_rhov_dust_buf, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for(sycl::range<1>{bufs_size}, [=](sycl::item<1> id){
            auto rho_ij   = rho_dust_buf_acc[id];
            auto rhov_ij  = rhov_dust_buf_acc[id];

            using Tcons = shammath::DustConsState<Tvec>;

            auto d_flux_x = DustFlux::flux(
                Tcons{rho_ij[0], rhov_ij[0]},
                Tcons{rho_ij[1], rhov_ij[1]}) ;

            flux_rho_dust_buf_acc[id]  = d_flux_x.rho;
            flux_rhov_dust_buf_acc[id] = d_flux_x.rhovel;
        });
    });

    // copy datas back into the original flux buffer
    for(u32 loc_id = 0; loc_id < nb_dust; loc_id++)
    {
        q.submit([&](sycl::handler &cgh) {

            sycl::accessor flux_rho_dust_buf_acc{flux_rho_dust_buf, cgh, sycl::read_only};
            sycl::accessor flux_rhov_dust_buf_acc{flux_rhov_dust_buf, cgh, sycl::read_only};

            sycl::accessor flux_rho_dust{flux_rho_dust_face_xp, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor flux_rhov_dust{flux_rhov_dust_face_xp, cgh, sycl::write_only, sycl::no_init};
            
            cgh.parallel_for(sycl::range<1>{link_count}, [=](sycl::item<1> gid){
                u64 lin_id = link_count * loc_id + gid;
                flux_rho_dust[gid][loc_id] = flux_rho_dust_buf_acc[lin_id];
                flux_rhov_dust[gid][loc_id] = flux_rhov_dust_buf_acc[lin_id];
            });
        });
    }
    
}


 template<DustRiemannSolverMode mode, class Tvec, class Tscal>
 void dust_compute_fluxes_yp(
     sycl::queue &q,
     u32 link_count,
     u32 nb_dust,
     sycl::buffer<std::vector<std::array<Tscal,2>>> &rho_dust_face_yp,
     sycl::buffer<std::vector<std::array<Tvec,2>>> &rhov_dust_face_yp,
     sycl::buffer<std::vector<Tscal>> &flux_rho_dust_face_yp,
     sycl::buffer<std::vector<Tvec>> &flux_rhov_dust_face_yp) {
    
    using DustFlux = DustFluxCompute<Tvec,mode, yp>;
    u32 bufs_size = link_count*nb_dust;
    sycl::buffer<std::array<Tscal,2>> rho_dust_buf(bufs_size);
    sycl::buffer<std::array<Tvec,2>> rhov_dust_buf(bufs_size);
    sycl::buffer<Tscal> flux_rho_dust_buf(bufs_size);
    sycl::buffer<Tvec> flux_rhov_dust_buf(bufs_size);

    // copy data into large buffer 
    for(u32 loc_id = 0; loc_id < nb_dust; loc_id++)
    {
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor rho_dust_face_yp_acc{rho_dust_face_yp, cgh, sycl::read_only};
            sycl::accessor rhov_dust_face_yp_acc{rhov_dust_face_yp, cgh, sycl::read_only};

            sycl::accessor rho_dust_buf_acc{rho_dust_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor rhov_dust_buf_acc{rhov_dust_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{link_count}, [=](sycl::item<1> gid){
                u64 lin_id = link_count * loc_id + gid;
                rho_dust_buf_acc[lin_id]  = rho_dust_face_yp_acc[gid][loc_id];
                rhov_dust_buf_acc[lin_id] = rhov_dust_face_yp_acc[gid][loc_id];
            });
        });
    }

    // process datas in large buffer
    q.submit([&](sycl::handler &cgh) {
        sycl::accessor rho_dust_buf_acc{rho_dust_buf, cgh, sycl::read_only};
        sycl::accessor rhov_dust_buf_acc{rhov_dust_buf, cgh, sycl::read_only};

        sycl::accessor flux_rho_dust_buf_acc{flux_rho_dust_buf, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhov_dust_buf_acc{flux_rhov_dust_buf, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for(sycl::range<1>{bufs_size}, [=](sycl::item<1> id){
            auto rho_ij   = rho_dust_buf_acc[id];
            auto rhov_ij  = rhov_dust_buf_acc[id];

            using Tcons = shammath::DustConsState<Tvec>;

            auto d_flux_y = DustFlux::flux(
                Tcons{rho_ij[0], rhov_ij[0]},
                Tcons{rho_ij[1], rhov_ij[1]}) ;

            flux_rho_dust_buf_acc[id]  = d_flux_y.rho;
            flux_rhov_dust_buf_acc[id] = d_flux_y.rhovel;
        });
    });

    // copy datas back into the original flux buffer
    for(u32 loc_id = 0; loc_id < nb_dust; loc_id++)
    {
        q.submit([&](sycl::handler &cgh) {

            sycl::accessor flux_rho_dust_buf_acc{flux_rho_dust_buf, cgh, sycl::read_only};
            sycl::accessor flux_rhov_dust_buf_acc{flux_rhov_dust_buf, cgh, sycl::read_only};

            sycl::accessor flux_rho_dust{flux_rho_dust_face_yp, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor flux_rhov_dust{flux_rhov_dust_face_yp, cgh, sycl::write_only, sycl::no_init};
            
            cgh.parallel_for(sycl::range<1>{link_count}, [=](sycl::item<1> gid){
                u64 lin_id = link_count * loc_id + gid;
                flux_rho_dust[gid][loc_id] = flux_rho_dust_buf_acc[lin_id];
                flux_rhov_dust[gid][loc_id] = flux_rhov_dust_buf_acc[lin_id];
            });
        });
    }
}



 template<DustRiemannSolverMode mode, class Tvec, class Tscal>
 void dust_compute_fluxes_zp(
     sycl::queue &q,
     u32 link_count,
     u32 nb_dust,
     sycl::buffer<std::vector<std::array<Tscal,2>>> &rho_dust_face_zp,
     sycl::buffer<std::vector<std::array<Tvec,2>>> &rhov_dust_face_zp,
     sycl::buffer<std::vector<Tscal>> &flux_rho_dust_face_zp,
     sycl::buffer<std::vector<Tvec>> &flux_rhov_dust_face_zp) {
    
    using DustFlux = DustFluxCompute<Tvec,mode, zp>;
    u32 bufs_size = link_count*nb_dust;
    sycl::buffer<std::array<Tscal,2>> rho_dust_buf(bufs_size);
    sycl::buffer<std::array<Tvec,2>> rhov_dust_buf(bufs_size);
    sycl::buffer<Tscal> flux_rho_dust_buf(bufs_size);
    sycl::buffer<Tvec> flux_rhov_dust_buf(bufs_size);

    // copy data into large buffer 
    for(u32 loc_id = 0; loc_id < nb_dust; loc_id++)
    {
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor rho_dust_face_zp_acc{rho_dust_face_zp, cgh, sycl::read_only};
            sycl::accessor rhov_dust_face_zp_acc{rhov_dust_face_zp, cgh, sycl::read_only};

            sycl::accessor rho_dust_buf_acc{rho_dust_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor rhov_dust_buf_acc{rhov_dust_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{link_count}, [=](sycl::item<1> gid){
                u64 lin_id = link_count * loc_id + gid;
                rho_dust_buf_acc[lin_id]  = rho_dust_face_zp_acc[gid][loc_id];
                rhov_dust_buf_acc[lin_id] = rhov_dust_face_zp_acc[gid][loc_id];
            });
        });
    }

    // process datas in large buffer
    q.submit([&](sycl::handler &cgh) {
        sycl::accessor rho_dust_buf_acc{rho_dust_buf, cgh, sycl::read_only};
        sycl::accessor rhov_dust_buf_acc{rhov_dust_buf, cgh, sycl::read_only};

        sycl::accessor flux_rho_dust_buf_acc{flux_rho_dust_buf, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhov_dust_buf_acc{flux_rhov_dust_buf, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for(sycl::range<1>{bufs_size}, [=](sycl::item<1> id){
            auto rho_ij   = rho_dust_buf_acc[id];
            auto rhov_ij  = rhov_dust_buf_acc[id];

            using Tcons = shammath::DustConsState<Tvec>;

            auto d_flux_z = DustFlux::flux(
                Tcons{rho_ij[0], rhov_ij[0]},
                Tcons{rho_ij[1], rhov_ij[1]}) ;

            flux_rho_dust_buf_acc[id]  = d_flux_z.rho;
            flux_rhov_dust_buf_acc[id] = d_flux_z.rhovel;
        });
    });

    // copy datas back into the original flux buffer
    for(u32 loc_id = 0; loc_id < nb_dust; loc_id++)
    {
        q.submit([&](sycl::handler &cgh) {

            sycl::accessor flux_rho_dust_buf_acc{flux_rho_dust_buf, cgh, sycl::read_only};
            sycl::accessor flux_rhov_dust_buf_acc{flux_rhov_dust_buf, cgh, sycl::read_only};

            sycl::accessor flux_rho_dust{flux_rho_dust_face_zp, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor flux_rhov_dust{flux_rhov_dust_face_zp, cgh, sycl::write_only, sycl::no_init};
            
            cgh.parallel_for(sycl::range<1>{link_count}, [=](sycl::item<1> gid){
                u64 lin_id = link_count * loc_id + gid;
                flux_rho_dust[gid][loc_id] = flux_rho_dust_buf_acc[lin_id];
                flux_rhov_dust[gid][loc_id] = flux_rhov_dust_buf_acc[lin_id];
            });
        });
    }
    
}




 template<DustRiemannSolverMode mode, class Tvec, class Tscal>
 void dust_compute_fluxes_xm(
     sycl::queue &q,
     u32 link_count,
     u32 nb_dust,
     sycl::buffer<std::vector<std::array<Tscal,2>>> &rho_dust_face_xm,
     sycl::buffer<std::vector<std::array<Tvec,2>>> &rhov_dust_face_xm,
     sycl::buffer<std::vector<Tscal>> &flux_rho_dust_face_xm,
     sycl::buffer<std::vector<Tvec>> &flux_rhov_dust_face_xm) {
    
    using DustFlux = DustFluxCompute<Tvec,mode, xm>;
    u32 bufs_size = link_count*nb_dust;
    sycl::buffer<std::array<Tscal,2>> rho_dust_buf(bufs_size);
    sycl::buffer<std::array<Tvec,2>> rhov_dust_buf(bufs_size);
    sycl::buffer<Tscal> flux_rho_dust_buf(bufs_size);
    sycl::buffer<Tvec> flux_rhov_dust_buf(bufs_size);

    // copy data into large buffer 
    for(u32 loc_id = 0; loc_id < nb_dust; loc_id++)
    {
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor rho_dust_face_xm_acc{rho_dust_face_xm, cgh, sycl::read_only};
            sycl::accessor rhov_dust_face_xm_acc{rhov_dust_face_xm, cgh, sycl::read_only};

            sycl::accessor rho_dust_buf_acc{rho_dust_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor rhov_dust_buf_acc{rhov_dust_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{link_count}, [=](sycl::item<1> gid){
                u64 lin_id = link_count * loc_id + gid;
                rho_dust_buf_acc[lin_id]  = rho_dust_face_xm_acc[gid][loc_id];
                rhov_dust_buf_acc[lin_id] = rhov_dust_face_xm_acc[gid][loc_id];
            });
        });
    }

    // process datas in large buffer
    q.submit([&](sycl::handler &cgh) {
        sycl::accessor rho_dust_buf_acc{rho_dust_buf, cgh, sycl::read_only};
        sycl::accessor rhov_dust_buf_acc{rhov_dust_buf, cgh, sycl::read_only};

        sycl::accessor flux_rho_dust_buf_acc{flux_rho_dust_buf, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhov_dust_buf_acc{flux_rhov_dust_buf, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for(sycl::range<1>{bufs_size}, [=](sycl::item<1> id){
            auto rho_ij   = rho_dust_buf_acc[id];
            auto rhov_ij  = rhov_dust_buf_acc[id];

            using Tcons = shammath::DustConsState<Tvec>;

            auto d_flux_x = DustFlux::flux(
                Tcons{rho_ij[0], rhov_ij[0]},
                Tcons{rho_ij[1], rhov_ij[1]}) ;

            flux_rho_dust_buf_acc[id]  = d_flux_x.rho;
            flux_rhov_dust_buf_acc[id] = d_flux_x.rhovel;
        });
    });

    // copy datas back into the original flux buffer
    for(u32 loc_id = 0; loc_id < nb_dust; loc_id++)
    {
        q.submit([&](sycl::handler &cgh) {

            sycl::accessor flux_rho_dust_buf_acc{flux_rho_dust_buf, cgh, sycl::read_only};
            sycl::accessor flux_rhov_dust_buf_acc{flux_rhov_dust_buf, cgh, sycl::read_only};

            sycl::accessor flux_rho_dust{flux_rho_dust_face_xm, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor flux_rhov_dust{flux_rhov_dust_face_xm, cgh, sycl::write_only, sycl::no_init};
            
            cgh.parallel_for(sycl::range<1>{link_count}, [=](sycl::item<1> gid){
                u64 lin_id = link_count * loc_id + gid;
                flux_rho_dust[gid][loc_id] = flux_rho_dust_buf_acc[lin_id];
                flux_rhov_dust[gid][loc_id] = flux_rhov_dust_buf_acc[lin_id];
            });
        });
    }
    
}


 template<DustRiemannSolverMode mode, class Tvec, class Tscal>
 void dust_compute_fluxes_ym(
     sycl::queue &q,
     u32 link_count,
     u32 nb_dust,
     sycl::buffer<std::vector<std::array<Tscal,2>>> &rho_dust_face_ym,
     sycl::buffer<std::vector<std::array<Tvec,2>>> &rhov_dust_face_ym,
     sycl::buffer<std::vector<Tscal>> &flux_rho_dust_face_ym,
     sycl::buffer<std::vector<Tvec>> &flux_rhov_dust_face_ym) {
    
    using DustFlux = DustFluxCompute<Tvec,mode, ym>;
    u32 bufs_size = link_count*nb_dust;
    sycl::buffer<std::array<Tscal,2>> rho_dust_buf(bufs_size);
    sycl::buffer<std::array<Tvec,2>> rhov_dust_buf(bufs_size);
    sycl::buffer<Tscal> flux_rho_dust_buf(bufs_size);
    sycl::buffer<Tvec> flux_rhov_dust_buf(bufs_size);

    // copy data into large buffer 
    for(u32 loc_id = 0; loc_id < nb_dust; loc_id++)
    {
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor rho_dust_face_ym_acc{rho_dust_face_ym, cgh, sycl::read_only};
            sycl::accessor rhov_dust_face_ym_acc{rhov_dust_face_ym, cgh, sycl::read_only};

            sycl::accessor rho_dust_buf_acc{rho_dust_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor rhov_dust_buf_acc{rhov_dust_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{link_count}, [=](sycl::item<1> gid){
                u64 lin_id = link_count * loc_id + gid;
                rho_dust_buf_acc[lin_id]  = rho_dust_face_ym_acc[gid][loc_id];
                rhov_dust_buf_acc[lin_id] = rhov_dust_face_ym_acc[gid][loc_id];
            });
        });
    }

    // process datas in large buffer
    q.submit([&](sycl::handler &cgh) {
        sycl::accessor rho_dust_buf_acc{rho_dust_buf, cgh, sycl::read_only};
        sycl::accessor rhov_dust_buf_acc{rhov_dust_buf, cgh, sycl::read_only};

        sycl::accessor flux_rho_dust_buf_acc{flux_rho_dust_buf, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhov_dust_buf_acc{flux_rhov_dust_buf, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for(sycl::range<1>{bufs_size}, [=](sycl::item<1> id){
            auto rho_ij   = rho_dust_buf_acc[id];
            auto rhov_ij  = rhov_dust_buf_acc[id];

            using Tcons = shammath::DustConsState<Tvec>;

            auto d_flux_y = DustFlux::flux(
                Tcons{rho_ij[0], rhov_ij[0]},
                Tcons{rho_ij[1], rhov_ij[1]}) ;

            flux_rho_dust_buf_acc[id]  = d_flux_y.rho;
            flux_rhov_dust_buf_acc[id] = d_flux_y.rhovel;
        });
    });

    // copy datas back into the original flux buffer
    for(u32 loc_id = 0; loc_id < nb_dust; loc_id++)
    {
        q.submit([&](sycl::handler &cgh) {

            sycl::accessor flux_rho_dust_buf_acc{flux_rho_dust_buf, cgh, sycl::read_only};
            sycl::accessor flux_rhov_dust_buf_acc{flux_rhov_dust_buf, cgh, sycl::read_only};

            sycl::accessor flux_rho_dust{flux_rho_dust_face_ym, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor flux_rhov_dust{flux_rhov_dust_face_ym, cgh, sycl::write_only, sycl::no_init};
            
            cgh.parallel_for(sycl::range<1>{link_count}, [=](sycl::item<1> gid){
                u64 lin_id = link_count * loc_id + gid;
                flux_rho_dust[gid][loc_id] = flux_rho_dust_buf_acc[lin_id];
                flux_rhov_dust[gid][loc_id] = flux_rhov_dust_buf_acc[lin_id];
            });
        });
    }
}



 template<DustRiemannSolverMode mode, class Tvec, class Tscal>
 void dust_compute_fluxes_zm(
     sycl::queue &q,
     u32 link_count,
     u32 nb_dust,
     sycl::buffer<std::vector<std::array<Tscal,2>>> &rho_dust_face_zm,
     sycl::buffer<std::vector<std::array<Tvec,2>>> &rhov_dust_face_zm,
     sycl::buffer<std::vector<Tscal>> &flux_rho_dust_face_zm,
     sycl::buffer<std::vector<Tvec>> &flux_rhov_dust_face_zm) {
    
    using DustFlux = DustFluxCompute<Tvec,mode, zm>;
    u32 bufs_size = link_count*nb_dust;
    sycl::buffer<std::array<Tscal,2>> rho_dust_buf(bufs_size);
    sycl::buffer<std::array<Tvec,2>> rhov_dust_buf(bufs_size);
    sycl::buffer<Tscal> flux_rho_dust_buf(bufs_size);
    sycl::buffer<Tvec> flux_rhov_dust_buf(bufs_size);

    // copy data into large buffer 
    for(u32 loc_id = 0; loc_id < nb_dust; loc_id++)
    {
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor rho_dust_face_zm_acc{rho_dust_face_zm, cgh, sycl::read_only};
            sycl::accessor rhov_dust_face_zm_acc{rhov_dust_face_zm, cgh, sycl::read_only};

            sycl::accessor rho_dust_buf_acc{rho_dust_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor rhov_dust_buf_acc{rhov_dust_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{link_count}, [=](sycl::item<1> gid){
                u64 lin_id = link_count * loc_id + gid;
                rho_dust_buf_acc[lin_id]  = rho_dust_face_zm_acc[gid][loc_id];
                rhov_dust_buf_acc[lin_id] = rhov_dust_face_zm_acc[gid][loc_id];
            });
        });
    }

    // process datas in large buffer
    q.submit([&](sycl::handler &cgh) {
        sycl::accessor rho_dust_buf_acc{rho_dust_buf, cgh, sycl::read_only};
        sycl::accessor rhov_dust_buf_acc{rhov_dust_buf, cgh, sycl::read_only};

        sycl::accessor flux_rho_dust_buf_acc{flux_rho_dust_buf, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhov_dust_buf_acc{flux_rhov_dust_buf, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for(sycl::range<1>{bufs_size}, [=](sycl::item<1> id){
            auto rho_ij   = rho_dust_buf_acc[id];
            auto rhov_ij  = rhov_dust_buf_acc[id];

            using Tcons = shammath::DustConsState<Tvec>;

            auto d_flux_z = DustFlux::flux(
                Tcons{rho_ij[0], rhov_ij[0]},
                Tcons{rho_ij[1], rhov_ij[1]}) ;

            flux_rho_dust_buf_acc[id]  = d_flux_z.rho;
            flux_rhov_dust_buf_acc[id] = d_flux_z.rhovel;
        });
    });

    // copy datas back into the original flux buffer
    for(u32 loc_id = 0; loc_id < nb_dust; loc_id++)
    {
        q.submit([&](sycl::handler &cgh) {

            sycl::accessor flux_rho_dust_buf_acc{flux_rho_dust_buf, cgh, sycl::read_only};
            sycl::accessor flux_rhov_dust_buf_acc{flux_rhov_dust_buf, cgh, sycl::read_only};

            sycl::accessor flux_rho_dust{flux_rho_dust_face_zm, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor flux_rhov_dust{flux_rhov_dust_face_zm, cgh, sycl::write_only, sycl::no_init};
            
            cgh.parallel_for(sycl::range<1>{link_count}, [=](sycl::item<1> gid){
                u64 lin_id = link_count * loc_id + gid;
                flux_rho_dust[gid][loc_id] = flux_rho_dust_buf_acc[lin_id];
                flux_rhov_dust[gid][loc_id] = flux_rhov_dust_buf_acc[lin_id];
            });
        });
    }
    
}