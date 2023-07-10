// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once


#include "shamrock/scheduler/scheduler_mpi.hpp"
#include "shamsys/legacy/log.hpp"

#include "shammodels/sph/legacy/integrators/leapfrog.hpp"
#include <string>

//%Impl status : Clean unfinished


namespace models::sph {

    template<class flt, class Kernel>
    class BasicSPHGasUInterne {

        using vec3 = sycl::vec<flt, 3>;

        using u_morton = u32;
        using Stepper = integrators::sph::LeapfrogGeneral<flt, Kernel, u_morton>;

        static constexpr flt htol_up_tol  = 1.4;
        static constexpr flt htol_up_iter = 1.2;

        flt cfl_cour  = -1;
        flt cfl_force = -1;
        flt gpart_mass = -1;

        //TODO change SimBoxInfo into boundary condition to extract it from the models, then we can throw errors if the BC is not implemented for the model
        static constexpr bool periodic_bc = true;

        static constexpr flt eos_cs = 1;

        void check_valid();

        public:

        void init();

        f64 evolve(PatchScheduler &sched, f64 current_time, f64 target_time);
        void dump(std::string prefix);
        void restart_dump(std::string prefix);

        inline f64 simulate_until(PatchScheduler &sched, f64 start_time, f64 end_time, u32 freq_dump, u32 freq_restart_dump,std::string prefix_dump){
            f64 step_time = start_time;

            u32 step_cnt = 0;

            

            while(step_time < end_time && sycl::fabs(step_time - end_time) > 1e-8){

                logger::normal_ln("BasicSPHGas", "simulate until",
                    shambase::format_printf("%2.2f / %2.2f (%3.1f %%)",step_time,end_time,100*(step_time-start_time)/(end_time-start_time))
                );

                if(step_cnt % freq_dump){
                    dump(prefix_dump + "dump_" + shambase::format_printf("%06d",step_cnt));
                }

                if(step_cnt % freq_restart_dump){
                    restart_dump(prefix_dump + "restart_dump_" + shambase::format_printf("%06d",step_cnt));
                }

                step_time = evolve(sched, step_time, end_time);
                step_cnt++;
            }


            return step_time;
        }

        void close();


        inline void set_cfl_cour(flt Ccour){cfl_cour = Ccour;}
        inline void set_cfl_force(flt Cforce){cfl_force = Cforce;}
        inline void set_particle_mass(flt pmass){gpart_mass = pmass;}

    };


} // namespace models::sph