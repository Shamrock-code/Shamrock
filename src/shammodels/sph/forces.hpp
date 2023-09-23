// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

//%Impl status : Should rewrite

#pragma once

namespace shamrock::sph {

    template<class vec, class flt>
    inline vec sph_pressure_symetric_av(
        flt m_b,
        flt rho_a_sq,
        flt rho_b_sq,
        flt P_a,
        flt P_b,
        flt omega_a,
        flt omega_b,
        flt qa_ab,
        flt qb_ab,
        vec nabla_Wab_ha,
        vec nabla_Wab_hb
    ) {

        flt sub_fact_a = rho_a_sq * omega_a;
        flt sub_fact_b = rho_b_sq * omega_b;

        vec acc_a = ((P_a + qa_ab) / (sub_fact_a)) * nabla_Wab_ha;
        vec acc_b = ((P_b + qb_ab) / (sub_fact_b)) * nabla_Wab_hb;

        if (sub_fact_a == 0)
            acc_a = {0, 0, 0};
        if (sub_fact_b == 0)
            acc_b = {0, 0, 0};

        return -m_b * (acc_a + acc_b);
    }

    template<class vec, class flt>
    inline vec sph_pressure_symetric(
        flt m_b,
        flt rho_a_sq,
        flt rho_b_sq,
        flt P_a,
        flt P_b,
        flt omega_a,
        flt omega_b,
        vec nabla_Wab_ha,
        vec nabla_Wab_hb
    ) {

        flt sub_fact_a = rho_a_sq * omega_a;
        flt sub_fact_b = rho_b_sq * omega_b;

        vec acc_a = ((P_a) / (sub_fact_a)) * nabla_Wab_ha;
        vec acc_b = ((P_b) / (sub_fact_b)) * nabla_Wab_hb;

        if (sub_fact_a == 0)
            acc_a = {0, 0, 0};
        if (sub_fact_b == 0)
            acc_b = {0, 0, 0};

        return -m_b * (acc_a + acc_b);
    }


} // namespace shamrock::sph