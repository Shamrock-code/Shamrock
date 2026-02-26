// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeComputeFlux.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/ramses/modules/ComputeFluxUtilities.hpp"
#include "shammodels/ramses/modules/NodeComputeFlux.hpp"

using RiemannSolverMode     = shammodels::basegodunov::RiemannSolverMode;
using DustRiemannSolverMode = shammodels::basegodunov::DustRiemannSolverMode;
using Direction             = shammodels::basegodunov::modules::Direction;

template<class Tvec, class TgridVec, RiemannSolverMode mode, Direction dir>
void shammodels::basegodunov::modules::NodeComputeFluxGasDirMode<Tvec, TgridVec, mode, dir>::
    _impl_evaluate_internal() {
    StackEntry stack_loc{};

    auto edges = get_edges();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    auto graphs_dir = edges.cell_neigh_graph.get_refs_dir(dir);

    edges.rho_face.check_size(graphs_dir);
    edges.vel_face.check_size(graphs_dir);
    edges.press_face.check_size(graphs_dir);

    edges.flux_rho_face.resize_according_to(graphs_dir);
    edges.flux_rhov_face.resize_according_to(graphs_dir);
    edges.flux_rhoe_face.resize_according_to(graphs_dir);

    shambase::DistributedData<u32> counts_dir
        = graphs_dir.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().link_count; //* ndust;
          });

    using Flux = FluxCompute<Tvec, mode, dir>;

    sham::distributed_data_kernel_call(
        dev_sched,
        sham::DDMultiRef{
            edges.rho_face.link_fields, edges.vel_face.link_fields, edges.press_face.link_fields},
        sham::DDMultiRef{
            edges.flux_rho_face.link_fields,
            edges.flux_rhov_face.link_fields,
            edges.flux_rhoe_face.link_fields},
        counts_dir,
        [gamma = this->gamma](
            u32 link_id,
            const std::array<Tscal, 2> *rho_face,
            const std::array<Tvec, 2> *vel_face,
            const std::array<Tscal, 2> *press_face,
            Tscal *flux_rho_face,
            Tvec *flux_rhov_face,
            Tscal *flux_rhoe_face) {
            auto rho_ij   = rho_face[link_id];
            auto vel_ij   = vel_face[link_id];
            auto press_ij = press_face[link_id];

            using Tprim   = shammath::PrimState<Tvec>;
            auto flux_dir = Flux::flux(
                Tprim{rho_ij[0], press_ij[0], vel_ij[0]},
                Tprim{rho_ij[1], press_ij[1], vel_ij[1]},
                gamma);

            flux_rho_face[link_id]  = flux_dir.rho;
            flux_rhov_face[link_id] = flux_dir.rhovel;
            flux_rhoe_face[link_id] = flux_dir.rhoe;
        });
}

template<class Tvec, class TgridVec, RiemannSolverMode mode, Direction dir>
std::string shammodels::basegodunov::modules::NodeComputeFluxGasDirMode<Tvec, TgridVec, mode, dir>::
    _impl_get_tex() const {
    return "TODO";
}

template<class Tvec, class TgridVec, DustRiemannSolverMode mode, Direction dir>
void shammodels::basegodunov::modules::NodeComputeFluxDustDirMode<Tvec, TgridVec, mode, dir>::
    _impl_evaluate_internal() {
    StackEntry stack_loc{};

    auto edges = get_edges();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    auto graphs_dir = edges.cell_neigh_graph.get_refs_dir(dir);

    edges.rho_face.check_size(graphs_dir);
    edges.vel_face.check_size(graphs_dir);

    edges.flux_rho_face.resize_according_to(graphs_dir);
    edges.flux_rhov_face.resize_according_to(graphs_dir);

    shambase::DistributedData<u32> counts_dir
        = graphs_dir.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().link_count * ndust;
          });

    using Flux = DustFluxCompute<Tvec, mode, dir>;

    sham::distributed_data_kernel_call(
        dev_sched,
        sham::DDMultiRef{edges.rho_face.link_fields, edges.vel_face.link_fields},
        sham::DDMultiRef{edges.flux_rho_face.link_fields, edges.flux_rhov_face.link_fields},
        counts_dir,
        [](u32 link_id,
           const std::array<Tscal, 2> *rho_face,
           const std::array<Tvec, 2> *vel_face,
           Tscal *flux_rho_face,
           Tvec *flux_rhov_face) {
            auto rho_ij = rho_face[link_id];
            auto vel_ij = vel_face[link_id];

            using Tprim = shammath::DustPrimState<Tvec>;
            auto flux_dust_dir
                = Flux::dustflux(Tprim{rho_ij[0], vel_ij[0]}, Tprim{rho_ij[1], vel_ij[1]});

            flux_rho_face[link_id]  = flux_dust_dir.rho;
            flux_rhov_face[link_id] = flux_dust_dir.rhovel;
        });
}

template<class Tvec, class TgridVec, DustRiemannSolverMode mode, Direction dir>
std::string shammodels::basegodunov::modules::
    NodeComputeFluxDustDirMode<Tvec, TgridVec, mode, dir>::_impl_get_tex() const {
    return "TODO";
}

template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::Rusanov, Direction::xm>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::Rusanov, Direction::xp>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::Rusanov, Direction::ym>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::Rusanov, Direction::yp>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::Rusanov, Direction::zm>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::Rusanov, Direction::zp>;

template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::HLL, Direction::xm>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::HLL, Direction::xp>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::HLL, Direction::ym>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::HLL, Direction::yp>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::HLL, Direction::zm>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::HLL, Direction::zp>;

template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::HLLC, Direction::xm>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::HLLC, Direction::xp>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::HLLC, Direction::ym>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::HLLC, Direction::yp>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::HLLC, Direction::zm>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxGasDirMode<f64_3, i64_3, RiemannSolverMode::HLLC, Direction::zp>;

template class shammodels::basegodunov::modules::
    NodeComputeFluxDustDirMode<f64_3, i64_3, DustRiemannSolverMode::HB, Direction::xm>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxDustDirMode<f64_3, i64_3, DustRiemannSolverMode::HB, Direction::xp>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxDustDirMode<f64_3, i64_3, DustRiemannSolverMode::HB, Direction::ym>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxDustDirMode<f64_3, i64_3, DustRiemannSolverMode::HB, Direction::yp>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxDustDirMode<f64_3, i64_3, DustRiemannSolverMode::HB, Direction::zm>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxDustDirMode<f64_3, i64_3, DustRiemannSolverMode::HB, Direction::zp>;

template class shammodels::basegodunov::modules::
    NodeComputeFluxDustDirMode<f64_3, i64_3, DustRiemannSolverMode::DHLL, Direction::xm>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxDustDirMode<f64_3, i64_3, DustRiemannSolverMode::DHLL, Direction::xp>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxDustDirMode<f64_3, i64_3, DustRiemannSolverMode::DHLL, Direction::ym>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxDustDirMode<f64_3, i64_3, DustRiemannSolverMode::DHLL, Direction::yp>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxDustDirMode<f64_3, i64_3, DustRiemannSolverMode::DHLL, Direction::zm>;
template class shammodels::basegodunov::modules::
    NodeComputeFluxDustDirMode<f64_3, i64_3, DustRiemannSolverMode::DHLL, Direction::zp>;
