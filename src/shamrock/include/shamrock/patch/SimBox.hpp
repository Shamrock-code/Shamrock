// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SimBox.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shammath/CoordRange.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchCoordTransform.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include <type_traits>
#include <stdexcept>
#include <tuple>

namespace shamrock::patch {

    /**
     * @brief Store the information related to the size of the simulation box to convert patch
     * integer coordinates to floating point ones.
     */
    class SimulationBoxInfo {

        static constexpr u32 dim = 3;

        using var_t = FieldVariant<shammath::CoordRange>;

        PatchDataLayout &pdl;

        var_t bounding_box;
        PatchCoord<> patch_coord_bounding_box;

        public:
        inline SimulationBoxInfo(PatchDataLayout &pdl, PatchCoord<dim> patch_coord_bounding_box)
            : pdl(pdl), patch_coord_bounding_box(std::move(patch_coord_bounding_box)),
              bounding_box(shammath::CoordRange<f32>{}) {

            reset_box_size();
        }

        void set_patch_coord_bounding_box(PatchCoord<dim> new_patch_coord_box) {
            patch_coord_bounding_box = new_patch_coord_box;
            shamlog_debug_ln(
                "SimBox",
                "changed patch coord bounds :",
                std::pair{
                    u64_3{
                        new_patch_coord_box.coord_min[0],
                        new_patch_coord_box.coord_min[1],
                        new_patch_coord_box.coord_min[2]},
                    u64_3{
                        new_patch_coord_box.coord_max[0],
                        new_patch_coord_box.coord_max[1],
                        new_patch_coord_box.coord_max[2]}});
        }

        /**
         * @brief Get the stored bounding box of the domain
         *
         * @tparam T type of position vector
         * @return std::tuple<T, T> return [low bound, high bound[
         */
        template<class T>
        [[nodiscard]] std::tuple<T, T> get_bounding_box() const;

        /**
         * @brief Get the size of the stored bounding box of the domain
         *
         * @tparam T type of position vector
         * @return T the size of the bounding box
         */
        template<class T>
        inline T get_bounding_box_size() const {
            auto [bmin, bmax] = get_bounding_box<T>();
            return bmax - bmin;
        }

        /**
         * @brief Override the stored bounding box by the one given in new_box
         *
         * @tparam T type of position vector
         * @param new_box the new bounding box
         */
        template<class T>
        void set_bounding_box(shammath::CoordRange<T> new_box);

        /**
         * @brief Set the stored bounding box after an all-reduce operation on the supplied bounds.
         *
         * @details This function is used in the context of a distributed simulation.
         * It will all-reduce the bounding box of the domain over all the MPI ranks.
         * The all-reduce operation is done with the shamalgs::collective::allreduce_bounds
         * function.
         *
         * @tparam T type of position vector
         * @param new_box the new bounding box to be all_reduced accros ranks
         */
        template<class T>
        inline void allreduce_set_bounding_box(shammath::CoordRange<T> new_box) {
            std::pair<T, T> reduced = shamalgs::collective::allreduce_bounds(
                std::pair<T, T>{new_box.lower, new_box.upper});
            set_bounding_box<>(shammath::CoordRange<T>(reduced));
        }

        /**
         * @brief Get a PatchCoordTransform object that describes the conversion
         *        between patch coordinates and domain coordinates.
         *
         * @details
         * This function returns a PatchCoordTransform object that can be used to
         * convert between patch coordinates and domain coordinates. The
         * PatchCoordTransform object is templated on the type of position vector
         * used by the domain.
         *
         * @tparam T Type of position vector
         * @return PatchCoordTransform<T> A PatchCoordTransform object that can be
         *         used to transform between patch coordinates and domain
         *         coordinates.
         */
        template<class T>
        PatchCoordTransform<T> get_patch_transform() const;

        /**
         * @brief get the patch coordinates on the domain
         *
         * @tparam T type of position vector
         * @param p the patch
         * @return std::tuple<T,T> the [low bound, high bound[ coordinate of the patch in the domain
         */
        template<class T>
        std::tuple<T, T> patch_coord_to_domain(const Patch &p) const;

        // TODO implement box size reduction here

        /**
         * @brief Reset the bounding box of the simulation domain to the maximum
         *        extents of the main field.
         *
         * @details
         * This function resets the bounding box of the simulation domain to the
         * maximum extents of the main field. The bounding box is defined by the
         * PatchDataLayout object used to initialize the SimulationBoxInfo object.
         * The bounding box is a shammath::CoordRange object that contains the
         * coordinate range of the simulation domain.
         *
         * This function is called by the constructor of the SimulationBoxInfo
         * class, but it can also be called manually to reset the bounding box
         * after changing the main field.
         *
         * The bounding box is reset based on the type of the main field. If the
         * main field is of type f32_3, f64_3, u32_3, u64_3, or i64_3, the
         * bounding box is set to the maximum range of the corresponding type. If
         * the main field is of any other type, a std::runtime_error exception is
         * thrown.
         *
         * @throws std::runtime_error if the main field type is not one of the
         *         types listed above.
         */
        void reset_box_size();

        /// @todo replace vectype primtype in the code by primtype and sycl::vec<primtype,3> for the
        /// others
        template<class primtype>
        void clean_box(primtype tol);

        template<>
        inline void clean_box<f32>(f32 tol) {

            auto [bmin, bmax] = get_bounding_box<f32_3>();

            f32_3 center   = (bmin + bmax) / 2;
            f32_3 cur_delt = bmax - bmin;
            cur_delt /= 2;

            cur_delt *= tol;

            bmin = center - cur_delt;
            bmax = center + cur_delt;

            set_bounding_box<f32_3>({bmin, bmax});
        }

        template<>
        inline void clean_box<f64>(f64 tol) {
            auto [bmin, bmax] = get_bounding_box<f64_3>();

            f64_3 center   = (bmin + bmax) / 2;
            f64_3 cur_delt = bmax - bmin;
            cur_delt /= 2;

            cur_delt *= tol;

            bmin = center - cur_delt;
            bmax = center + cur_delt;

            set_bounding_box<f64_3>({bmin, bmax});
        }

        template<class primtype>
        inline std::tuple<sycl::vec<primtype, 3>, sycl::vec<primtype, 3>> get_box(Patch &p) {
            return patch_coord_to_domain<sycl::vec<primtype, 3>>(p);
        }

        template<class T>
        inline PatchCoordTransform<T> get_transform() {
            auto [bmin, bmax] = get_bounding_box<T>();
            return PatchCoordTransform<T>{
                patch_coord_bounding_box, shammath::CoordRange<T>{bmin, bmax}};
        }

        /**
         * @brief Serializes a SimulationBoxInfo object to a JSON object.
         *
         * @param j The JSON object to serialize to.
         * @param p The SimulationBoxInfo object to serialize.
         */
        void to_json(nlohmann::json &j);

        /**
         * @brief Deserializes a JSON object into a SimulationBoxInfo object.
         *
         * @param j The JSON object to deserialize from.
         * @param p The SimulationBoxInfo object to deserialize into.
         */
        void from_json(const nlohmann::json &j);
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // out of line implementation of the simbox
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    [[nodiscard]] inline std::tuple<T, T> SimulationBoxInfo::get_bounding_box() const {

        if (!pdl.check_main_field_type<T>()) {

            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the chosen type for the main field does not match the required template type\n"
                "call : "
                + std::string(__PRETTY_FUNCTION__));
        }

        const shammath::CoordRange<T> *pval
            = std::get_if<shammath::CoordRange<T>>(&bounding_box.value);

        if (!pval) {

            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the type in SimulationBoxInfo does not match the one in the layout\n"
                "call : "
                + std::string(__PRETTY_FUNCTION__));
        }

        return {pval->lower, pval->upper};
    }

    template<class T>
    inline void SimulationBoxInfo::set_bounding_box(shammath::CoordRange<T> new_box) {
        new_box.check_throw_ranges();
        if (pdl.check_main_field_type<T>()) {
            bounding_box.value = new_box;
        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "The main field is not of the required type\n"
                "call : "
                + std::string(__PRETTY_FUNCTION__));
        }
    }

    template<class T>
    PatchCoordTransform<T> SimulationBoxInfo::get_patch_transform() const {

        auto [bmin, bmax] = get_bounding_box<T>();

        shammath::CoordRange<T> tmp{bmin, bmax};

        if (tmp.is_err_mode()) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "the box size is not set, please resize the box to the domain size");
        }

        return PatchCoordTransform<T>{patch_coord_bounding_box.get_patch_range(), tmp};
    }

    template<class T>
    inline std::tuple<T, T> SimulationBoxInfo::patch_coord_to_domain(const Patch &p) const {

        PatchCoordTransform<T> transform = get_patch_transform<T>();

        auto [obj_min, obj_max] = transform.to_obj_coord(p);

        return {obj_min, obj_max};
    }

    inline void SimulationBoxInfo::reset_box_size() {

        if (pdl.check_main_field_type<f32_3>()) {
            bounding_box.value = shammath::CoordRange<f32_3>::max_range();
        } else if (pdl.check_main_field_type<f64_3>()) {
            bounding_box.value = shammath::CoordRange<f64_3>::max_range();
        } else if (pdl.check_main_field_type<u32_3>()) {
            bounding_box.value = shammath::CoordRange<u32_3>::max_range();
        } else if (pdl.check_main_field_type<u64_3>()) {
            bounding_box.value = shammath::CoordRange<u64_3>::max_range();
        } else if (pdl.check_main_field_type<i64_3>()) {
            bounding_box.value = shammath::CoordRange<i64_3>::max_range();
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "the chosen type for the main field is not handled");
        }
    }

} // namespace shamrock::patch
