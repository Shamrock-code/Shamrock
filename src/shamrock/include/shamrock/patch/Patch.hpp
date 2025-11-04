// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Patch.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Header file for the patch struct and related function
 *
 */

#include "shambase/aliases_int.hpp"
#include "PatchCoord.hpp"
#include "shammath/CoordRange.hpp"
#include "shamsys/MpiWrapper.hpp"

namespace shamrock::patch {

    template<u32 dim>
    MPI_Datatype get_patch_mpi_type();

    /**
     * @brief Patch object that contain generic patch information
     *
     */
    struct Patch {

        ////////////////////////////////////////////////////////////////////////////////////////////
        // Constexpr defs
        ////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * \var dim
         * \brief dimension of the patch (only 3 so far)
         *
         * \var splts_count
         * \brief if a patch splits, this gives the number of childs
         *
         * \var err_node_flag
         * \brief value of `node_owner_id` if the patch is invalid
         *
         */

        static constexpr u32 dim = 3U;

        static_assert(dim < 4, "the patch object is implemented only up to dim 3");

        static constexpr u32 splts_count = 1U << dim;

        static constexpr u32 err_node_flag = u32_max;

        ////////////////////////////////////////////////////////////////////////////////////////////
        // Members
        ////////////////////////////////////////////////////////////////////////////////////////////

        /**
         * \var id_patch
         * \brief unique key that identify the patch
         *
         * \var pack_node_index
         * \brief this value mean "to pack with index xxx in the global patch table"
         * and not "to pack with id_pach == xxx"
         *
         * \var load_value
         * \brief if synchronized contain the load value of the patch
         *
         * \var coord_min
         * \brief
         *
         * \var coord_max
         * \brief
         *
         * \var data_count
         * \brief number of element in the corresponding patchdata
         *
         * \var node_owner_id
         * \brief node rank owner of this patch
         */

        u64 id_patch;
        u64 pack_node_index;
        u64 load_value;

        std::array<u64, dim> coord_min;
        std::array<u64, dim> coord_max;

        u32 node_owner_id;

        ////////////////////////////////////////////////////////////////////////////////////////////
        // functions
        ////////////////////////////////////////////////////////////////////////////////////////////

        /**
         * @brief check if patch equals
         *
         * @param rhs
         * @return true
         * @return false
         */
        bool operator==(const Patch &rhs);

        /**
         * @brief Make the patch in error mode (patch struct that will be flushed on sync)
         */
        inline void set_err_mode() { node_owner_id = err_node_flag; }

        /**
         * @brief check if a patch is in error mode
         *
         * @return true this patch is in error mode it should be flushed out
         * @return false  this patch is not in error mode it is a valid one
         */
        [[nodiscard]] inline bool is_err_mode() const { return node_owner_id == err_node_flag; }

        [[nodiscard]] std::array<u64, dim> get_split_coord() const;

        [[nodiscard]] std::array<Patch, splts_count> get_split() const;

        [[nodiscard]] static Patch merge_patch(std::array<Patch, splts_count> patches);

        /**
         * @brief return an open interval of the corresponding patch coordinates given the div &
         * offset
         *
         * @tparam T type to convert coordinates to
         * @param divfact
         * @param offset
         * @return std::tuple<sycl::vec<T,3>,sycl::vec<T,3>>
         */
        template<class T>
        std::tuple<sycl::vec<T, 3>, sycl::vec<T, 3>> convert_coord(
            sycl::vec<u64, 3> src_offset, sycl::vec<T, 3> divfact, sycl::vec<T, 3> offset) const;

        /**
         * @brief check if particle is in the asked range, given the ouput of @convert_coord
         *
         * @tparam T the type of coordinate
         * @param val the value to check against
         * @param min_val the min range given by convert_coord
         * @param max_val the max range given by convert_coord
         * @return true is in the patch
         * @return false is not in the patch
         */
        template<class T>
        inline static bool is_in_patch_converted(
            sycl::vec<T, 3> val, sycl::vec<T, 3> min_val, sycl::vec<T, 3> max_val);

        inline void override_from_coord(PatchCoord<dim> pc) {
            coord_min[0] = pc.coord_min[0];
            coord_min[1] = pc.coord_min[1];
            coord_min[2] = pc.coord_min[2];
            coord_max[0] = pc.coord_max[0];
            coord_max[1] = pc.coord_max[1];
            coord_max[2] = pc.coord_max[2];
        }

        [[nodiscard]] inline PatchCoord<dim> get_coords() const { return {coord_min, coord_max}; }

        inline shammath::CoordRange<u64_3> get_patch_range() {
            return get_coords().get_patch_range();
        }
    };

    ////////////////////////////////////////////
    // out of line implementation
    ////////////////////////////////////////////

    inline bool Patch::operator==(const Patch &rhs) {

        bool ret_val = true;

        ret_val = ret_val && (id_patch == rhs.id_patch);

        ret_val = ret_val && (pack_node_index == rhs.pack_node_index);
        ret_val = ret_val && (load_value == rhs.load_value);

#pragma unroll
        for (u32 i = 0; i < dim; i++) {
            ret_val = ret_val && (coord_min[i] == rhs.coord_min[i]);
        }

#pragma unroll
        for (u32 i = 0; i < dim; i++) {
            ret_val = ret_val && (coord_max[i] == rhs.coord_max[i]);
        }

        ret_val = ret_val && (node_owner_id == rhs.node_owner_id);

        return ret_val;
    }

    template<class T>
    inline std::tuple<sycl::vec<T, 3>, sycl::vec<T, 3>> Patch::convert_coord(
        sycl::vec<u64, 3> src_offset, sycl::vec<T, 3> divfact, sycl::vec<T, 3> offset) const {
        return PatchCoord<dim>::convert_coord(
            coord_min,
            coord_max,
            {src_offset.x(), src_offset.y(), src_offset.z()},
            divfact,
            offset);
    }

    template<class T>
    inline bool Patch::is_in_patch_converted(
        sycl::vec<T, 3> val, sycl::vec<T, 3> min_val, sycl::vec<T, 3> max_val) {
        return (
            (min_val.x() <= val.x()) && (val.x() < max_val.x()) && (min_val.y() <= val.y())
            && (val.y() < max_val.y()) && (min_val.z() <= val.z()) && (val.z() < max_val.z()));
    }

    [[nodiscard]] inline auto Patch::get_split_coord() const -> std::array<u64, dim> {
        return PatchCoord<dim>::get_split_coord(coord_min, coord_max);
    }

    [[nodiscard]] inline auto Patch::get_split() const -> std::array<Patch, splts_count> {

        // init vars
        Patch p0, p1, p2, p3, p4, p5, p6, p7;

        // setup internal fields
        p0 = *this; // copy of the current state
        p0.load_value /= 8;

        p1 = p0;
        p2 = p0;
        p3 = p0;
        p4 = p0;
        p5 = p0;
        p6 = p0;
        p7 = p0;

        std::array<PatchCoord<dim>, splts_count> splts_c
            = PatchCoord<dim>::get_split(coord_min, coord_max);

        p0.override_from_coord(splts_c[0]);
        p1.override_from_coord(splts_c[1]);
        p2.override_from_coord(splts_c[2]);
        p3.override_from_coord(splts_c[3]);
        p4.override_from_coord(splts_c[4]);
        p5.override_from_coord(splts_c[5]);
        p6.override_from_coord(splts_c[6]);
        p7.override_from_coord(splts_c[7]);

        return {p0, p1, p2, p3, p4, p5, p6, p7};
    }

    [[nodiscard]] inline Patch Patch::merge_patch(std::array<Patch, splts_count> patches) {

        PatchCoord merged_c = PatchCoord<dim>::merge(
            {patches[0].get_coords(),
             patches[1].get_coords(),
             patches[2].get_coords(),
             patches[3].get_coords(),
             patches[4].get_coords(),
             patches[5].get_coords(),
             patches[6].get_coords(),
             patches[7].get_coords()});

        Patch ret{};
        ret = patches[0];

        ret.coord_min[0] = merged_c.coord_min[0];
        ret.coord_min[1] = merged_c.coord_min[1];
        ret.coord_min[2] = merged_c.coord_min[2];
        ret.coord_max[0] = merged_c.coord_max[0];
        ret.coord_max[1] = merged_c.coord_max[1];
        ret.coord_max[2] = merged_c.coord_max[2];

        ret.pack_node_index = u64_max;

        ret.load_value += patches[1].load_value;
        ret.load_value += patches[2].load_value;
        ret.load_value += patches[3].load_value;
        ret.load_value += patches[4].load_value;
        ret.load_value += patches[5].load_value;
        ret.load_value += patches[6].load_value;
        ret.load_value += patches[7].load_value;

        return ret;
    }

} // namespace shamrock::patch
