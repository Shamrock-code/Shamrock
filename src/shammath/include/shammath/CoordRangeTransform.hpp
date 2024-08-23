// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file CoordRangeTransform.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "CoordRange.hpp"
#include "shambackends/vec.hpp"

namespace shammath {

    template<class Tsource, class Tdest>
    class CoordRangeTransform {

        using SourceProp = shambase::VectorProperties<Tsource>;
        using DestProp   = shambase::VectorProperties<Tdest>;

        using component_source_t = typename SourceProp::component_type;
        using component_dest_t   = typename DestProp::component_type;

        static constexpr bool source_is_int = SourceProp::is_uint_based;

        static constexpr bool dest_is_int = SourceProp::is_uint_based;

        enum TransformFactMode { multiply, divide };

        TransformFactMode mode;

        // written as Patch->Coord transform
        Tdest fact;

        Tdest dest_coord_min;
        Tsource source_coord_min;

        public:
        static_assert(
            SourceProp::dimension == DestProp::dimension,
            "input and output dimensions should be the same");

        CoordRangeTransform(CoordRange<Tsource> source_range, CoordRange<Tdest> dest_range);

        CoordRange<Tdest> transform(CoordRange<Tsource> rnge) const;
        CoordRange<Tsource> reverse_transform(CoordRange<Tdest> rnge) const;

        Tdest transform(Tsource coord) const;
        Tsource reverse_transform(Tdest rnge) const;

        void print_transform() const;
    };

    //////////////////////////////////
    // out of line impl
    //////////////////////////////////

    template<class Tsource, class Tdest>
    inline CoordRange<Tdest>
    CoordRangeTransform<Tsource, Tdest>::transform(CoordRange<Tsource> rnge) const {

        Tsource pmin = rnge.lower;
        Tsource pmax = rnge.upper;

        if (mode == multiply) {
            return {
                ((pmin - source_coord_min).template convert<component_dest_t>()) * fact
                    + dest_coord_min,
                ((pmax - source_coord_min).template convert<component_dest_t>()) * fact
                    + dest_coord_min};
        } else {
            return {
                ((pmin - source_coord_min).template convert<component_dest_t>()) / fact
                    + dest_coord_min,
                ((pmax - source_coord_min).template convert<component_dest_t>()) / fact
                    + dest_coord_min};
        }
    }

    template<class Tsource, class Tdest>
    inline CoordRange<Tsource>
    CoordRangeTransform<Tsource, Tdest>::reverse_transform(CoordRange<Tdest> rnge) const {

        Tsource pmin;
        Tsource pmax;

        if (mode == multiply) {
            return {
                ((rnge.lower - dest_coord_min) / fact).template convert<component_source_t>()
                    + source_coord_min,
                ((rnge.upper - dest_coord_min) / fact).template convert<component_source_t>()
                    + source_coord_min};
        } else {
            return {
                ((rnge.lower - dest_coord_min) * fact).template convert<component_source_t>()
                    + source_coord_min,
                ((rnge.upper - dest_coord_min) * fact).template convert<component_source_t>()
                    + source_coord_min};
        }
    }

    template<class Tsource, class Tdest>
    inline Tdest CoordRangeTransform<Tsource, Tdest>::transform(Tsource coord) const {

        if (mode == multiply) {
            return ((coord - source_coord_min).template convert<component_dest_t>()) * fact
                   + dest_coord_min;
        } else {
            return ((coord - source_coord_min).template convert<component_dest_t>()) / fact
                   + dest_coord_min;
        }
    }

    template<class Tsource, class Tdest>
    inline Tsource CoordRangeTransform<Tsource, Tdest>::reverse_transform(Tdest coord) const {

        if (mode == multiply) {
            return ((coord - dest_coord_min) / fact).template convert<component_source_t>()
                   + source_coord_min;
        } else {
            return ((coord - dest_coord_min) * fact).template convert<component_source_t>()
                   + source_coord_min;
        }
    }

} // namespace shammath
