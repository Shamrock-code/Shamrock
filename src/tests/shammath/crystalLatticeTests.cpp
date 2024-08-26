// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/string.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shammath/CoordRange.hpp"
#include "shammath/crystalLattice.hpp"
#include "shammath/sphkernels.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <random>
#include <vector>

f64 compute_sum(shammath::CoordRange<f64_3> box, u32 id, std::vector<f64_3> &parts) {

    f64_3 delt = box.delt();

    f64 sum = 0;

    f64_3 target = parts[id];
    // u32 tmp = 0;
    for (i32 pi = -1; pi < 2; pi++) {
        for (i32 pj = -1; pj < 2; pj++) {
            for (i32 pk = -1; pk < 2; pk++) {

                f64_3 offset = {
                    delt.x() * pi,
                    delt.y() * pj,
                    delt.z() * pk,
                };

                for (const auto &ra : parts) {
                    f64_3 d = (ra - target) + offset;

                    f64 dsq = sycl::dot(d, d);

                    sum += shammath::M4<f64>::W_1d(dsq, 5);

                    // if(dsq < 14){
                    // tmp ++;}
                }
            }
        }
    }
    // logger::raw_ln(tmp);

    return sum;
}

bool all_sum_are_equals(shammath::CoordRange<f64_3> box, std::vector<f64_3> &parts) {

    f64 val = 0;

    bool found_diff = false;

    for (u32 i = 0; i < parts.size(); i++) {
        f64 comp = compute_sum(box, i, parts);

        if (i == 0) {
            val = comp;
        }

        if (fabs(comp - val) > 1e-14) {
            found_diff = true;
            logger::raw_ln(box.lower, box.upper, i, comp, val, fabs(comp - val));
        }
    }

    return !found_diff;
}

bool check_periodicity(std::array<i32, 3> coord_min, std::array<i32, 3> coord_max) {

    shammath::CoordRange<f64_3> box{};

    try {

        box = shammath::LatticeHCP<f64_3>::get_periodic_box(1, coord_min, coord_max);

        auto gen = shammath::LatticeHCP<f64_3>::Iterator{1., coord_min, coord_max};

        std::vector<f64_3> parts = gen.next_n(100000);

        return all_sum_are_equals(box, parts);

    } catch (shammath::LatticeError exp) {
        return true;
    }

    return true;
}

TestStart(
    Unittest,
    "shammath/crystalLattice/LatticeHCP/get_periodic_box",
    lattice_get_periodic_box_test,
    1) {
    std::mt19937 eng(0x1111);

    for (u32 i = 0; i < 100; i++) {
        i32 xmin = shamalgs::random::mock_value(eng, -7, 0);
        i32 ymin = shamalgs::random::mock_value(eng, -7, 0);
        i32 zmin = shamalgs::random::mock_value(eng, -7, 0);
        i32 xmax = shamalgs::random::mock_value(eng, 0, 7);
        i32 ymax = shamalgs::random::mock_value(eng, 0, 7);
        i32 zmax = shamalgs::random::mock_value(eng, 0, 7);

        shamtest ::asserts().assert_bool(
            shambase::format(
                "check periodicity : ({} {} {}) ({} {} {}) ({} {} {}) ",
                xmin,
                ymin,
                zmin,
                xmax,
                ymax,
                zmax,
                xmax - xmin,
                ymax - ymin,
                zmax - zmin

                ),
            check_periodicity({xmin, ymin, zmin}, {xmax, ymax, zmax}));
    }
}

TestStart(
    Unittest,
    "shammath/crystalLattice/LatticeHCP/nearest_periodic_box_indices",
    lattice_nearest_periodic_box_indices_test,
    1) {
    std::mt19937 eng(0x1111);

    for (u32 i = 0; i < 100; i++) {
        i32 xmin = shamalgs::random::mock_value(eng, -7, 0);
        i32 ymin = shamalgs::random::mock_value(eng, -7, 0);
        i32 zmin = shamalgs::random::mock_value(eng, -7, 0);
        i32 xmax = shamalgs::random::mock_value(eng, 0, 7);
        i32 ymax = shamalgs::random::mock_value(eng, 0, 7);
        i32 zmax = shamalgs::random::mock_value(eng, 0, 7);

        if (xmin == xmax || ymin == ymax || zmin == zmax)
            continue;

        std::pair<std::array<i32, 3>, std::array<i32, 3>> out
            = shammath::LatticeHCP<f64_3>::nearest_periodic_box_indices(
                {xmin, ymin, zmin}, {xmax, ymax, zmax});

        shamtest ::asserts().assert_bool(
            shambase::format(
                "check periodicity : ({} {} {}) ({} {} {}) ({} {} {})",
                xmin,
                ymin,
                zmin,
                xmax,
                ymax,
                zmax,
                xmax - xmin,
                ymax - ymin,
                zmax - zmin),
            shammath::LatticeHCP<f64_3>::can_make_periodic_box(out.first, out.second));
    }
}
