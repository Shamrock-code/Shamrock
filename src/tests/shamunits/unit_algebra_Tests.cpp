// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file unit_algebra_Tests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Compile-time checks of the Unit / Dimension algebra (shamunits/Unit.hpp)
 *
 */

#include "shamtest/shamtest.hpp"
#include "shamunits/Unit.hpp"

namespace {
    using namespace shamunits;

    template<class A, class B>
    inline constexpr bool same_type = false;
    template<class A>
    inline constexpr bool same_type<A, A> = true;

    /// Relative comparison: factors built two different ways may differ by a few ulps
    template<class T>
    constexpr bool close(T a, T b, T rel) {
        T d  = a - b;
        T ab = (b < 0) ? -b : b;
        return ((d < 0) ? -d : d) <= rel * ab;
    }

    /// The whole algebra, instantiated for one precision. Local stand-ins: the real tables
    /// (shamunits::units) arrive in a later PR.
    template<class T>
    constexpr bool check_algebra() {
        constexpr Unit<T> s  = base_unit<T>({.second = 1});
        constexpr Unit<T> m  = base_unit<T>({.metre = 1});
        constexpr Unit<T> kg = base_unit<T>({.kilogram = 1});

        constexpr Unit<T> newton = kg * m / upow(s, 2);
        constexpr Unit<T> joule  = newton * m;
        constexpr Unit<T> watt   = joule / s;
        constexpr Unit<T> au     = T(149597870700.) * m;

        static_assert(newton.dim == Dimension{.second = -2, .metre = 1, .kilogram = 1});
        static_assert(joule.dim == Dimension{.second = -2, .metre = 2, .kilogram = 1});
        static_assert(watt.dim == (joule / s).dim);
        static_assert((T(1) / s).dim == Dimension{.second = -1});
        static_assert((m / upow(s, 2) * kg) == newton);

        static_assert((m * T(1e3)) == (T(1e3) * m));
        static_assert((joule / T(1e7)).si_factor == T(1) / T(1e7));
        static_assert((T(2) / s).si_factor == T(2));
        static_assert(scale(T(1e6)).dim == Dimension{});

        static_assert(upow(au, 0) == Unit<T>{});
        static_assert(upow(au, 1) == au);
        static_assert(upow(au, 2).si_factor == au.si_factor * au.si_factor);
        static_assert(
            upow(Unit<T>{.dim = {.metre = 1}, .si_factor = 2}, -2)
            == Unit<T>{.dim = {.metre = -2}, .si_factor = T(.25)});
        static_assert(upow(au, 3).si_factor > 0); // fits in float; no overflow on the way
        static_assert((upow(au, 1) * upow(au, 2)).dim == upow(au, 3).dim);
        static_assert(close((upow(au, -2) * upow(au, 3)).si_factor, au.si_factor, T(1e-6)));
        return true;
    }

    static_assert(check_algebra<double>());
    static_assert(check_algebra<float>());
    static_assert(check_algebra<long double>());

    static_assert(
        dim_pow(Dimension{.metre = 1, .ampere = -2}, 3) == Dimension{.metre = 3, .ampere = -6});

    // Default precision and deduction
    static_assert(same_type<Unit<>, Unit<double>>);
    static_assert(same_type<decltype(base_unit({.metre = 1})), Unit<double>>);
    static_assert(same_type<decltype(scale(1e6f)), Unit<float>>);

    // No implicit conversion: a scalar must have exactly the unit's type, and precisions never mix
    template<class A, class B>
    concept can_multiply = requires(A a, B b) { a * b; };
    template<class A, class B>
    concept can_divide = requires(A a, B b) { a / b; };
    static_assert(can_multiply<float, Unit<float>> && can_multiply<Unit<float>, float>);
    static_assert(can_divide<float, Unit<float>> && can_divide<Unit<float>, float>);
    static_assert(!can_multiply<double, Unit<float>> && !can_multiply<Unit<float>, double>);
    static_assert(!can_divide<double, Unit<float>> && !can_divide<Unit<float>, double>);
    static_assert(!can_multiply<int, Unit<double>> && !can_divide<Unit<double>, int>);
    static_assert(
        !can_multiply<Unit<float>, Unit<double>> && !can_divide<Unit<float>, Unit<double>>);

    // Explicit precision conversion
    constexpr Unit<double> au_d = 149597870700. * base_unit({.metre = 1});
    static_assert(unit_cast<float>(au_d).si_factor == 149597870700.f);
    static_assert(unit_cast<float>(au_d).dim == au_d.dim);

    // Unit<T> is a structural type for each precision, both as a concrete NTTP and through the
    // deduced placeholder template<Unit u>. This is what UnitSystem::get<u>() relies on; it
    // compiles on clang 15 (only a raw template<double> is rejected before clang 18).
    template<Unit u>
    struct UnitNttp {
        static constexpr auto value         = u;
        static constexpr int metre_exponent = u.dim.metre;
    };
    template<int n>
    struct IntNttp {
        static constexpr int value = n;
    };
    static_assert(UnitNttp<au_d>::value == au_d);
    static_assert(UnitNttp<upow(au_d, 2)>::metre_exponent == 2);
    static_assert(same_type<decltype(UnitNttp<unit_cast<float>(au_d)>::value), const Unit<float>>);
    static_assert(!same_type<UnitNttp<unit_cast<float>(au_d)>, UnitNttp<au_d>>);
    static_assert(IntNttp<UnitNttp<upow(au_d, -3)>::value.dim.metre>::value == -3);

} // namespace

NEW_TEST(Unittest, "shamunits/unit_algebra", 1) {
    // the static_asserts above are the test; this keeps it visible in the test list
    REQUIRE_EQUAL(au_d.dim.metre, 1);
}
