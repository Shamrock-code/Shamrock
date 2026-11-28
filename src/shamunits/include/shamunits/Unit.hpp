// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Unit.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief A physical unit as a value: its SI dimension and its magnitude in SI base units
 *
 * Units, prefixes and physical constants are all the same thing: a dimension plus a factor.
 * Everything here is constexpr and STL-free so it can be used in device code.
 *
 * Unit<Tscal> is a structural type for any floating-point Tscal: it can be passed by value as a
 * template argument (`template<Unit u>`, or `template<Unit<double> u>`), which is what lets
 * UnitSystem::get<u>() keep every exponent a constant expression. Only a raw
 * `template<double>` parameter is unsupported on clang < 18, so do not add one.
 *
 * The algebra never converts precisions implicitly, not even for a scalar operand: a scalar must
 * have exactly the unit's type (`149597870700. * Unit<float>{}` does not compile, write
 * `149597870700.f`), and units of different precisions are converted with unit_cast<T>() first.
 * Spell the precision when brace-initialising (`Unit<float>{.dim = ..., .si_factor = ...}`):
 * class template argument deduction from an aggregate initializer needs clang >= 17.
 */

namespace shamunits {

    /// Exponents of the seven SI base dimensions
    struct Dimension {
        int second   = 0; ///< time exponent
        int metre    = 0; ///< length exponent
        int kilogram = 0; ///< mass exponent
        int ampere   = 0; ///< electric current exponent
        int kelvin   = 0; ///< temperature exponent
        int mole     = 0; ///< amount of substance exponent
        int candela  = 0; ///< luminous intensity exponent

        /// Exponent-wise equality
        constexpr bool operator==(const Dimension &) const = default;
    };

    /// Dimension of a product: exponents add
    constexpr Dimension operator*(Dimension a, Dimension b) noexcept {
        return {
            .second   = a.second + b.second,
            .metre    = a.metre + b.metre,
            .kilogram = a.kilogram + b.kilogram,
            .ampere   = a.ampere + b.ampere,
            .kelvin   = a.kelvin + b.kelvin,
            .mole     = a.mole + b.mole,
            .candela  = a.candela + b.candela,
        };
    }

    /// Dimension of a quotient: exponents subtract
    constexpr Dimension operator/(Dimension a, Dimension b) noexcept {
        return {
            .second   = a.second - b.second,
            .metre    = a.metre - b.metre,
            .kilogram = a.kilogram - b.kilogram,
            .ampere   = a.ampere - b.ampere,
            .kelvin   = a.kelvin - b.kelvin,
            .mole     = a.mole - b.mole,
            .candela  = a.candela - b.candela,
        };
    }

    /// Dimension raised to an integer power: exponents scale
    constexpr Dimension dim_pow(Dimension d, int n) noexcept {
        return {
            .second   = d.second * n,
            .metre    = d.metre * n,
            .kilogram = d.kilogram * n,
            .ampere   = d.ampere * n,
            .kelvin   = d.kelvin * n,
            .mole     = d.mole * n,
            .candela  = d.candela * n,
        };
    }

    /// A unit: its dimension and its magnitude expressed in SI base units, in precision Tscal
    template<class Tscal = double>
    struct Unit {
        Dimension dim{};     ///< SI dimension
        Tscal si_factor = 1; ///< value of one of this unit in SI base units

        /// Exact equality of dimension and factor
        constexpr bool operator==(const Unit &) const = default;
    };

    /// Explicit precision conversion, e.g. `unit_cast<float>(units::metre)`
    template<class Tdst, class Tsrc>
    constexpr Unit<Tdst> unit_cast(Unit<Tsrc> u) noexcept {
        return {.dim = u.dim, .si_factor = static_cast<Tdst>(u.si_factor)};
    }

    /// Product of two units: exponents add, factors multiply
    template<class T>
    constexpr Unit<T> operator*(Unit<T> a, Unit<T> b) noexcept {
        return {.dim = a.dim * b.dim, .si_factor = a.si_factor * b.si_factor};
    }

    /// Quotient of two units: exponents subtract, factors divide
    template<class T>
    constexpr Unit<T> operator/(Unit<T> a, Unit<T> b) noexcept {
        return {.dim = a.dim / b.dim, .si_factor = a.si_factor / b.si_factor};
    }

    /// Scale a unit, e.g. `149597870700. * metre`
    template<class T>
    constexpr Unit<T> operator*(T s, Unit<T> u) noexcept {
        return {.dim = u.dim, .si_factor = s * u.si_factor};
    }

    /// Scale a unit, e.g. `metre * 1e3`
    template<class T>
    constexpr Unit<T> operator*(Unit<T> u, T s) noexcept {
        return {.dim = u.dim, .si_factor = u.si_factor * s};
    }

    /// Divide a unit by a scalar, e.g. `joule / 1e7`
    template<class T>
    constexpr Unit<T> operator/(Unit<T> u, T s) noexcept {
        return {.dim = u.dim, .si_factor = u.si_factor / s};
    }

    /// Scalar over a unit, e.g. `1. / second`: the dimension is inverted
    template<class T>
    constexpr Unit<T> operator/(T s, Unit<T> u) noexcept {
        return {.dim = Dimension{} / u.dim, .si_factor = s / u.si_factor};
    }

    /**
     * @brief Integer power of a unit
     *
     * Named upow rather than pow so argument-dependent lookup never competes with std::pow or
     * sycl::pow. This is the only place in the unit algebra that can divide (for n < 0), and it
     * folds at compile time whenever its argument is a constant expression.
     */
    template<class T>
    constexpr Unit<T> upow(Unit<T> u, int n) noexcept {
        T f   = 1;
        T b   = u.si_factor;
        int k = (n < 0) ? -n : n;
        while (k != 0) {
            if ((k & 1) != 0) {
                f *= b;
            }
            k >>= 1;
            if (k != 0) { // squaring past the last bit could overflow when the result fits
                b *= b;
            }
        }
        return {.dim = dim_pow(u.dim, n), .si_factor = (n < 0) ? T(1) / f : f};
    }

    /// A base unit of the given dimension with factor 1, e.g. `base_unit({.metre = 1})`
    template<class T = double>
    constexpr Unit<T> base_unit(Dimension d) noexcept {
        return {.dim = d, .si_factor = T(1)};
    }

    /// A dimensionless scale factor, e.g. a prefix: `scale(1e6)` (precision of the argument)
    template<class T>
    constexpr Unit<T> scale(T s) noexcept {
        return {.dim = Dimension{}, .si_factor = s};
    }

} // namespace shamunits
