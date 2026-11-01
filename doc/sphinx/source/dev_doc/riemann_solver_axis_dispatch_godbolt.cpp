// Self-contained reproducer used to compare the "axis permutation" dispatch
// (rotate to +x, solve, rotate back) against calling the "_n" solver
// directly with the target face's unit normal.
//
// Paste into godbolt.org, select x86-64 clang, and diff the two
// `via_mz_dispatch` / `via_flux_n` entries in the generated assembly.
// See dev_doc/riemann_solver.md, section "Axis permutation vs projection".

// --- minimal stand-in for shambackends::vec (sycl::vec<double,3>) ---
struct Vec3 {
    double x, y, z;
    constexpr double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    constexpr double &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
};
constexpr Vec3 operator-(Vec3 a) { return Vec3{-a.x, -a.y, -a.z}; }
constexpr Vec3 operator+(Vec3 a, Vec3 b) { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
constexpr Vec3 operator*(Vec3 a, double s) { return Vec3{a.x * s, a.y * s, a.z * s}; }

// --- shammath::DustPrimState / DustConsState (trimmed) ---
template<class Tvec_>
struct DustPrimState {
    using Tvec = Tvec_;
    double rho{};
    Tvec vel{};
};

template<class Tvec_>
struct DustConsState {
    using Tvec = Tvec_;
    double rho{};
    Tvec rhovel{};

    constexpr DustConsState &operator*=(double f) {
        rho *= f;
        rhovel = rhovel * f;
        return *this;
    }
};

template<class Tvec>
constexpr DustConsState<Tvec> operator+(
    const DustConsState<Tvec> &a, const DustConsState<Tvec> &b) {
    return DustConsState<Tvec>{a.rho + b.rho, a.rhovel + b.rhovel};
}

// --- riemann_common.hpp helpers actually used on the mz path ---
template<class Tvec>
inline constexpr DustConsState<Tvec> d_hydro_flux_n(DustPrimState<Tvec> d_prim, Tvec n, double vn) {
    DustConsState<Tvec> d_flux;
    d_flux.rho    = d_prim.rho * vn;
    d_flux.rhovel = d_prim.vel * (d_prim.rho * vn);
    return d_flux;
}
template<class Tvec>
inline constexpr DustConsState<Tvec> d_hydro_flux_n(DustPrimState<Tvec> d_prim, Tvec n) {
    const double vn = n[0] * d_prim.vel[0] + n[1] * d_prim.vel[1] + n[2] * d_prim.vel[2];
    return d_hydro_flux_n(d_prim, n, vn);
}

template<class Tcons>
inline constexpr Tcons d_x_to_z(Tcons c) {
    Tcons d_cst;
    d_cst.rho       = c.rho;
    d_cst.rhovel[0] = -c.rhovel[2];
    d_cst.rhovel[1] = c.rhovel[1];
    d_cst.rhovel[2] = c.rhovel[0];
    return d_cst;
}
template<class Tcons>
inline constexpr Tcons d_invert_axis(Tcons c) {
    Tcons d_cst;
    d_cst.rho    = c.rho;
    d_cst.rhovel = -(c.rhovel);
    return d_cst;
}
template<class Tprim>
inline constexpr Tprim d_prim_z_to_x(Tprim p) {
    Tprim pprime;
    pprime.rho    = p.rho;
    pprime.vel[0] = p.vel[2];
    pprime.vel[1] = p.vel[1];
    pprime.vel[2] = -p.vel[0];
    return pprime;
}
template<class Tprim>
inline constexpr Tprim d_prim_invert_axis(Tprim p) {
    Tprim pprime;
    pprime.rho = p.rho;
    pprime.vel = -(p.vel);
    return pprime;
}

// --- a generic Riemann solver, standing in for any of Rusanov/HLL/HLLC/
//     dust-HLL/Huang-Bai: only the "_n" variant does real physics, every
//     per-axis wrapper below is pure plumbing around it. ---
template<class Tprim>
inline constexpr auto riemann_solver_flux_n(Tprim d_primL, Tprim d_primR, typename Tprim::Tvec n) {
    const auto vnL = n[0] * d_primL.vel[0] + n[1] * d_primL.vel[1] + n[2] * d_primL.vel[2];
    const auto vnR = n[0] * d_primR.vel[0] + n[1] * d_primR.vel[1] + n[2] * d_primR.vel[2];

    const auto fL = d_hydro_flux_n(d_primL, n, vnL);
    const auto fR = d_hydro_flux_n(d_primR, n, vnR);

    DustConsState<typename Tprim::Tvec> d_flux{};

    if (vnL > 0 && vnR > 0)
        d_flux = fL;
    else if (vnL < 0 && vnR < 0)
        d_flux = fR;
    else if (vnL < 0 && vnR > 0)
        d_flux *= 0;
    else if (vnL > 0 && vnR < 0)
        d_flux = (fL + fR);

    return d_flux;
}

template<class Tprim>
inline constexpr auto riemann_solver_flux_x(Tprim d_primL, Tprim d_primR) {
    return riemann_solver_flux_n(d_primL, d_primR, typename Tprim::Tvec{1, 0, 0});
}

template<class Tprim>
inline constexpr auto riemann_solver_flux_z(Tprim pL, Tprim pR) {
    return d_x_to_z(riemann_solver_flux_x(d_prim_z_to_x(pL), d_prim_z_to_x(pR)));
}

template<class Tprim>
inline constexpr auto riemann_solver_flux_mz(Tprim pL, Tprim pR) {
    return d_invert_axis(riemann_solver_flux_z(d_prim_invert_axis(pL), d_prim_invert_axis(pR)));
}

// --- the two call sites to diff ---
using Prim = DustPrimState<Vec3>;
using Cons = DustConsState<Vec3>;

Cons via_mz_dispatch(Prim pL, Prim pR) { return riemann_solver_flux_mz(pL, pR); }

Cons via_flux_n(Prim pL, Prim pR) { return riemann_solver_flux_n(pL, pR, Vec3{0, 0, -1}); }
