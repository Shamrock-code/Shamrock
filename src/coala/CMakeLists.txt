
target_sources(shamrock_lib
    PRIVATE
    setup.cpp
    math_functions.cpp
    kernel_approx/flux/flux_components.cpp
    kernel_approx/flux/flux_function.cpp
    kernel_approx/intflux/termA/termA_intflux_components.cpp
    kernel_approx/intflux/termA/termA_intflux_function.cpp
    kernel_approx/intflux/termB/termB_intflux_components.cpp
    kernel_approx/intflux/termB/termB_intflux_function.cpp
    polynomials_legendre.cpp
    reconstruction_gh.cpp
    limiter.cpp
    GQ_legendre_nodes_weights.cpp
    L2proj_GQ.cpp
    generate_tabflux_tabintflux.cpp
    generate_flux_intflux.cpp
    solver_DG.cpp 
    generate_gij.cpp
    compute_coag.cpp
    iterate_coag.cpp
    coala.cpp
    PUBLIC
    )







