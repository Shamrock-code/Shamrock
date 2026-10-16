# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
# ~~~

######################
# precompiled headers
#
# A handful of headers (shambackends/sycl.hpp, shambackends/DeviceBuffer.hpp,
# nlohmann/json.hpp) get re-parsed from scratch in every one of the hundreds of
# translation units that include them directly or transitively, and dominate total
# build time. shambackends is the earliest target in the dependency graph that
# already needs all three, so it builds the actual PCH (sham_pch_root); every other
# target that depends on shambackends, directly or transitively, reuses that same
# precompiled object instead of re-parsing the headers itself (sham_pch_reuse).
######################

set(SHAMROCK_PCH_HEADERS "<shambackends/sycl.hpp>" "<shambackends/DeviceBuffer.hpp>"
                         "<nlohmann/json.hpp>"
)

# Precompile SHAMROCK_PCH_HEADERS for `target` and make the result available for others to
# reuse via sham_pch_reuse(). Only call this once, on the root target (shambackends).
function(sham_pch_root target)
    if(SHAMROCK_USE_PCH)
        target_precompile_headers(${target} PRIVATE ${SHAMROCK_PCH_HEADERS})
    endif()
endfunction()

# Reuse the PCH built by sham_pch_root(shambackends) for `target`. `target` must depend
# (directly or transitively) on shambackends.
function(sham_pch_reuse target)
    if(SHAMROCK_USE_PCH)
        target_precompile_headers(${target} REUSE_FROM shambackends)
    endif()
endfunction()
