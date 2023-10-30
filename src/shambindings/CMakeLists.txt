

# Whenever this glob's value changes, cmake will rerun and update the build with the
# new/removed files.
file(GLOB_RECURSE glob_bindings_sources CONFIGURE_DEPENDS "*.cpp")

target_sources(shamrock_lib PRIVATE ${glob_bindings_sources})
