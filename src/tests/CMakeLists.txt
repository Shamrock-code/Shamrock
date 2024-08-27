if("${BUILD_TEST}")

    # Whenever this glob's value changes, cmake will rerun and update the build with the
    # new/removed files.
    file(GLOB_RECURSE test_list CONFIGURE_DEPENDS "*.cpp")

    target_sources(shamrock_test PRIVATE ${test_list})

endif()
