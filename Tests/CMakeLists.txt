if (BEE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(Units)
endif ()

if (BEE_BUILD_BENCHMARKS)
    add_subdirectory(Benchmarks)
endif ()