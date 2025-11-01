project(SharkUnitTest LANGUAGES CXX)

CPMAddPackage(
        NAME gtest
        GITHUB_REPOSITORY google/googletest
        GIT_TAG v1.17.0
        VERSION 1.17.0
        OPTIONS
        "INSTALL_GTEST OFF"
        "gtest_force_shared_crt ON"
)

set(${CMAKE_CURRENT_BINARY_DIR})

AddTestProgram(SharkTest.cpp "Shark::Shark;GTest::gtest;GTest::gtest_main;GTest::gmock" "Core")
AddTestProgram(CommonTest.cpp "Shark::Shark;GTest::gtest;GTest::gtest_main;GTest::gmock" "Math")
AddTestProgram(BitsTest.cpp "Shark::Shark;GTest::gtest;GTest::gtest_main;GTest::gmock" "Math")
