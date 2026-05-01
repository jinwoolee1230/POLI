project(robin_tests)

set(TEST_SOURCES graph_solvers_test.cpp
        graph_core_test.cpp
        core_test.cpp
        problems_test.cpp
        robin_test.cpp
        utils_test.cpp
        benchmarks.cpp
        math_test.cpp)

add_executable(robin_unit_tests main.cc ${TEST_SOURCES})
target_link_libraries(robin_unit_tests PUBLIC robin::robin
        PRIVATE OpenMP::OpenMP_CXX)

# CTest test discovery
include(CTest)
include(Catch)
catch_discover_tests(robin_unit_tests)

# copy test data
# Copy test data files to binary directory
file(COPY ./data
        DESTINATION .
        FILES_MATCHING
        PATTERN *.mtx)
