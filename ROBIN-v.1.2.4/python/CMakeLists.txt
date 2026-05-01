cmake_minimum_required(VERSION 3.18)

project(robin_python_bindings)

# Set build type
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Specify the C++ standard to handle a build error related to `std::align_val_t_error` in Xenium.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG master
)
FetchContent_MakeAvailable(pybind11)

# NOTE (hlim): To generate `libpmc.a` instead of `libpmc.so`.
# On a local computer, either `libpmc.a` or `libpmc.so` is fine,
# but for PyPI, it should be `libpmc.a` to make it more independent of the environment.
# More details can be found here: ttps://github.com/jingnanshi/pmc/pull/2
option(PMC_BUILD_SHARED "Build PMC as a shared library" OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS "Python Interpreter Version: ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

# fix for clang
# see: https://github.com/pybind/pybind11/issues/1818
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(spark_robin PUBLIC -fsized-deallocation)
endif ()

if (DEFINED SKBUILD)
    message(STATUS "Building with Scikit-Build")
endif ()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../)
  # 'robin_cpp': user-defined build directory
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ ${CMAKE_CURRENT_BINARY_DIR}/robin_cpp)
else()
  message(STATUS "Performing out-of-tree build, fetching ROBIN v${CMAKE_PROJECT_VERSION} Release from Github")
  include(FetchContent)
  FetchContent_Declare(
    ext_robin PREFIX robin
    URL https://github.com/MIT-SPARK/ROBIN/archive/refs/tags/v${CMAKE_PROJECT_VERSION}.tar.gz)
  FetchContent_MakeAvailable(ext_robin)
endif()

pybind11_add_module(spark_robin robin_py/robin_py.cpp)

target_link_libraries(spark_robin PUBLIC robin::robin pmc)

install(TARGETS spark_robin DESTINATION .)
