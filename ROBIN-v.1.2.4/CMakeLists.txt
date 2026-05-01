cmake_minimum_required(VERSION 3.5..3.31)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(robin
        VERSION 1.2.3
        DESCRIPTION "Robust outlier rejection based on measurement compatibility graphs"
        LANGUAGES CXX
        )

#
# Disable in-tree building
#
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Prevented in-tree built. Create a build folder outside source instead.")
endif ()

#
# Default build type to debug
#
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

#
# Update CMake module path
#
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(DownloadExternal)

#
# Build options
#
option(BUILD_DOCS "Build documentation." OFF)
option(BUILD_TESTS "Enable testing with ctest." ON)
option(BUILD_MATLAB_BINDINGS "Build MATLAB bindings" OFF)
option(USE_ASAN "Enable address sanitizer" OFF)
option(USE_SYSTEM_EIGEN3 "Use system pre-installed Eigen" ON)
option(ENABLE_DIAGNOSTIC_PRINT "Enable printing of diagnostic messages" OFF)

if (ENABLE_DIAGNOSTIC_PRINT)
    message(STATUS "Enable printing of diagnostic messages.")
    add_definitions(-DROBIN_DIAG_PRINT)
endif ()

#
# Dependencies
#
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_external_dependency("Eigen3" "Eigen3::Eigen" "${CMAKE_CURRENT_LIST_DIR}/cmake/DownloadEigen.cmake")
find_package(OpenMP QUIET REQUIRED)
robin_download_pmc()
robin_download_xenium()
add_subdirectory(${pmc_SOURCE_DIR} ${pmc_BINARY_DIR})

#
# Add robin target
#
include(GNUInstallDirs)

file(GLOB
        SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(robin STATIC ${SOURCE_FILES})
add_library(robin::robin ALIAS robin)
target_link_libraries(robin
        PUBLIC Eigen3::Eigen
               xenium
        PRIVATE OpenMP::OpenMP_CXX
        PRIVATE pmc)
target_include_directories(
        robin
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if (USE_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fPIC")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
endif ()

#
# Install Robin
#
include(CMakePackageConfigHelpers)

install(
        TARGETS robin
        EXPORT robinTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

write_basic_package_version_file(
        robinConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/robinConfig.cmake.in
        ${PROJECT_BINARY_DIR}/robinConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

export(
        EXPORT robinTargets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/robinTargets.cmake
        NAMESPACE robin::
)

install(
    TARGETS xenium pmc
    EXPORT robinTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)


install(
        EXPORT robinTargets
        FILE robinTargets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
        NAMESPACE robin::
)

install(
        FILES
        ${PROJECT_BINARY_DIR}/robinConfig.cmake
        ${PROJECT_BINARY_DIR}/robinConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(DIRECTORY include/robin DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# To set forcibly xeniumConfig.cmake
install(FILES cmake/xeniumConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xenium
        )

export(PACKAGE robin)

# Tests
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

# MATLAB bindings
if (BUILD_MATLAB_BINDINGS)
    if (USE_ASAN)
        message(WARNING "MATLAB binding might not work with USE_ASAN=ON. Consider turning it off by -DUSE_ASAN=OFF.")
    endif ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    add_subdirectory(matlab)
endif ()

# Documentation
if (BUILD_DOCS)
    find_package(Doxygen 1.8)
    if (DOXYGEN_FOUND)
        add_subdirectory(docs)
    else ()
        set(BUILD_DOCS OFF)
    endif ()
endif ()

# Summary of configuration
message(STATUS "===============================================================")
message(STATUS "============= Robin Configuration Options =====================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "BUILD_DOCS: ${BUILD_DOCS}")
message(STATUS "BUILD_MATLAB_BINDINGS: ${BUILD_MATLAB_BINDINGS}")
message(STATUS "USE_ASAN: ${USE_ASAN}")
message(STATUS "ENABLE_DIAGNOSTIC_PRINT: ${ENABLE_DIAGNOSTIC_PRINT}")
message(STATUS "===============================================================")
