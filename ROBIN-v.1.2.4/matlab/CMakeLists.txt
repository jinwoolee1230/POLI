cmake_minimum_required(VERSION 3.10)

project(robin_matlab_bindings)

find_package(Matlab COMPONENTS MX_LIBRARY)

if (Matlab_FOUND)
    message(STATUS "MATLAB root directory found: ${Matlab_ROOT_DIR}.")
    matlab_add_mex(
            NAME findInlierStructure_mex
            SRC findInlierStructure_mex.cpp
            LINK_TO Eigen3::Eigen robin
    )
    set_target_properties(findInlierStructure_mex PROPERTIES COMPILE_FLAGS "-fvisibility=default")

    matlab_add_mex(
            NAME makeVecAvgInvGraph_mex
            SRC makeVecAvgInvGraph_mex.cpp
            LINK_TO Eigen3::Eigen robin
    )
    set_target_properties(makeVecAvgInvGraph_mex PROPERTIES COMPILE_FLAGS "-fvisibility=default")

    matlab_add_mex(
            NAME makeRotAvgInvGraph_mex
            SRC makeRotAvgInvGraph_mex.cpp
            LINK_TO Eigen3::Eigen robin
    )
    set_target_properties(makeRotAvgInvGraph_mex PROPERTIES COMPILE_FLAGS "-fvisibility=default")

    matlab_add_mex(
            NAME make3dRegInvGraph_mex
            SRC make3dRegInvGraph_mex.cpp
            LINK_TO Eigen3::Eigen robin
    )
    set_target_properties(make3dRegInvGraph_mex PROPERTIES COMPILE_FLAGS "-fvisibility=default")

    matlab_add_mex(
            NAME make3dNormalRegInvGraph_mex
            SRC make3dNormalRegInvGraph_mex.cpp
            LINK_TO Eigen3::Eigen robin
    )
    set_target_properties(make3dNormalRegInvGraph_mex PROPERTIES COMPILE_FLAGS "-fvisibility=default")

    # copy MATLAB .m files to binary directory
    file(COPY .
            DESTINATION .
            FILES_MATCHING
            PATTERN *.m)
else ()
    message(WARNING "MATLAB root directory not found. Failed to build MATLAB bindings.")
    set(BUILD_MATLAB_BINDINGS OFF)
endif ()