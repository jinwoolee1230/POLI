get_filename_component(XENIUM_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
include(CMakeFindDependencyMacro)

find_dependency(Threads REQUIRED)

include("${XENIUM_CMAKE_DIR}/xeniumTargets.cmake")
