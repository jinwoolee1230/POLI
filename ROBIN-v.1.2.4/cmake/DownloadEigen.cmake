include(FetchContent)
FetchContent_Declare(eigen URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
                     PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/eigen.patch UPDATE_DISCONNECTED 1)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
  if(${CMAKE_VERSION} GREATER_EQUAL 3.25)
    add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} SYSTEM EXCLUDE_FROM_ALL)
  else()
    # Emulate the SYSTEM flag introduced in CMake 3.25. Withouth this flag the compiler will
    # consider this 3rdparty headers as source code and fail due the -Werror flag.
    add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
    get_target_property(eigen_include_dirs eigen INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(eigen PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${eigen_include_dirs}")
  endif()
endif()

