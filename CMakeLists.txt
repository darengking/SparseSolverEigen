cmake_minimum_required(VERSION 2.8)

PROJECT(SparseSolve)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
SET(CMAKE_CXX_FLAGS "-std=c++0x -O2 -g -Wall")

FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

INCLUDE_DIRECTORIES(include)

# ----- Add library target -----
ADD_LIBRARY(sparselinear STATIC src/sparse_solve.cpp)


# ----- Testing -----
ADD_EXECUTABLE(oscar_test test/oscar_test.cpp)
target_link_libraries(oscar_test sparselinear)
ADD_EXECUTABLE(fista_test test/fista_test.cpp)
target_link_libraries(fista_test sparselinear)

# ----- Install -----
#IF ( INSTALL_LIBS )
#INSTALL(FILES ${lib_interfaces} DESTINATION lib/lasso/include)
#INSTALL(TARGETS lasso
#  RUNTIME DESTINATION lib/lasso/bin
#  LIBRARY DESTINATION lib/lasso/lib
#  ARCHIVE DESTINATION lib/lasso/lib
#)
#ENDIF()
