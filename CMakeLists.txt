cmake_minimum_required(VERSION 2.8.3)
project( sptam )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

#ds determine build type, default build type Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message("build type: '${CMAKE_BUILD_TYPE}'")

# declare cache entry parameters
SET(SHOW_TRACKED_FRAMES OFF CACHE BOOL "Enable/Disable OpenCV frame visualization for the tracker.")
SET(SHOW_PROFILING OFF CACHE BOOL "Enable/Disable Profiling of each step.")
SET(G2O_USE_NEW_OWNERSHIP ON CACHE BOOL "Enable/Disable new g2o ownership model (from revision: 7828003).")
SET(USE_EUROC_CALIBRATION ON CACHE BOOL "Enable/Disable hardcoded EuRoC calibration (ros bags do not contain camera info).")

if( SHOW_TRACKED_FRAMES )
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHOW_TRACKED_FRAMES")
  message("showing tracked frames")
endif()

if( SHOW_PROFILING )
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHOW_PROFILING")
  message("profiling enabled")
endif()

if(G2O_USE_NEW_OWNERSHIP)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DG2O_USE_NEW_OWNERSHIP")
  message("using new g2o ownership model")
endif()

if(USE_EUROC_CALIBRATION)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_EUROC_CALIBRATION")
  message("using internal EuRoC camera calibration")
endif()

## Find catkin macros and libraries
find_package( catkin REQUIRED COMPONENTS
  roscpp
  cmake_modules
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  message_filters
  image_geometry
  pcl_ros
  nav_msgs
  tf_conversions
  image_transport
)

# Find Boost Library
find_package(Boost COMPONENTS thread system regex REQUIRED)
include_directories(${BOOST_INCLUDE_DIR})

# Find yaml-cpp Library
find_package(PkgConfig)
pkg_check_modules(YamlCpp yaml-cpp)

# Find OpenCV library
find_package(OpenCV REQUIRED)

# Find PCL library
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Find Suitesparse library
find_package(SuiteSparse REQUIRED)
include_directories(${CHOLMOD_INCLUDE_DIR})

# Find G2O Library
find_package(G2O REQUIRED)
# select required components
SET(G2O_LIBRARIES ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_CSPARSE_EXTENSION} ${G2O_TYPES_SBA})

#ds specify additional locations of header files
include_directories(
  ${G2O_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  src
)

#ds build info
message("using OpenCV version: '${OpenCV_VERSION}' (${OpenCV_DIR})")
message("using Eigen version: '3' (${EIGEN3_INCLUDE_DIR})")
message("using g2o installation: '${G2O_INCLUDE_DIR}'")

#ds set additional flags
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" OR "${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=gnu++11 -march=native -Wall -Ofast -DNDEBUG -fPIC")
  message("adding flags for build type '${CMAKE_BUILD_TYPE}': '--std=gnu++11 -march=native -Wall -Ofast -DNDEBUG -fPIC'")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=gnu++11 -march=native -Wall -O0 -g -fPIC -fstack-check")
  message("adding flags for build type  '${CMAKE_BUILD_TYPE}': '--std=gnu++11 -march=native -Wall -O0 -g -fPIC -fstack-check'")
endif()

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES sptam
   CATKIN_DEPENDS roscpp std_msgs sensor_msgs geometry_msgs cv_bridge message_filters image_geometry pcl_ros nav_msgs tf_conversions
#  DEPENDS system_lib
)

###########
## Build ##
###########

# Build the sptam library
add_subdirectory( src/sptam )

set( SPTAM_LIBRARIES
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ${PCL_LIBRARIES}
  ${catkin_LIBRARIES}
  ${G2O_LIBRARIES}
  ${SUITESPARSE_LIBRARIES}
  ${YamlCpp_LIBRARIES}
  cxsparse
  sptam
)

#ds set up catkin package (exported components)
catkin_package(
  INCLUDE_DIRS
  ${G2O_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  src

  LIBRARIES
  SPTAM_LIBRARIES
)

# Build the sptam node
add_subdirectory( src/ros )

#############
## Install ##
#############

#TODO

#############
## Testing ##
#############

#TODO
