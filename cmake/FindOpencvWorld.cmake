########################################################################
# CMake module for finding OpencvWorld
#
# The following variables will be defined:
#
#  OPENCVWORLD_FOUND
#  OPENCVWORLD_INCLUDE_DIR
#  OPENCVWORLD_LIBRARY
#

unset(OpenCV_INCLUDE_DIR CACHE)
unset(OpenCV_FOUND CACHE)
unset(OpenCV_LIBRARY CACHE)


set(build_arch "x64")


find_path(OPENCVWORLD_INCLUDE_DIR
  NAMES opencv2/opencv.hpp
  PATHS ${PROJECT_SOURCE_DIR}/thirdparty/opencv/include
  NO_DEFAULT_PATH)
find_library(OPENCVWORLD_LIBRARY_debug
  NAMES opencv_world310d
  PATHS ${PROJECT_SOURCE_DIR}/thirdparty/opencv/${build_arch}/debug
  NO_DEFAULT_PATH)

find_library(OPENCVWORLD_LIBRARY_release
   NAMES opencv_world310
   PATHS ${PROJECT_SOURCE_DIR}/thirdparty/opencv/${build_arch}/release
   NO_DEFAULT_PATH)

#set(OPENCVWORLD_LIBRARY optimized ${OPENCVWORLD_LIBRARY_release} debug ${OPENCVWORLD_LIBRARY_debug})
set(OPENCVWORLD_LIBRARY ${OPENCVWORLD_LIBRARY_release})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpencvWorld DEFAULT_MSG OPENCVWORLD_LIBRARY OPENCVWORLD_INCLUDE_DIR)
