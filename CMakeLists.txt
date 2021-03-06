cmake_minimum_required(VERSION 3.17)
SET(PROJECT_NAME KalmanFilter)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OPENCV_INCLUDE_DIR})

add_executable(${PROJECT_NAME} main.cpp main.cpp myKalman.h myUKF.h)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})

#[[
#设置CMAKE最低版本
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
#设置项目名称
SET(PROJECT_NAME Image_Test_01)
#建立项目
PROJECT(${PROJECT_NAME})
#寻找OpenCV库
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
#打印调试信息
MESSAGE(STATUS "Project: ${PROJECT_NAME}")
MESSAGE(STATUS "OpenCV library status:")
MESSAGE(STATUS "    version: ${OpenCV_VERSION}")
MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#获取代码，在项目中，将所有代码都放在src文件夹中
AUX_SOURCE_DIRECTORY(src DIR_SRCS)
MESSAGE(STATUS "Src file: ${DIR_SRCS}")
#编译可执行程序
ADD_EXECUTABLE(${PROJECT_NAME} ${DIR_SRCS})
#添加链接库
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})
]]

