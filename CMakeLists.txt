CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project(YFaceSwap)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D DLIB_NO_GUI_SUPPORT")

add_definitions(-DDLIB_NO_GUI_SUPPORT)
#remove_definitions(-DDLIB_CHECK_FOR_VERSION_MISMATCH)
#remove_definitions(-DDLIB_VERSION_MISMATCH_CHECK__EXPECTED_VERSION_19_17_0)

#set(OUTPUT_NAME "YFaceSwapApp")
set(OUTPUT_NAME "YFaceSwapLib")

set(OPENCV_DIR "/home/michael/DevelopProj/Thirdparty/OpenCV/opencv-3.4.0/")
set(DLIB_DIR "/home/michael/DevelopProj/Thirdparty/DLib/dlib-19.17")

find_package(OpenCV 3.4.0 REQUIRED)
find_package(dlib 19.17 REQUIRED)
#find_package(Threads REQUIRED)

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

message(STATUS "PROJECT_SOURCE_DIR is: ${PROJECT_SOURCE_DIR}")

INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/YFaceSwap/FaceSwap")
INCLUDE_DIRECTORIES("/usr/local/include")
INCLUDE_DIRECTORIES("/usr/lib/jvm/java-8-openjdk-amd64/include")
INCLUDE_DIRECTORIES("/usr/lib/jvm/java-8-openjdk-amd64/include/linux")
# add_executable(${OUTPUT_NAME}
# ${PROJECT_SOURCE_DIR}/YFaceSwap/main.cpp 
# ${DLIB_DIR}/dlib/all/source.cpp
# ${PROJECT_SOURCE_DIR}/YFaceSwap/FaceSwap/FaceDetector.cpp 
# ${PROJECT_SOURCE_DIR}/YFaceSwap/FaceSwap/FaceExchanger.cpp
# ${PROJECT_SOURCE_DIR}/YFaceSwap/FaceSwap/FaceDetectorAndTracker.cpp
# ${PROJECT_SOURCE_DIR}/YFaceSwap/FaceSwap/FaceSwapper.cpp)


add_library(${OUTPUT_NAME}
${PROJECT_SOURCE_DIR}/YFaceSwap/main.cpp 
${DLIB_DIR}/dlib/all/source.cpp
${PROJECT_SOURCE_DIR}/YFaceSwap/FaceSwap/FaceDetector.cpp 
${PROJECT_SOURCE_DIR}/YFaceSwap/FaceSwap/FaceExchanger.cpp
${PROJECT_SOURCE_DIR}/YFaceSwap/FaceSwap/FaceDetectorAndTracker.cpp
${PROJECT_SOURCE_DIR}/YFaceSwap/FaceSwap/FaceSwapper.cpp
${PROJECT_SOURCE_DIR}/Java/src/com_yfaceswap_FaceSwapLib.cpp)

set_target_properties(${OUTPUT_NAME} PROPERTIES SUFFIX ".so")

link_directories("/usr/local/lib")
target_link_libraries(${OUTPUT_NAME}
${OpenCV_LIBS}
#${DLIB_LIBRARIES}
#dlib::dlib
lapack
blas)