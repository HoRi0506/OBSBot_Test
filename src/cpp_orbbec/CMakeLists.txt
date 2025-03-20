cmake_minimum_required(VERSION 3.31.6)
project(cpp_orbbec CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# (필요 시) vcpkg toolchain
#set(CMAKE_TOOLCHAIN_FILE "C:/project/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "VCPkg toolchain file")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. OpenCV
set(OpenCV_DIR "C:/project/opencv/build/x64/vc16/lib")  # 실제 경로에 맞춰 수정 
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found.")
endif()

# 2. Open3D
set(CMAKE_PREFIX_PATH "C:/project/open3d")  # 실제 경로에 맞춰 수정
find_package(Open3D REQUIRED CONFIG)
if(Open3D_FOUND)
    message(STATUS "Open3D version: ${Open3D_VERSION}")
    include_directories(${Open3D_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Open3D not found.")
endif()

# 3. Orbbec SDK (Azure Kinect SDK Wrapper)
set(OrbbecSDK_DIR "C:/project/OrbbecSDK_K4A_Wrapper_v1.10.3_windows_202408091749")  # 실제 경로에 맞춰 수정
include_directories("${OrbbecSDK_DIR}/include")
link_directories("${OrbbecSDK_DIR}/lib")

# 여기서 Body Tracking(k4abt)은 사용 안 함
# set(K4ABT_DIR "C:/Program Files/Azure Kinect Body Tracking SDK")
# include_directories("${K4ABT_DIR}/sdk/include")
# link_directories("${K4ABT_DIR}/sdk/windows-desktop/amd64/release/lib")

# 4. 소스 코드
add_executable(${PROJECT_NAME} src/main.cpp)

# 5. 라이브러리 링크 (k4a만 링크)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        k4a
        Open3D::Open3D
        ${OpenCV_LIBS}
)

# OrbbecSDK DLL 복사 Post Build Command
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${OrbbecSDK_DIR}/bin"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)

# OpenCV DLL 복사 Post Build Command
if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/project/opencv/build/x64/vc16/bin/opencv_world4110d.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
else()
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/project/opencv/build/x64/vc16/bin/opencv_world4110.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()
