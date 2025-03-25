cmake_minimum_required(VERSION 3.31.6)
project(cpp_orbbec CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# vcpkg toolchain
# CMake projects should use: "-DCMAKE_TOOLCHAIN_FILE=C:/clone/vcpkg-master/scripts/buildsystems/vcpkg.cmake"
set(CMAKE_TOOLCHAIN_FILE "C:/clone/vcpkg-master/scripts/buildsystems/vcpkg.cmake" CACHE STRING "VCPkg toolchain file")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(quirc_DIR "C:/clone/vcpkg-master/installed/x64-windows/share/quirc")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};C:/clone/vcpkg-master/installed/x64-windows")

# 1. OpenCV
set(OpenCV_DIR "C:/clone/vcpkg-master/installed/x64-windows/share/opencv4")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found.")
endif()

# 2. Open3D
set(Open3D_DIR "C:/clone/open3d")
find_package(Open3D REQUIRED CONFIG)
if(Open3D_FOUND)
    message(STATUS "Open3D version: ${Open3D_VERSION}")
    include_directories(${Open3D_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Open3D not found.")
endif()

# 3. Orbbec SDK (Azure Kinect SDK Wrapper)
set(OrbbecSDK_DIR "C:/clone/OrbbecSDK_K4A_Wrapper_v1.10.3_windows_202408091749")
include_directories("${OrbbecSDK_DIR}/include")
link_directories("${OrbbecSDK_DIR}/lib")

# Azure Kinect Body Tracking SDK
set(AZSDK_DIR "C:/Program Files/Azure Kinect Body Tracking SDK/sdk")
include_directories("${AZSDK_DIR}/include")
link_directories("${AZSDK_DIR}/windows-desktop/amd64/release/lib")

# 4. 소스 코드
add_executable(${PROJECT_NAME} src/main.cpp)

# 5. 라이브러리 링크
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        k4a
        k4abt
        Open3D::Open3D
        ${OpenCV_LIBS}
)

# OrbbecSDK DLL 복사
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${OrbbecSDK_DIR}/bin"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)

# Open3D DLL 복사
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/clone/open3d/bin/Open3D.dll"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
        $<$<CONFIG:Release>:>)

# k4abt.dll 복사
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/Program Files/Azure Kinect Body Tracking SDK/sdk/windows-desktop/amd64/release/bin/k4abt.dll"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
        $<$<CONFIG:Release>:>)

# k4abt onnx 모델 복사
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/Program Files/Azure Kinect Body Tracking SDK/sdk/windows-desktop/amd64/release/bin/dnn_model_2_0_op11.onnx"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
        $<$<CONFIG:Release>:>
)

# (Lite 버전 등 추가 복사)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/Program Files/Azure Kinect Body Tracking SDK/sdk/windows-desktop/amd64/release/bin/dnn_model_2_0_lite_op11.onnx"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
        $<$<CONFIG:Release>:>
)

# onnxruntime.dll 복사
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/Program Files/Azure Kinect Body Tracking SDK/sdk/windows-desktop/amd64/release/bin/onnxruntime.dll"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
        $<$<CONFIG:Release>:>)
