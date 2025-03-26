#define NOMINMAX
#include <k4a/k4a.h>                              // Orbbec SDK 헤더
#include <k4abt.h>                                // Azure Kinect Body Tracking SDK 헤더
#include <open3d/visualization/visualizer/VisualizerWithKeyCallback.h> // Open3D 시각화 (키 콜백 포함)
#include <open3d/visualization/visualizer/ViewControl.h>                // Open3D 뷰 컨트롤 헤더
#include <open3d/geometry/PointCloud.h>             // Open3D 포인트 클라우드 자료형
#include <open3d/geometry/LineSet.h>                // Open3D 선 세트 자료형
#include <open3d/geometry/TriangleMesh.h>           // Open3D 삼각 메쉬 자료형
#include <opencv2/opencv.hpp>                      // OpenCV 헤더 (영상 처리)
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>

//---------------------------------------------------------
// Body Tracking 관련 설정
//---------------------------------------------------------

// 전체 Body Tracking 관절 개수 (Azure Kinect Body Tracking SDK에서 정의된 값)
static const int kNumBodyJoints = K4ABT_JOINT_COUNT;

// 스켈레톤 연결 정보: 각 관절 사이를 연결할 쌍들을 정의 (예: 골반->척추 등)
static const std::vector<std::pair<int, int>> kBodySkeletonConnection = {
    {K4ABT_JOINT_PELVIS,        K4ABT_JOINT_SPINE_NAVEL},
    {K4ABT_JOINT_SPINE_NAVEL,   K4ABT_JOINT_SPINE_CHEST},
    {K4ABT_JOINT_SPINE_CHEST,   K4ABT_JOINT_NECK},
    {K4ABT_JOINT_NECK,          K4ABT_JOINT_HEAD},

    {K4ABT_JOINT_SPINE_CHEST,   K4ABT_JOINT_CLAVICLE_LEFT},
    {K4ABT_JOINT_CLAVICLE_LEFT, K4ABT_JOINT_SHOULDER_LEFT},
    {K4ABT_JOINT_SHOULDER_LEFT, K4ABT_JOINT_ELBOW_LEFT},
    {K4ABT_JOINT_ELBOW_LEFT,    K4ABT_JOINT_WRIST_LEFT},
    {K4ABT_JOINT_WRIST_LEFT,    K4ABT_JOINT_HAND_LEFT},

    {K4ABT_JOINT_SPINE_CHEST,   K4ABT_JOINT_CLAVICLE_RIGHT},
    {K4ABT_JOINT_CLAVICLE_RIGHT,K4ABT_JOINT_SHOULDER_RIGHT},
    {K4ABT_JOINT_SHOULDER_RIGHT,K4ABT_JOINT_ELBOW_RIGHT},
    {K4ABT_JOINT_ELBOW_RIGHT,   K4ABT_JOINT_WRIST_RIGHT},
    {K4ABT_JOINT_WRIST_RIGHT,   K4ABT_JOINT_HAND_RIGHT},

    {K4ABT_JOINT_PELVIS,        K4ABT_JOINT_HIP_LEFT},
    {K4ABT_JOINT_HIP_LEFT,      K4ABT_JOINT_KNEE_LEFT},
    {K4ABT_JOINT_KNEE_LEFT,     K4ABT_JOINT_ANKLE_LEFT},
    {K4ABT_JOINT_ANKLE_LEFT,    K4ABT_JOINT_FOOT_LEFT},

    {K4ABT_JOINT_PELVIS,        K4ABT_JOINT_HIP_RIGHT},
    {K4ABT_JOINT_HIP_RIGHT,     K4ABT_JOINT_KNEE_RIGHT},
    {K4ABT_JOINT_KNEE_RIGHT,    K4ABT_JOINT_ANKLE_RIGHT},
    {K4ABT_JOINT_ANKLE_RIGHT,   K4ABT_JOINT_FOOT_RIGHT},
};

//---------------------------------------------------------
// 스켈레톤의 외형 설정 (기본 색상, 크기 배율, 색상 프리셋 등)
//---------------------------------------------------------
static Eigen::Vector3d g_skelColor(1.0, 1.0, 0.0);  // 기본 스켈레톤 색상: 노란색
static double g_skelSizeScale = 1.0;                // 스켈레톤 메쉬의 크기 배율 (기본: 1.0)
static std::vector<Eigen::Vector3d> g_skelColorPresets = {
    {1.0, 1.0, 0.0}, // 노란색
    {1.0, 0.0, 0.0}, // 빨간색
    {0.0, 1.0, 0.0}, // 초록색
    {0.0, 0.0, 1.0}  // 파란색
};
static int g_currentSkelColorIndex = 0;             // 현재 사용중인 색상 인덱스

//---------------------------------------------------------
// 작은 Sphere와 Cylinder를 이용하여 스켈레톤 메쉬를 생성하는 유틸리티 함수들
//---------------------------------------------------------

// 주어진 중심과 반지름을 갖는 구(스피어) 메쉬 생성
static std::shared_ptr<open3d::geometry::TriangleMesh>
CreateSphereMesh(const Eigen::Vector3d &center, double radius = 0.02, int resolution = 10)
{
    // 스켈레톤 크기 배율을 반영하여 구 생성 후, center로 평행 이동
    auto sphere = open3d::geometry::TriangleMesh::CreateSphere(radius * g_skelSizeScale, resolution);
    sphere->Translate(center, /*relative=*/true);
    return sphere;
}

// 두 점 p1과 p2를 연결하는 원통(Cylinder) 메쉬 생성
static std::shared_ptr<open3d::geometry::TriangleMesh>
CreateCylinderMesh(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, double radius = 0.01, int resolution = 20)
{
    Eigen::Vector3d diff = p2 - p1;
    double height = diff.norm();
    // 두 점이 거의 같은 경우 구를 생성하여 반환
    if (height < 1e-6)
        return open3d::geometry::TriangleMesh::CreateSphere(0.0);
    // 원통 생성 (배율 적용)
    auto cyl = open3d::geometry::TriangleMesh::CreateCylinder(radius * g_skelSizeScale, height, resolution, 2);
    
    // 기본 원통은 z축 방향이므로, 두 점의 방향에 맞게 회전시킴
    Eigen::Vector3d axisZ(0, 0, 1);
    Eigen::Vector3d axisNew = diff.normalized();
    Eigen::Vector3d cross = axisZ.cross(axisNew);
    double c_norm = cross.norm();
    if (c_norm > 1e-6) {
        cross /= c_norm;
        double angle = std::acos(axisZ.dot(axisNew));
        cyl->Rotate(Eigen::AngleAxisd(angle, cross).toRotationMatrix(), Eigen::Vector3d(0, 0, 0));
    }
    // 원통을 시작점 p1로 평행 이동
    cyl->Translate(p1, /*relative=*/true);
    return cyl;
}

//---------------------------------------------------------
// 관절 좌표들을 바탕으로 스켈레톤 메쉬를 생성하는 함수
// 각 관절에 대해 구(Sphere)를 생성하고, 연결 관계에 따라 원통(Cylinder)을 생성하여 합침
//---------------------------------------------------------
static std::shared_ptr<open3d::geometry::TriangleMesh>
CreateSkeletonMesh(const std::vector<Eigen::Vector3d> &joints3D)
{
    using namespace open3d::geometry;
    auto skeletonMesh = std::make_shared<TriangleMesh>();
    
    // 각 관절 위치에 구를 생성하여 스켈레톤 메쉬에 추가
    for (size_t j = 0; j < joints3D.size(); j++) {
        if (joints3D[j].norm() < 1e-6)
            continue;
        auto sphere = CreateSphereMesh(joints3D[j], /*radius=*/0.02);
        sphere->PaintUniformColor(g_skelColor); // 구에 기본 색상 적용
        *skeletonMesh += *sphere;
    }
    // 관절 사이 연결 정보를 바탕으로 원통을 생성하여 메쉬에 추가
    for (auto &conn : kBodySkeletonConnection) {
        int i1 = conn.first, i2 = conn.second;
        if (i1 < (int)joints3D.size() && i2 < (int)joints3D.size()) {
            Eigen::Vector3d p1 = joints3D[i1];
            Eigen::Vector3d p2 = joints3D[i2];
            if (p1.norm() < 1e-6 || p2.norm() < 1e-6)
                continue;
            auto cyl = CreateCylinderMesh(p1, p2, /*radius=*/0.01);
            cyl->PaintUniformColor(g_skelColor);
            *skeletonMesh += *cyl;
        }
    }
    return skeletonMesh;
}

//---------------------------------------------------------
// Open3D Visualizer 확장 클래스 (키 입력 처리 포함)
//---------------------------------------------------------
class MyVisualizer : public open3d::visualization::VisualizerWithKeyCallback {
public:
    MyVisualizer() : exit_flag_(false) {}
    // 종료 플래그 반환
    bool ShouldExit() const { return exit_flag_; }
    // ESC 키 입력시 종료하도록 콜백 등록
    void RegisterExitKey() {
        RegisterKeyCallback(GLFW_KEY_ESCAPE, [this](open3d::visualization::Visualizer*) -> bool {
            exit_flag_ = true;
            return true;
        });
    }
private:
    bool exit_flag_;
};

//---------------------------------------------------------
// 전역 변수 (카메라 FOV 모드, 영상 사용 여부 등)
//---------------------------------------------------------

// FOV 모드 선택 변수 (1: NFOV Unbinned, 2: WFOV Unbinned, 3: WFOV Binned, 4: NFOV 2x2 Binned)
static int g_fovMode = 1;
// 컬러와 Depth 영상 사용 여부 (항상 사용하도록 설정)
static bool g_useColor = true;
static bool g_useDepth = true;
// 포인트 클라우드 시각화 사용 여부
static bool g_usePointCloud = false;

// 선택한 FOV 모드에 따른 해상도 (기본 값: NFOV Unbinned 640x576)
static int g_fovWidth = 640;
static int g_fovHeight = 576;

// Azure Kinect 디바이스, 변환, 바디 트래커 관련 변수
static k4a_device_t g_device = nullptr;
static k4a_transformation_t g_transform = nullptr;
static k4abt_tracker_t g_tracker = nullptr;

// Open3D 시각화 관련 전역 변수
static MyVisualizer g_vis;
static std::shared_ptr<open3d::geometry::PointCloud> g_pcPtr;
static std::shared_ptr<open3d::geometry::TriangleMesh> g_skelMesh;

// 컬러와 Depth 영상이 결합되어 보여질 창 이름
static std::string g_combinedWin = "Color&Depth";

// 포인트 클라우드 및 바디가 처음 업데이트 되었는지 여부 플래그
static bool g_firstCloud = true;
static bool g_firstBodyDetected = false;

//---------------------------------------------------------
// 그리드(LineSet) 생성 함수 (배경에 표시할 격자)
//---------------------------------------------------------
static std::shared_ptr<open3d::geometry::LineSet> CreateGridLineSet(int gridSize = 10, float step = 1.0f)
{
    using namespace open3d::geometry;
    auto ls = std::make_shared<LineSet>();
    std::vector<Eigen::Vector3d> pts;
    std::vector<Eigen::Vector2i> lns;
    int idx = 0;
    // x축 방향 선 생성
    for (int x = -gridSize; x <= gridSize; x++) {
        Eigen::Vector3d p1(x * step, 0.0, -gridSize * step);
        Eigen::Vector3d p2(x * step, 0.0, gridSize * step);
        pts.push_back(p1);
        pts.push_back(p2);
        lns.push_back({ idx * 2, idx * 2 + 1 });
        idx++;
    }
    // z축 방향 선 생성
    for (int z = -gridSize; z <= gridSize; z++) {
        Eigen::Vector3d p1(-gridSize * step, 0.0, z * step);
        Eigen::Vector3d p2(gridSize * step, 0.0, z * step);
        pts.push_back(p1);
        pts.push_back(p2);
        lns.push_back({ idx * 2, idx * 2 + 1 });
        idx++;
    }
    ls->points_ = pts;
    ls->lines_ = lns;
    // 그리드 색상 (회색 계열) 적용
    ls->PaintUniformColor({ 0.6, 0.6, 0.6 });
    return ls;
}

//---------------------------------------------------------
// Open3D 시각화 창 재생성 함수
// FOV 모드에 따라 창 크기 및 내부 Open3D 객체들을 다시 생성
//---------------------------------------------------------
static bool RecreateVisualizerWindow(MyVisualizer &vis)
{
    using namespace open3d::visualization;
    vis.Close();
    vis.DestroyVisualizerWindow();
    int pcWidth, pcHeight;
    if (g_fovMode == 1) { pcWidth = 640; pcHeight = 576; }
    else if (g_fovMode == 2) { pcWidth = 1024; pcHeight = 1024; }
    else if (g_fovMode == 3) { pcWidth = 512; pcHeight = 512; }
    else if (g_fovMode == 4) { pcWidth = 320; pcHeight = 288; }
    if (!vis.CreateVisualizerWindow("3D Point Cloud", pcWidth, pcHeight))
        return false;
    vis.RegisterExitKey();
    g_pcPtr = std::make_shared<open3d::geometry::PointCloud>();
    vis.AddGeometry(g_pcPtr);
    auto axes = open3d::geometry::TriangleMesh::CreateCoordinateFrame(1.0, {0, 0, 0});
    vis.AddGeometry(axes);
    auto grid = CreateGridLineSet(10, 0.2f);
    vis.AddGeometry(grid);
    g_skelMesh = std::make_shared<open3d::geometry::TriangleMesh>();
    g_skelMesh->Clear();
    vis.GetRenderOption().background_color_ = {0, 0, 0};
    vis.AddGeometry(g_skelMesh);
    return true;
}

//---------------------------------------------------------
// 불필요한 캡처 프레임 제거 함수
// (버퍼에 남은 프레임들을 반복적으로 제거)
//---------------------------------------------------------
static void DrainExtraFrames()
{
    while (true) {
        k4a_capture_t tmp = nullptr;
        auto r = k4a_device_get_capture(g_device, &tmp, 0);
        if (r == K4A_WAIT_RESULT_SUCCEEDED && tmp)
            k4a_capture_release(tmp);
        else
            break;
    }
}

//---------------------------------------------------------
// Orbbec SDK 디바이스 및 관련 객체 초기화 함수
// 카메라 설정, 캘리브레이션, 트래커 생성 등을 수행
//---------------------------------------------------------
bool InitializeAll()
{
    // 디바이스 열기
    if (K4A_FAILED(k4a_device_open(K4A_DEVICE_DEFAULT, &g_device))) {
        std::cerr << "device_open fail.\n";
        return false;
    }
    // 기본 디바이스 설정 (모든 기능 비활성화 후 필요한 항목만 활성화)
    k4a_device_configuration_t cfg = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    if (g_useColor) {
        cfg.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
        cfg.color_resolution = K4A_COLOR_RESOLUTION_720P;
    } else {
        cfg.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    }
    if (g_useDepth) {
        // 선택된 FOV 모드에 따른 Depth 모드 설정
        if (g_fovMode == 1)
            cfg.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;         // NFOV Unbinned: 640×576
        else if (g_fovMode == 2)
            cfg.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;          // WFOV Unbinned: 1024×1024
        else if (g_fovMode == 3)
            cfg.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;          // WFOV Binned: 512×512
        else if (g_fovMode == 4)
            cfg.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;          // NFOV 2x2 Binned: 320×288
    } else {
        cfg.depth_mode = K4A_DEPTH_MODE_OFF;
    }
    cfg.camera_fps = K4A_FRAMES_PER_SECOND_15;
    cfg.synchronized_images_only = true;
    
    // 카메라 시작
    if (K4A_FAILED(k4a_device_start_cameras(g_device, &cfg))) {
        std::cerr << "start_cameras fail.\n";
        k4a_device_close(g_device);
        g_device = nullptr;
        return false;
    }
    
    // 컬러 또는 Depth 사용 시 캘리브레이션 및 트래커 생성
    if (g_useColor || g_useDepth) {
        k4a_calibration_t cal;
        k4a_depth_mode_t selected_mode = (g_fovMode == 1 ? K4A_DEPTH_MODE_NFOV_UNBINNED :
                                        (g_fovMode == 2 ? K4A_DEPTH_MODE_WFOV_UNBINNED :
                                        (g_fovMode == 3 ? K4A_DEPTH_MODE_WFOV_2X2BINNED : K4A_DEPTH_MODE_NFOV_2X2BINNED)));
        if (K4A_FAILED(k4a_device_get_calibration(g_device, selected_mode, K4A_COLOR_RESOLUTION_720P, &cal))) {
            std::cerr << "get_calibration fail.\n";
            return false;
        }
        g_transform = k4a_transformation_create(&cal);
        if (!g_transform)
            return false;
        k4abt_tracker_configuration_t tcfg = K4ABT_TRACKER_CONFIG_DEFAULT;
        if (K4A_FAILED(k4abt_tracker_create(&cal, tcfg, &g_tracker))) {
            std::cerr << "k4abt_tracker_create fail.\n";
            return false;
        }
    }
    
    // 단일 "Color&Depth" 창 생성 (좌우 결합)
    if (g_useColor && g_useDepth) {
        cv::namedWindow(g_combinedWin, cv::WINDOW_NORMAL);
        cv::resizeWindow(g_combinedWin, g_fovWidth * 2, g_fovHeight);
    }
    // 포인트 클라우드 시각화 사용 시 Open3D 창 생성
    if (g_usePointCloud) {
        int pcWidth, pcHeight;
        if (g_fovMode == 1) { pcWidth = 640; pcHeight = 576; }
        else if (g_fovMode == 2) { pcWidth = 1024; pcHeight = 1024; }
        else if (g_fovMode == 3) { pcWidth = 512; pcHeight = 512; }
        else if (g_fovMode == 4) { pcWidth = 320; pcHeight = 288; }
        if (!g_vis.CreateVisualizerWindow("3D Point Cloud", pcWidth, pcHeight))
            return false;
        g_vis.RegisterExitKey();
        g_pcPtr = std::make_shared<open3d::geometry::PointCloud>();
        g_vis.AddGeometry(g_pcPtr);
        auto axes = open3d::geometry::TriangleMesh::CreateCoordinateFrame(1.0, {0, 0, 0});
        g_vis.AddGeometry(axes);
        auto grid = CreateGridLineSet(10, 0.2f);
        g_vis.AddGeometry(grid);
        g_skelMesh = std::make_shared<open3d::geometry::TriangleMesh>();
        g_skelMesh->Clear();
        g_vis.AddGeometry(g_skelMesh);
        g_vis.GetRenderOption().background_color_ = {0, 0, 0};
    }
    g_firstCloud = true;
    g_firstBodyDetected = false;
    return true;
}

//---------------------------------------------------------
// 자원 해제 함수: 디바이스, 트래커, 변환 객체, Open3D 창 등 모두 정리
//---------------------------------------------------------
void FinalizeAll()
{
    if (g_usePointCloud) {
        g_vis.Close();
        g_vis.DestroyVisualizerWindow();
    }
    if (g_tracker) {
        k4abt_tracker_destroy(g_tracker);
        g_tracker = nullptr;
    }
    if (g_transform) {
        k4a_transformation_destroy(g_transform);
        g_transform = nullptr;
    }
    if (g_device) {
        k4a_device_stop_cameras(g_device);
        k4a_device_close(g_device);
        g_device = nullptr;
    }
    cv::destroyWindow(g_combinedWin);
}

//---------------------------------------------------------
// 색상과 Depth 영상을 정렬하여 출력 이미지 생성 함수
//---------------------------------------------------------
static k4a_image_t CreateAlignedColorImage(
    k4a_transformation_t transform,
    k4a_image_t depthImg,
    k4a_image_t colorImg)
{
    if (!transform || !depthImg || !colorImg)
        return nullptr;
    int dw = k4a_image_get_width_pixels(depthImg);
    int dh = k4a_image_get_height_pixels(depthImg);
    k4a_image_t outImg = nullptr;
    if (K4A_SUCCEEDED(k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32, dw, dh, dw * 4, &outImg))) {
        if (K4A_FAILED(k4a_transformation_color_image_to_depth_camera(transform, depthImg, colorImg, outImg))) {
            k4a_image_release(outImg);
            return nullptr;
        }
        return outImg;
    }
    return nullptr;
}

//---------------------------------------------------------
// Depth 영상 위에 2D Body Skeleton을 그리는 함수 (OpenCV 사용)
//---------------------------------------------------------
static void DrawBody2DOnDepth(cv::Mat &depthVis,
    const k4abt_skeleton_t &skeleton,
    const k4a_calibration_t &cal)
{
    // 관절 간 선 그리기
    for (auto &conn : kBodySkeletonConnection) {
        k4a_float2_t p1, p2; int v1 = 0, v2 = 0;
        k4a_calibration_3d_to_2d(&cal, &skeleton.joints[conn.first].position,
            K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &p1, &v1);
        k4a_calibration_3d_to_2d(&cal, &skeleton.joints[conn.second].position,
            K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &p2, &v2);
        if (v1 && v2) {
            cv::line(depthVis, { int(p1.xy.x), int(p1.xy.y) },
                     { int(p2.xy.x), int(p2.xy.y) }, cv::Scalar(255), 2);
        }
    }
    // 각 관절에 원(circle) 그리기
    for (int j = 0; j < kNumBodyJoints; j++) {
        k4a_float2_t p2d; int valid = 0;
        k4a_calibration_3d_to_2d(&cal, &skeleton.joints[j].position,
            K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &p2d, &valid);
        if (valid) {
            cv::circle(depthVis, { int(p2d.xy.x), int(p2d.xy.y) },
                       3, cv::Scalar(255), -1);
        }
    }
}

//---------------------------------------------------------
// main 함수: 프로그램의 진입점
// 사용자로부터 FOV 모드 및 포인트 클라우드 사용 여부를 입력받아
// Orbbec SDK를 초기화하고, 영상 처리 및 Azure Kinect Body Tracking, 시각화 작업을 수행함
//---------------------------------------------------------
int main()
{
    // FOV 모드 선택 메뉴 출력
    // 1: NFOV Unbinned (640x576), 2: WFOV Unbinned (1024x1024),
    // 3: WFOV Binned (512x512), 4: NFOV 2x2 Binned (320x288)
    std::cout << "Select FOV mode:\n"
              << " 1: NFOV Unbinned (640x576)\n"
              << " 2: WFOV Unbinned (1024x1024)\n"
              << " 3: WFOV Binned (512x512)\n"
              << " 4: NFOV 2x2 Binned (320x288)\n"
              << "Enter number: ";
    std::cin >> g_fovMode;
    if (g_fovMode == 1) {
        g_fovWidth = 640;
        g_fovHeight = 576;
    } else if (g_fovMode == 2) {
        g_fovWidth = 1024;
        g_fovHeight = 1024;
    } else if (g_fovMode == 3) {
        g_fovWidth = 512;
        g_fovHeight = 512;
    } else if (g_fovMode == 4) {
        g_fovWidth = 320;
        g_fovHeight = 288;
    } else {
        g_fovMode = 1;
        g_fovWidth = 640;
        g_fovHeight = 576;
    }
    // 컬러와 Depth 영상은 항상 사용
    g_useColor = true;
    g_useDepth = true;
    
    // 포인트 클라우드 사용 여부 입력 받음
    std::cout << "Use PointCloud? (y/n): ";
    {
        std::string s;
        std::cin >> s;
        if (!s.empty() && (s[0] == 'y' || s[0] == 'Y'))
            g_usePointCloud = true;
    }
    
    // 포인트 클라우드 창이 사용되지 않는 경우 관련 함수 호출하지 않도록 처리
RESTART_ALL:
    if (!InitializeAll()) {
        std::cerr << "init fail.\n";
        return -1;
    }
    
    bool quit = false;
    while (!quit) {
        // 포인트 클라우드 창 업데이트 (사용 중이면 이벤트 처리)
        if (g_usePointCloud) {
            g_vis.PollEvents();
            if (g_vis.ShouldExit())
                break;
        }
        k4a_capture_t cap = nullptr;
        auto r = k4a_device_get_capture(g_device, &cap, 15);
        if (r == K4A_WAIT_RESULT_TIMEOUT)
            continue;
        else if (r == K4A_WAIT_RESULT_FAILED) {
            std::cerr << "get_capture fail.\n";
            FinalizeAll();
            goto RESTART_ALL;
        }
        if (!cap)
            continue;
        // Depth와 Color 이미지 가져오기
        k4a_image_t depthImg = (g_useDepth) ? k4a_capture_get_depth_image(cap) : nullptr;
        k4a_image_t colorImg = (g_useColor) ? k4a_capture_get_color_image(cap) : nullptr;
        
        //--------------- Depth 이미지 처리 ---------------
        cv::Mat dv;
        if (depthImg) {
            int dw = k4a_image_get_width_pixels(depthImg);
            int dh = k4a_image_get_height_pixels(depthImg);
            uint16_t* dptr = (uint16_t*)k4a_image_get_buffer(depthImg);
            cv::Mat dmat(dh, dw, CV_16UC1, dptr);
            dv = cv::Mat(dh, dw, CV_8UC1);
            // 각 픽셀 값에 대해 범위 조정 (4000 초과 시 255, 그 외는 16으로 나눔)
            for (int rr = 0; rr < dh; rr++) {
                for (int cc = 0; cc < dw; cc++) {
                    uint16_t v = dmat.at<uint16_t>(rr, cc);
                    dv.at<uchar>(rr, cc) = (v > 4000) ? 255 : (v / 16);
                }
            }
        }
        
        //--------------- Color 이미지 처리 ---------------
        cv::Mat showColor;
        if (g_useColor && colorImg) {
            int cw = k4a_image_get_width_pixels(colorImg);
            int ch = k4a_image_get_height_pixels(colorImg);
            int cstride = k4a_image_get_stride_bytes(colorImg);
            uint8_t* cbuf = (uint8_t*)k4a_image_get_buffer(colorImg);
            cv::Mat colorBGRA(ch, cw, CV_8UC4, (void*)cbuf, cstride);
            cv::Mat colorBGR;
            // BGRA -> BGR로 색상 변환
            cv::cvtColor(colorBGRA, colorBGR, cv::COLOR_BGRA2BGR);
            // 선택된 FOV 모드 해상도에 맞게 리사이즈
            cv::resize(colorBGR, showColor, { g_fovWidth, g_fovHeight });
        }
        
        //--------------- Body Tracking 및 Skeleton 생성 ---------------
        if (g_tracker && cap) {
            if (K4A_FAILED(k4abt_tracker_enqueue_capture(g_tracker, cap, 0)))
                std::cerr << "tracker_enqueue fail.\n";
            else {
                k4abt_frame_t bodyFrame = nullptr;
                auto popRes = k4abt_tracker_pop_result(g_tracker, &bodyFrame, 100);
                if (popRes == K4A_WAIT_RESULT_SUCCEEDED && bodyFrame) {
                    size_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
                    cv::Mat dvCopy = dv.clone();
                    if (g_usePointCloud) {
                        if (g_skelMesh)
                            g_skelMesh->Clear();
                    }
                    // 각 인식된 Body에 대해 Skeleton 처리
                    for (size_t i = 0; i < numBodies; i++) {
                        k4abt_skeleton_t skeleton;
                        k4abt_frame_get_body_skeleton(bodyFrame, i, &skeleton);
                        k4a_depth_mode_t selected_mode = (g_fovMode == 1 ? K4A_DEPTH_MODE_NFOV_UNBINNED :
                                                        (g_fovMode == 2 ? K4A_DEPTH_MODE_WFOV_UNBINNED :
                                                        (g_fovMode == 3 ? K4A_DEPTH_MODE_WFOV_2X2BINNED : K4A_DEPTH_MODE_NFOV_2X2BINNED)));
                        k4a_calibration_t calibration;
                        if (K4A_RESULT_SUCCEEDED == k4a_device_get_calibration(g_device,
                                selected_mode,
                                K4A_COLOR_RESOLUTION_720P, &calibration))
                        {
                            // 2D로 Body Skeleton 그리기
                            DrawBody2DOnDepth(dvCopy, skeleton, calibration);
                        }
                        
                        // 왼손, 오른손의 거리 및 좌표 정보 표시
                        {
                            int leftHandIndex = 8;
                            k4a_float2_t leftHand2d;
                            int validLeft = 0;
                            k4a_calibration_3d_to_2d(&calibration, &skeleton.joints[leftHandIndex].position,
                                K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &leftHand2d, &validLeft);
                            if (validLeft) {
                                k4a_float3_t leftPos = skeleton.joints[leftHandIndex].position;
                                double leftX = leftPos.v[0] / 1000.0;
                                double leftY = -leftPos.v[1] / 1000.0;
                                double leftZ = -leftPos.v[2] / 1000.0;
                                double leftDistance = std::sqrt(leftX * leftX + leftY * leftY + leftZ * leftZ);
                                std::string leftText = "L: " + std::to_string(leftDistance).substr(0, 4) + " m";
                                cv::putText(dvCopy, leftText, cv::Point((int)leftHand2d.xy.x + 5, (int)leftHand2d.xy.y - 5),
                                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255), 2);
                                std::string leftCoord = "(" + std::to_string(leftX).substr(0, 4) + ", " +
                                                        std::to_string(leftY).substr(0, 4) + ", " +
                                                        std::to_string(leftZ).substr(0, 4) + ")";
                                cv::putText(dvCopy, leftCoord, cv::Point((int)leftHand2d.xy.x + 5, (int)leftHand2d.xy.y + 15),
                                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255), 2);
                            }
                        }
                        {
                            int rightHandIndex = 15;
                            k4a_float2_t rightHand2d;
                            int validRight = 0;
                            k4a_calibration_3d_to_2d(&calibration, &skeleton.joints[rightHandIndex].position,
                                K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &rightHand2d, &validRight);
                            if (validRight) {
                                k4a_float3_t rightPos = skeleton.joints[rightHandIndex].position;
                                double rightX = rightPos.v[0] / 1000.0;
                                double rightY = -rightPos.v[1] / 1000.0;
                                double rightZ = -rightPos.v[2] / 1000.0;
                                double rightDistance = std::sqrt(rightX * rightX + rightY * rightY + rightZ * rightZ);
                                std::string rightText = "R: " + std::to_string(rightDistance).substr(0, 4) + " m";
                                cv::putText(dvCopy, rightText, cv::Point((int)rightHand2d.xy.x + 5, (int)rightHand2d.xy.y - 5),
                                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255), 2);
                                std::string rightCoord = "(" + std::to_string(rightX).substr(0, 4) + ", " +
                                                         std::to_string(rightY).substr(0, 4) + ", " +
                                                         std::to_string(rightZ).substr(0, 4) + ")";
                                cv::putText(dvCopy, rightCoord, cv::Point((int)rightHand2d.xy.x + 5, (int)rightHand2d.xy.y + 15),
                                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255), 2);
                            }
                        }
                        // 3D 좌표로 스켈레톤 메쉬 생성
                        std::vector<Eigen::Vector3d> j3d(kNumBodyJoints, { 0, 0, 0 });
                        for (int j = 0; j < kNumBodyJoints; j++) {
                            k4a_float3_t pos = skeleton.joints[j].position;
                            double X = pos.v[0] / 1000.0;
                            double Y = -pos.v[1] / 1000.0;
                            double Z = -pos.v[2] / 1000.0;
                            j3d[j] = { X, Y, Z };
                        }
                        auto newMesh = CreateSkeletonMesh(j3d);
                        if (g_usePointCloud)  // 포인트 클라우드 관련 메쉬 업데이트
                            *g_skelMesh += *newMesh;
                    }
                    // Depth 영상 및 Color 영상을 결합하여 출력
                    cv::Mat showDepth;
                    cv::resize(dvCopy, showDepth, { g_fovWidth, g_fovHeight });
                    cv::Mat showDepthColor;
                    cv::cvtColor(showDepth, showDepthColor, cv::COLOR_GRAY2BGR);
                    cv::Mat combined;
                    if (!showColor.empty() && !showDepthColor.empty())
                        cv::hconcat(showColor, showDepthColor, combined);
                    cv::imshow(g_combinedWin, combined);
                    
                    if (g_usePointCloud) {
                        g_vis.UpdateGeometry(g_skelMesh);
                        g_vis.UpdateRender();
                        if (!g_firstBodyDetected) {
                            g_firstBodyDetected = true;
                            g_vis.ResetViewPoint(true);
                        }
                        g_vis.PollEvents();
                    }
                    k4abt_frame_release(bodyFrame);
                }
            }
        }
        
        //--------------- 포인트 클라우드 생성 (Depth 이미지 기반) ---------------
        if (depthImg && g_usePointCloud && g_transform) {
            int dw = k4a_image_get_width_pixels(depthImg);
            int dh = k4a_image_get_height_pixels(depthImg);
            k4a_image_t pcImg = nullptr;
            if (K4A_SUCCEEDED(k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM, dw, dh, dw * 3 * sizeof(int16_t), &pcImg))) {
                if (K4A_SUCCEEDED(k4a_transformation_depth_image_to_point_cloud(g_transform, depthImg, K4A_CALIBRATION_TYPE_DEPTH, pcImg))) {
                    g_pcPtr->Clear();
                    g_pcPtr->points_.reserve(dw * dh);
                    g_pcPtr->colors_.reserve(dw * dh);
                    int16_t* b = (int16_t*)k4a_image_get_buffer(pcImg);
                    for (int i = 0; i < dw * dh; i++) {
                        int16_t xm = b[3 * i + 0];
                        int16_t ym = b[3 * i + 1];
                        int16_t zm = b[3 * i + 2];
                        if (zm == 0)
                            continue;
                        double X = (double)xm / 1000.0;
                        double Y = (double)(-ym) / 1000.0;
                        double Z = (double)(-zm) / 1000.0;
                        g_pcPtr->points_.push_back({ X, Y, Z });
                        double dist = sqrt(X * X + Y * Y + Z * Z);
                        double maxD = 4.0;
                        if (dist > maxD)
                            dist = maxD;
                        double t = dist / maxD;
                        g_pcPtr->colors_.push_back({ t, 0.0, 1.0 - t });
                    }
                    if (!g_pcPtr->points_.empty() && g_firstCloud) {
                        g_vis.ResetViewPoint(true);
                        g_firstCloud = false;
                    }
                    g_vis.UpdateGeometry(g_pcPtr);
                    g_vis.UpdateRender();
                }
                k4a_image_release(pcImg);
            }
        }
        
        // 이미지 및 캡처 객체 해제
        if (colorImg)
            k4a_image_release(colorImg);
        if (depthImg)
            k4a_image_release(depthImg);
        k4a_capture_release(cap);
        DrainExtraFrames();
        
        // 키 입력 처리: 'q' 또는 ESC면 종료, 'r'면 포인트 클라우드 창 리셋, 'd'면 재초기화
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            quit = true;
            break;
        } else if (key == 'r' || key == 'R') {
            if (g_usePointCloud) {
                g_vis.Close();
                g_vis.DestroyVisualizerWindow();
                int pcWidth, pcHeight;
                if (g_fovMode == 1) { pcWidth = 640; pcHeight = 576; }
                else if (g_fovMode == 2) { pcWidth = 1024; pcHeight = 1024; }
                else if (g_fovMode == 3) { pcWidth = 512; pcHeight = 512; }
                else if (g_fovMode == 4) { pcWidth = 320; pcHeight = 288; }
                if (!g_vis.CreateVisualizerWindow("3D Point Cloud", pcWidth, pcHeight))
                    break;
                g_vis.RegisterExitKey();
                g_pcPtr = std::make_shared<open3d::geometry::PointCloud>();
                g_vis.AddGeometry(g_pcPtr);
                auto axes = open3d::geometry::TriangleMesh::CreateCoordinateFrame(1.0, { 0, 0, 0 });
                g_vis.AddGeometry(axes);
                auto grid = CreateGridLineSet(10, 0.2f);
                g_vis.AddGeometry(grid);
                g_skelMesh = std::make_shared<open3d::geometry::TriangleMesh>();
                g_skelMesh->Clear();
                g_vis.AddGeometry(g_skelMesh);
                g_vis.GetRenderOption().background_color_ = { 0, 0, 0 };
                g_firstCloud = true;
                g_firstBodyDetected = false;
            }
        } else if (key == 'd' || key == 'D') {
            FinalizeAll();
            goto RESTART_ALL;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    FinalizeAll();
    return 0;
}