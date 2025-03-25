#define NOMINMAX
#include <k4a/k4a.h>
#include <k4abt.h>
#include <open3d/visualization/visualizer/VisualizerWithKeyCallback.h>
#include <open3d/visualization/visualizer/ViewControl.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/TriangleMesh.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>

//---------------------------------------------------------
// Body Tracking
//---------------------------------------------------------
static const int kNumBodyJoints = K4ABT_JOINT_COUNT;
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
// Skeleton appearance 설정 (색상, 크기 배율)
//---------------------------------------------------------
static Eigen::Vector3d g_skelColor(1.0, 1.0, 0.0);  // 기본: 노란색
static double g_skelSizeScale = 1.0;                // 기본 배율: 1.0
static std::vector<Eigen::Vector3d> g_skelColorPresets = {
    {1.0, 1.0, 0.0}, // 노란색
    {1.0, 0.0, 0.0}, // 빨간색
    {0.0, 1.0, 0.0}, // 초록색
    {0.0, 0.0, 1.0}  // 파란색
};
static int g_currentSkelColorIndex = 0;

//---------------------------------------------------------
// 작은 Sphere / Cylinder를 이용해서 "굵은" 스켈레톤 Mesh 생성 유틸
//---------------------------------------------------------
static std::shared_ptr<open3d::geometry::TriangleMesh>
CreateSphereMesh(const Eigen::Vector3d &center, double radius = 0.02, int resolution = 10)
{
    auto sphere = open3d::geometry::TriangleMesh::CreateSphere(radius * g_skelSizeScale, resolution);
    sphere->Translate(center, /*relative=*/true);
    return sphere;
}

static std::shared_ptr<open3d::geometry::TriangleMesh>
CreateCylinderMesh(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, double radius = 0.01, int resolution = 20)
{
    Eigen::Vector3d diff = p2 - p1;
    double height = diff.norm();
    if (height < 1e-6)
        return open3d::geometry::TriangleMesh::CreateSphere(0.0);
    auto cyl = open3d::geometry::TriangleMesh::CreateCylinder(radius * g_skelSizeScale, height, resolution, 2);
    Eigen::Vector3d axisZ(0, 0, 1);
    Eigen::Vector3d axisNew = diff.normalized();
    Eigen::Vector3d cross = axisZ.cross(axisNew);
    double c_norm = cross.norm();
    if (c_norm > 1e-6) {
        cross /= c_norm;
        double angle = std::acos(axisZ.dot(axisNew));
        cyl->Rotate(Eigen::AngleAxisd(angle, cross).toRotationMatrix(), Eigen::Vector3d(0, 0, 0));
    }
    cyl->Translate(p1, /*relative=*/true);
    return cyl;
}

//---------------------------------------------------------
// Skeleton Mesh 생성 (각 관절마다 Sphere, 연결마다 Cylinder)
//---------------------------------------------------------
static std::shared_ptr<open3d::geometry::TriangleMesh>
CreateSkeletonMesh(const std::vector<Eigen::Vector3d> &joints3D)
{
    using namespace open3d::geometry;
    auto skeletonMesh = std::make_shared<TriangleMesh>();
    for (size_t j = 0; j < joints3D.size(); j++) {
        if (joints3D[j].norm() < 1e-6)
            continue;
        auto sphere = CreateSphereMesh(joints3D[j], /*radius=*/0.02);
        sphere->PaintUniformColor(g_skelColor);
        *skeletonMesh += *sphere;
    }
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

class MyVisualizer : public open3d::visualization::VisualizerWithKeyCallback {
public:
    MyVisualizer() : exit_flag_(false) {}
    bool ShouldExit() const { return exit_flag_; }
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
// 전역 변수 (FOV 모드 선택 관련)
//---------------------------------------------------------
// g_fovMode: 1 - NFOV Unbinned (640×576), 2 - WFOV Unbinned (1024×1024),
//            3 - WFOV Binned (512×512), 4 - NFOV 2x2 Binned (320×288)
static int g_fovMode = 1;
static bool g_useColor = true;    // 항상 사용
static bool g_useDepth = true;    // 항상 사용
static bool g_usePointCloud = false;
static int g_fovWidth = 640;      // 기본 해상도
static int g_fovHeight = 576;     // 기본 해상도

static k4a_device_t g_device = nullptr;
static k4a_transformation_t g_transform = nullptr;
static k4abt_tracker_t g_tracker = nullptr;

static MyVisualizer g_vis;
static std::shared_ptr<open3d::geometry::PointCloud> g_pcPtr;
static std::shared_ptr<open3d::geometry::TriangleMesh> g_skelMesh;

static std::string g_combinedWin = "Color&Depth";
static bool g_firstCloud = true;
static bool g_firstBodyDetected = false;

//---------------------------------------------------------
// 그리드 생성 함수
//---------------------------------------------------------
static std::shared_ptr<open3d::geometry::LineSet> CreateGridLineSet(int gridSize = 10, float step = 1.0f)
{
    using namespace open3d::geometry;
    auto ls = std::make_shared<LineSet>();
    std::vector<Eigen::Vector3d> pts;
    std::vector<Eigen::Vector2i> lns;
    int idx = 0;
    for (int x = -gridSize; x <= gridSize; x++) {
        Eigen::Vector3d p1(x * step, 0.0, -gridSize * step);
        Eigen::Vector3d p2(x * step, 0.0, gridSize * step);
        pts.push_back(p1);
        pts.push_back(p2);
        lns.push_back({ idx * 2, idx * 2 + 1 });
        idx++;
    }
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
    ls->PaintUniformColor({ 0.6, 0.6, 0.6 });
    return ls;
}

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

bool InitializeAll()
{
    if (K4A_FAILED(k4a_device_open(K4A_DEVICE_DEFAULT, &g_device))) {
        std::cerr << "device_open fail.\n";
        return false;
    }
    k4a_device_configuration_t cfg = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    if (g_useColor) {
        cfg.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
        cfg.color_resolution = K4A_COLOR_RESOLUTION_720P;
    } else {
        cfg.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    }
    if (g_useDepth) {
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
    if (K4A_FAILED(k4a_device_start_cameras(g_device, &cfg))) {
        std::cerr << "start_cameras fail.\n";
        k4a_device_close(g_device);
        g_device = nullptr;
        return false;
    }
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

static void DrawBody2DOnDepth(cv::Mat &depthVis,
    const k4abt_skeleton_t &skeleton,
    const k4a_calibration_t &cal)
{
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

int main()
{
    // FOV 모드 선택: 1 - NFOV Unbinned (640x576), 2 - WFOV Unbinned (1024x1024),
    //              3 - WFOV Binned (512x512), 4 - NFOV 2x2 Binned (320x288)
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
    // color와 depth는 항상 사용
    g_useColor = true;
    g_useDepth = true;
    
    std::cout << "Use PointCloud? (y/n): ";
    {
        std::string s;
        std::cin >> s;
        if (!s.empty() && (s[0] == 'y' || s[0] == 'Y'))
            g_usePointCloud = true;
    }
    
    // point cloud 창이 사용되지 않을 경우, g_vis 관련 함수 호출은 전혀 발생하지 않도록 함
RESTART_ALL:
    if (!InitializeAll()) {
        std::cerr << "init fail.\n";
        return -1;
    }
    
    bool quit = false;
    while (!quit) {
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
        k4a_image_t depthImg = (g_useDepth) ? k4a_capture_get_depth_image(cap) : nullptr;
        k4a_image_t colorImg = (g_useColor) ? k4a_capture_get_color_image(cap) : nullptr;
        
        // Depth 이미지 처리
        cv::Mat dv;
        if (depthImg) {
            int dw = k4a_image_get_width_pixels(depthImg);
            int dh = k4a_image_get_height_pixels(depthImg);
            uint16_t* dptr = (uint16_t*)k4a_image_get_buffer(depthImg);
            cv::Mat dmat(dh, dw, CV_16UC1, dptr);
            dv = cv::Mat(dh, dw, CV_8UC1);
            for (int rr = 0; rr < dh; rr++) {
                for (int cc = 0; cc < dw; cc++) {
                    uint16_t v = dmat.at<uint16_t>(rr, cc);
                    dv.at<uchar>(rr, cc) = (v > 4000) ? 255 : (v / 16);
                }
            }
        }
        
        // Color 이미지 처리
        cv::Mat showColor;
        if (g_useColor && colorImg) {
            int cw = k4a_image_get_width_pixels(colorImg);
            int ch = k4a_image_get_height_pixels(colorImg);
            int cstride = k4a_image_get_stride_bytes(colorImg);
            uint8_t* cbuf = (uint8_t*)k4a_image_get_buffer(colorImg);
            cv::Mat colorBGRA(ch, cw, CV_8UC4, (void*)cbuf, cstride);
            cv::Mat colorBGR;
            cv::cvtColor(colorBGRA, colorBGR, cv::COLOR_BGRA2BGR);
            cv::resize(colorBGR, showColor, { g_fovWidth, g_fovHeight });
        }
        
        // Body Tracking 및 Skeleton 생성
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
                            DrawBody2DOnDepth(dvCopy, skeleton, calibration);
                        }
                        
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
                        std::vector<Eigen::Vector3d> j3d(kNumBodyJoints, { 0, 0, 0 });
                        for (int j = 0; j < kNumBodyJoints; j++) {
                            k4a_float3_t pos = skeleton.joints[j].position;
                            double X = pos.v[0] / 1000.0;
                            double Y = -pos.v[1] / 1000.0;
                            double Z = -pos.v[2] / 1000.0;
                            j3d[j] = { X, Y, Z };
                        }
                        auto newMesh = CreateSkeletonMesh(j3d);
                        if (g_usePointCloud)  // point cloud 관련 mesh는 g_vis가 있을 때만 업데이트
                            *g_skelMesh += *newMesh;
                    }
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
        
        if (colorImg)
            k4a_image_release(colorImg);
        if (depthImg)
            k4a_image_release(depthImg);
        k4a_capture_release(cap);
        DrainExtraFrames();
        
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