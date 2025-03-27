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
#include <deque>
#include <map>

// Body Tracking 관련 설정
static const int kNumBodyJoints = K4ABT_JOINT_COUNT;

static const std::vector<std::pair<int, int>> kBodySkeletonConnection = {
    {K4ABT_JOINT_PELVIS, K4ABT_JOINT_SPINE_NAVEL}, {K4ABT_JOINT_SPINE_NAVEL, K4ABT_JOINT_SPINE_CHEST},
    {K4ABT_JOINT_SPINE_CHEST, K4ABT_JOINT_NECK},    {K4ABT_JOINT_NECK, K4ABT_JOINT_HEAD},
    {K4ABT_JOINT_SPINE_CHEST, K4ABT_JOINT_CLAVICLE_LEFT}, {K4ABT_JOINT_CLAVICLE_LEFT, K4ABT_JOINT_SHOULDER_LEFT},
    {K4ABT_JOINT_SHOULDER_LEFT, K4ABT_JOINT_ELBOW_LEFT},  {K4ABT_JOINT_ELBOW_LEFT, K4ABT_JOINT_WRIST_LEFT},
    {K4ABT_JOINT_WRIST_LEFT, K4ABT_JOINT_HAND_LEFT},       {K4ABT_JOINT_SPINE_CHEST, K4ABT_JOINT_CLAVICLE_RIGHT},
    {K4ABT_JOINT_CLAVICLE_RIGHT, K4ABT_JOINT_SHOULDER_RIGHT}, {K4ABT_JOINT_SHOULDER_RIGHT, K4ABT_JOINT_ELBOW_RIGHT},
    {K4ABT_JOINT_ELBOW_RIGHT, K4ABT_JOINT_WRIST_RIGHT},       {K4ABT_JOINT_WRIST_RIGHT, K4ABT_JOINT_HAND_RIGHT},
    {K4ABT_JOINT_PELVIS, K4ABT_JOINT_HIP_LEFT}, {K4ABT_JOINT_HIP_LEFT, K4ABT_JOINT_KNEE_LEFT},
    {K4ABT_JOINT_KNEE_LEFT, K4ABT_JOINT_ANKLE_LEFT}, {K4ABT_JOINT_ANKLE_LEFT, K4ABT_JOINT_FOOT_LEFT},
    {K4ABT_JOINT_PELVIS, K4ABT_JOINT_HIP_RIGHT}, {K4ABT_JOINT_HIP_RIGHT, K4ABT_JOINT_KNEE_RIGHT},
    {K4ABT_JOINT_KNEE_RIGHT, K4ABT_JOINT_ANKLE_RIGHT}, {K4ABT_JOINT_ANKLE_RIGHT, K4ABT_JOINT_FOOT_RIGHT},
};

// 스켈레톤 외형 설정
static double g_skelSizeScale = 1.0;
static std::vector<Eigen::Vector3d> g_skelColorPresets = {
    {1.0, 0.0, 0.0}, // 빨강
    {0.0, 1.0, 0.0}, // 초록
    {0.0, 0.0, 1.0}, // 파랑
    {1.0, 1.0, 0.0}, // 노랑
    {1.0, 0.0, 1.0}  // 자홍
};

// 센서 방향 및 회전 행렬 전역 변수
static k4abt_sensor_orientation_t g_sensorOrientation = K4ABT_SENSOR_ORIENTATION_FLIP180;
static Eigen::Matrix3d g_rotationMatrix;

// 함수 선언: RotateText
static cv::Mat RotateText(const std::string &text, double angle, cv::Scalar color, double scale, int thickness);

// 회전 행렬 계산 함수
static Eigen::Matrix3d GetRotationMatrix(k4abt_sensor_orientation_t orientation) {
    Eigen::Matrix3d R;
    switch (orientation) {
    case K4ABT_SENSOR_ORIENTATION_DEFAULT:
        R.setIdentity();
        break;
    case K4ABT_SENSOR_ORIENTATION_CLOCKWISE90:
        R << 0, 1, 0,
             -1, 0, 0,
              0, 0, 1;
        break;
    case K4ABT_SENSOR_ORIENTATION_COUNTERCLOCKWISE90:
        R << 0, -1, 0,
             1, 0, 0,
             0, 0, 1;
        break;
    case K4ABT_SENSOR_ORIENTATION_FLIP180:
        R << -1, 0, 0,
              0, -1, 0,
              0, 0, 1;
        break;
    default:
        R.setIdentity();
        break;
    }
    return R;
}

// OpenCV 회전 코드 계산 함수
static int GetRotationCode(k4abt_sensor_orientation_t orientation) {
    switch (orientation) {
    case K4ABT_SENSOR_ORIENTATION_DEFAULT:
        return -1; // 회전 없음
    case K4ABT_SENSOR_ORIENTATION_CLOCKWISE90:
        return cv::ROTATE_90_COUNTERCLOCKWISE;
    case K4ABT_SENSOR_ORIENTATION_COUNTERCLOCKWISE90:
        return cv::ROTATE_90_CLOCKWISE;
    case K4ABT_SENSOR_ORIENTATION_FLIP180:
        return cv::ROTATE_180;
    default:
        return -1;
    }
}

// 스켈레톤 메쉬 생성 유틸리티 함수
static std::shared_ptr<open3d::geometry::TriangleMesh>
CreateSphereMesh(const Eigen::Vector3d &center, double radius = 0.02, int resolution = 10) {
    auto sphere = open3d::geometry::TriangleMesh::CreateSphere(radius * g_skelSizeScale, resolution);
    sphere->Translate(center, true);
    return sphere;
}

static std::shared_ptr<open3d::geometry::TriangleMesh>
CreateCylinderMesh(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, double radius = 0.01, int resolution = 20) {
    Eigen::Vector3d diff = p2 - p1;
    double height = diff.norm();
    if (height < 1e-6) return open3d::geometry::TriangleMesh::CreateSphere(0.0);
    auto cyl = open3d::geometry::TriangleMesh::CreateCylinder(radius * g_skelSizeScale, height, resolution, 2);
    Eigen::Vector3d axisZ(0, 0, 1), axisNew = diff.normalized();
    Eigen::Vector3d cross = axisZ.cross(axisNew);
    double c_norm = cross.norm();
    if (c_norm > 1e-6) {
        cross /= c_norm;
        double angle = std::acos(axisZ.dot(axisNew));
        cyl->Rotate(Eigen::AngleAxisd(angle, cross).toRotationMatrix(), Eigen::Vector3d(0, 0, 0));
    }
    cyl->Translate(p1, true);
    return cyl;
}

// 색상을 인자로 받아 각 사람별로 다른 색상 적용
static std::shared_ptr<open3d::geometry::TriangleMesh>
CreateSkeletonMesh(const std::vector<Eigen::Vector3d> &joints3D, const Eigen::Vector3d &color) {
    auto skeletonMesh = std::make_shared<open3d::geometry::TriangleMesh>();
    for (size_t j = 0; j < joints3D.size(); j++) {
        if (joints3D[j].norm() < 1e-6) continue;
        auto sphere = CreateSphereMesh(joints3D[j], 0.02);
        sphere->PaintUniformColor(color);
        *skeletonMesh += *sphere;
    }
    for (auto &conn : kBodySkeletonConnection) {
        int i1 = conn.first, i2 = conn.second;
        if (i1 < (int)joints3D.size() && i2 < (int)joints3D.size()) {
            Eigen::Vector3d p1 = joints3D[i1], p2 = joints3D[i2];
            if (p1.norm() < 1e-6 || p2.norm() < 1e-6) continue;
            auto cyl = CreateCylinderMesh(p1, p2, 0.01);
            cyl->PaintUniformColor(color);
            *skeletonMesh += *cyl;
        }
    }
    return skeletonMesh;
}

// Open3D Visualizer 확장 클래스
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

// 전역 변수
static int g_fovMode = 1;
static bool g_useColor = true, g_useDepth = true, g_usePointCloud = false;
static int g_fovWidth = 640, g_fovHeight = 576;
static k4a_device_t g_device = nullptr;
static k4a_transformation_t g_transform = nullptr;
static k4abt_tracker_t g_tracker = nullptr;
static MyVisualizer g_vis;
static std::shared_ptr<open3d::geometry::PointCloud> g_pcPtr;
static std::shared_ptr<open3d::geometry::TriangleMesh> g_skelMesh;
static std::string g_combinedWin = "Color&Depth";
static bool g_firstCloud = true, g_firstBodyDetected = false;
static bool g_showHandInfo = false; // 손목 정보 표시 여부

// ID 기반 스켈레톤 버퍼와 색상 매핑
static const size_t kFrameBufferSize = 3; // 평균을 낼 프레임 수
static std::map<uint32_t, std::deque<k4abt_skeleton_t>> g_skeletonBuffers; // ID -> deque
static std::map<uint32_t, Eigen::Vector3d> g_idToColor; // ID -> 색상

static std::shared_ptr<open3d::geometry::LineSet>
CreateGridLineSet(int gridSize = 10, float step = 1.0f) {
    auto ls = std::make_shared<open3d::geometry::LineSet>();
    std::vector<Eigen::Vector3d> pts;
    std::vector<Eigen::Vector2i> lns;
    int idx = 0;
    for (int x = -gridSize; x <= gridSize; x++) {
        pts.push_back({x * step, 0.0, -gridSize * step});
        pts.push_back({x * step, 0.0, gridSize * step});
        lns.push_back({idx * 2, idx * 2 + 1});
        idx++;
    }
    for (int z = -gridSize; z <= gridSize; z++) {
        pts.push_back({-gridSize * step, 0.0, z * step});
        pts.push_back({gridSize * step, 0.0, z * step});
        lns.push_back({idx * 2, idx * 2 + 1});
        idx++;
    }
    ls->points_ = pts;
    ls->lines_ = lns;
    ls->PaintUniformColor({0.6, 0.6, 0.6});
    return ls;
}

static bool RecreateVisualizerWindow(MyVisualizer &vis) {
    vis.Close();
    vis.DestroyVisualizerWindow();
    int pcWidth, pcHeight;
    if (g_fovMode == 1) { pcWidth = 640; pcHeight = 576; }
    else if (g_fovMode == 2) { pcWidth = 1024; pcHeight = 1024; }
    else if (g_fovMode == 3) { pcWidth = 512; pcHeight = 512; }
    else { pcWidth = 320; pcHeight = 288; }
    if (!vis.CreateVisualizerWindow("3D Point Cloud", pcWidth, pcHeight)) return false;
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

bool InitializeAll() {
    if (K4A_FAILED(k4a_device_open(K4A_DEVICE_DEFAULT, &g_device))) {
        std::cerr << "device_open fail.\n";
        return false;
    }

    k4a_device_configuration_t cfg = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    cfg.camera_fps = K4A_FRAMES_PER_SECOND_15;
    if (g_useColor) {
        cfg.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
        cfg.color_resolution = K4A_COLOR_RESOLUTION_720P;
    } else {
        cfg.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    }

    if (g_useDepth) {
        if (g_fovMode == 1)      cfg.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
        else if (g_fovMode == 2) cfg.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
        else if (g_fovMode == 3) cfg.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
        else                     cfg.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
    } else {
        cfg.depth_mode = K4A_DEPTH_MODE_OFF;
    }

    cfg.synchronized_images_only = true;

    if (K4A_FAILED(k4a_device_start_cameras(g_device, &cfg))) {
        std::cerr << "start_cameras fail.\n";
        k4a_device_close(g_device);
        g_device = nullptr;
        return false;
    }

    if (g_useColor || g_useDepth) {
        k4a_calibration_t cal;
        k4a_depth_mode_t selected_mode =
            (g_fovMode == 1
                 ? K4A_DEPTH_MODE_NFOV_UNBINNED
                 : g_fovMode == 2
                       ? K4A_DEPTH_MODE_WFOV_UNBINNED
                       : g_fovMode == 3 ? K4A_DEPTH_MODE_WFOV_2X2BINNED
                                        : K4A_DEPTH_MODE_NFOV_2X2BINNED);

        if (K4A_FAILED(k4a_device_get_calibration(g_device, selected_mode, K4A_COLOR_RESOLUTION_720P, &cal))) {
            std::cerr << "get_calibration fail.\n";
            return false;
        }

        g_transform = k4a_transformation_create(&cal);
        if (!g_transform)
            return false;

        k4abt_tracker_configuration_t tcfg = K4ABT_TRACKER_CONFIG_DEFAULT;
        tcfg.sensor_orientation = g_sensorOrientation;
        tcfg.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU;

        if (K4A_FAILED(k4abt_tracker_create(&cal, tcfg, &g_tracker))) {
            std::cerr << "k4abt_tracker_create fail.\n";
            return false;
        }

        g_rotationMatrix = GetRotationMatrix(g_sensorOrientation);
    }

    if (g_useColor && g_useDepth) {
        cv::namedWindow(g_combinedWin, cv::WINDOW_NORMAL);
        cv::resizeWindow(g_combinedWin, g_fovWidth * 2, g_fovHeight);
    }

    if (g_usePointCloud) {
        int pcWidth, pcHeight;
        if (g_fovMode == 1)      { pcWidth = 640;  pcHeight = 576; }
        else if (g_fovMode == 2) { pcWidth = 1024; pcHeight = 1024; }
        else if (g_fovMode == 3) { pcWidth = 512;  pcHeight = 512; }
        else                     { pcWidth = 320;  pcHeight = 288; }

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
    g_skeletonBuffers.clear();
    g_idToColor.clear();
    g_showHandInfo = false; // 초기값: 손목 정보 표시 안 함
    return true;
}

void FinalizeAll() {
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

static k4a_image_t
CreateAlignedColorImage(k4a_transformation_t transform, k4a_image_t depthImg, k4a_image_t colorImg) {
    if (!transform || !depthImg || !colorImg) return nullptr;
    int dw = k4a_image_get_width_pixels(depthImg), dh = k4a_image_get_height_pixels(depthImg);
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

// RotateText 함수 정의
static cv::Mat
RotateText(const std::string &text, double angle, cv::Scalar color, double scale, int thickness) {
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    cv::Mat textImg(textSize.height + baseline, textSize.width, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::putText(textImg, text, cv::Point(0, textSize.height),
                cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness);
    cv::Mat rotatedText;
    cv::Point2f center(textImg.cols / 2.0F, textImg.rows / 2.0F);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(textImg, rotatedText, rotMat, textImg.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return rotatedText;
}

static void
DrawBody2DOnDepth(cv::Mat &depthVis, const k4abt_skeleton_t &skeleton, const k4a_calibration_t &cal, bool showHandInfo) {
    // 스켈레톤 그리기
    for (auto &conn : kBodySkeletonConnection) {
        k4a_float2_t p1, p2;
        int v1 = 0, v2 = 0;
        k4a_calibration_3d_to_2d(&cal, &skeleton.joints[conn.first].position,
                                 K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &p1, &v1);
        k4a_calibration_3d_to_2d(&cal, &skeleton.joints[conn.second].position,
                                 K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &p2, &v2);
        if (v1 && v2) {
            cv::line(depthVis,
                     {int(p1.xy.x), int(p1.xy.y)},
                     {int(p2.xy.x), int(p2.xy.y)},
                     cv::Scalar(0, 255, 0), 4);
        }
    }
    for (int j = 0; j < kNumBodyJoints; j++) {
        k4a_float2_t p2d;
        int valid = 0;
        k4a_calibration_3d_to_2d(&cal, &skeleton.joints[j].position,
                                 K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &p2d, &valid);
        if (valid) {
            cv::circle(depthVis, {int(p2d.xy.x), int(p2d.xy.y)}, 5, cv::Scalar(0, 0, 255), -1);
        }
    }

    // 손목 정보 표시 (g_showHandInfo가 true일 때만)
    if (showHandInfo) {
        // 왼손 손목 정보
        int leftWristIndex = K4ABT_JOINT_WRIST_LEFT;
        k4a_float2_t leftWrist2d;
        int validLeft = 0;
        k4a_calibration_3d_to_2d(&cal, &skeleton.joints[leftWristIndex].position,
                                 K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &leftWrist2d, &validLeft);
        if (validLeft) {
            k4a_float3_t leftPos = skeleton.joints[leftWristIndex].position;
            Eigen::Vector3d p(leftPos.v[0] / 1000.0, -leftPos.v[1] / 1000.0, -leftPos.v[2] / 1000.0);
            Eigen::Vector3d p_default = g_rotationMatrix.transpose() * p;
            double leftDistance = p_default.norm();
            std::string leftText = "L: " + std::to_string(leftDistance).substr(0, 4) + " m";
            std::string leftCoord = "(" + std::to_string(p_default[0]).substr(0, 4) + ", " +
                                    std::to_string(p_default[1]).substr(0, 4) + ", " +
                                    std::to_string(p_default[2]).substr(0, 4) + ")";
            double angle = (g_sensorOrientation == K4ABT_SENSOR_ORIENTATION_FLIP180) ? 180.0 : 0.0;
            cv::Mat textImg1 = RotateText(leftText, angle, cv::Scalar(0, 0, 255), 0.8, 2);
            cv::Mat textImg2 = RotateText(leftCoord, angle, cv::Scalar(0, 0, 255), 0.8, 2);
            cv::Rect roi1((int)leftWrist2d.xy.x + 5, (int)leftWrist2d.xy.y - 5 - textImg1.rows, textImg1.cols, textImg1.rows);
            cv::Rect roi2((int)leftWrist2d.xy.x + 5, (int)leftWrist2d.xy.y + 20 - textImg2.rows, textImg2.cols, textImg2.rows);
            if (roi1.x >= 0 && roi1.y >= 0 && roi1.x + roi1.width <= depthVis.cols && roi1.y + roi1.height <= depthVis.rows)
                textImg1.copyTo(depthVis(roi1));
            if (roi2.x >= 0 && roi2.y >= 0 && roi2.x + roi2.width <= depthVis.cols && roi2.y + roi2.height <= depthVis.rows)
                textImg2.copyTo(depthVis(roi2));
        }

        // 오른손 손목 정보
        int rightWristIndex = K4ABT_JOINT_WRIST_RIGHT;
        k4a_float2_t rightWrist2d;
        int validRight = 0;
        k4a_calibration_3d_to_2d(&cal, &skeleton.joints[rightWristIndex].position,
                                 K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &rightWrist2d, &validRight);
        if (validRight) {
            k4a_float3_t rightPos = skeleton.joints[rightWristIndex].position;
            Eigen::Vector3d p(rightPos.v[0] / 1000.0, -rightPos.v[1] / 1000.0, -rightPos.v[2] / 1000.0);
            Eigen::Vector3d p_default = g_rotationMatrix.transpose() * p;
            double rightDistance = p_default.norm();
            std::string rightText = "R: " + std::to_string(rightDistance).substr(0, 4) + " m";
            std::string rightCoord = "(" + std::to_string(p_default[0]).substr(0, 4) + ", " +
                                     std::to_string(p_default[1]).substr(0, 4) + ", " +
                                     std::to_string(p_default[2]).substr(0, 4) + ")";
            double angle = (g_sensorOrientation == K4ABT_SENSOR_ORIENTATION_FLIP180) ? 180.0 : 0.0;
            cv::Mat textImg1 = RotateText(rightText, angle, cv::Scalar(0, 0, 255), 0.8, 2);
            cv::Mat textImg2 = RotateText(rightCoord, angle, cv::Scalar(0, 0, 255), 0.8, 2);
            cv::Rect roi1((int)rightWrist2d.xy.x + 5, (int)rightWrist2d.xy.y - 5 - textImg1.rows, textImg1.cols, textImg1.rows);
            cv::Rect roi2((int)rightWrist2d.xy.x + 5, (int)rightWrist2d.xy.y + 20 - textImg2.rows, textImg2.cols, textImg2.rows);
            if (roi1.x >= 0 && roi1.y >= 0 && roi1.x + roi1.width <= depthVis.cols && roi1.y + roi1.height <= depthVis.rows)
                textImg1.copyTo(depthVis(roi1));
            if (roi2.x >= 0 && roi2.y >= 0 && roi2.x + roi2.width <= depthVis.cols && roi2.y + roi2.height <= depthVis.rows)
                textImg2.copyTo(depthVis(roi2));
        }
    }
}

static k4abt_skeleton_t
ComputeAverageSkeleton(const std::deque<k4abt_skeleton_t> &buffer) {
    k4abt_skeleton_t avgSkeleton;
    if (buffer.empty()) return avgSkeleton;

    size_t bufferSize = buffer.size();
    for (int j = 0; j < kNumBodyJoints; j++) {
        float sumX = 0.0f, sumY = 0.0f, sumZ = 0.0f;
        for (const auto &skel : buffer) {
            sumX += skel.joints[j].position.xyz.x;
            sumY += skel.joints[j].position.xyz.y;
            sumZ += skel.joints[j].position.xyz.z;
        }
        avgSkeleton.joints[j].position.xyz.x = sumX / bufferSize;
        avgSkeleton.joints[j].position.xyz.y = sumY / bufferSize;
        avgSkeleton.joints[j].position.xyz.z = sumZ / bufferSize;
        avgSkeleton.joints[j].confidence_level = buffer.back().joints[j].confidence_level;
    }
    return avgSkeleton;
}

int main() {
    std::cout << "Select FOV mode:\n"
              << " 1: NFOV Unbinned (640x576)\n"
              << " 2: WFOV Unbinned (1024x1024)\n"
              << " 3: WFOV Binned (512x512)\n"
              << " 4: NFOV 2x2 Binned (320x288)\n"
              << "Enter number: ";
    std::cin >> g_fovMode;
    if (g_fovMode == 1)      { g_fovWidth = 640;  g_fovHeight = 576; }
    else if (g_fovMode == 2) { g_fovWidth = 1024; g_fovHeight = 1024; }
    else if (g_fovMode == 3) { g_fovWidth = 512;  g_fovHeight = 512; }
    else if (g_fovMode == 4) { g_fovWidth = 320;  g_fovHeight = 288; }
    else {
        g_fovMode = 1;
        g_fovWidth = 640;
        g_fovHeight = 576;
    }

    g_useColor = true;
    g_useDepth = true;

    std::cout << "Use PointCloud? (y/n): ";
    std::string s;
    std::cin >> s;
    if (!s.empty() && (s[0] == 'y' || s[0] == 'Y'))
        g_usePointCloud = true;

RESTART_ALL:
    if (!InitializeAll()) {
        std::cerr << "init fail.\n";
        return -1;
    }

    bool quit = false;
    static std::chrono::steady_clock::time_point prevTime = std::chrono::steady_clock::now();

    while (!quit) {
        if (g_usePointCloud) {
            g_vis.PollEvents();
            if (g_vis.ShouldExit())
                break;
        }

        k4a_capture_t cap = nullptr;
        auto r = k4a_device_get_capture(g_device, &cap, 0); // 타임아웃 0으로 즉시 반환
        if (r == K4A_WAIT_RESULT_TIMEOUT || r == K4A_WAIT_RESULT_FAILED || !cap) {
            continue;
        }

        k4a_image_t depthImg = g_useDepth ? k4a_capture_get_depth_image(cap) : nullptr;
        k4a_image_t colorImg = g_useColor ? k4a_capture_get_color_image(cap) : nullptr;

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

        cv::Mat showColor;
        if (g_useColor && colorImg) {
            int cw = k4a_image_get_width_pixels(colorImg);
            int ch = k4a_image_get_height_pixels(colorImg);
            int cstride = k4a_image_get_stride_bytes(colorImg);
            uint8_t* cbuf = (uint8_t*)k4a_image_get_buffer(colorImg);
            cv::Mat colorBGRA(ch, cw, CV_8UC4, (void*)cbuf, cstride);
            cv::Mat colorBGR;
            cv::cvtColor(colorBGRA, colorBGR, cv::COLOR_BGRA2BGR);
            cv::resize(colorBGR, showColor, {g_fovWidth, g_fovHeight});
        }

        // Body Tracking
        if (g_tracker && cap) {
            if (K4A_FAILED(k4abt_tracker_enqueue_capture(g_tracker, cap, 0)))
                std::cerr << "tracker_enqueue fail.\n";
            else {
                k4abt_frame_t bodyFrame = nullptr;
                auto popRes = k4abt_tracker_pop_result(g_tracker, &bodyFrame, 10);
                if (popRes == K4A_WAIT_RESULT_SUCCEEDED && bodyFrame) {
                    size_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
                    cv::Mat dvCopy;
                    cv::cvtColor(dv, dvCopy, cv::COLOR_GRAY2BGR);

                    if (g_usePointCloud && g_skelMesh)
                        g_skelMesh->Clear();

                    for (size_t i = 0; i < numBodies; i++) {
                        uint32_t id = k4abt_frame_get_body_id(bodyFrame, i);
                        k4abt_skeleton_t skeleton;
                        k4abt_frame_get_body_skeleton(bodyFrame, i, &skeleton);

                        // ID -> Skeleton Buffer
                        if (g_skeletonBuffers.find(id) == g_skeletonBuffers.end()) {
                            g_skeletonBuffers[id] = std::deque<k4abt_skeleton_t>();
                            g_idToColor[id] = g_skelColorPresets[g_idToColor.size() % g_skelColorPresets.size()];
                        }
                        if (g_skeletonBuffers[id].size() >= kFrameBufferSize) {
                            g_skeletonBuffers[id].pop_front();
                        }
                        g_skeletonBuffers[id].push_back(skeleton);

                        k4abt_skeleton_t avgSkeleton = ComputeAverageSkeleton(g_skeletonBuffers[id]);

                        // 2D 그리기
                        k4a_depth_mode_t selected_mode =
                            (g_fovMode == 1
                                 ? K4A_DEPTH_MODE_NFOV_UNBINNED
                                 : g_fovMode == 2
                                       ? K4A_DEPTH_MODE_WFOV_UNBINNED
                                       : g_fovMode == 3 ? K4A_DEPTH_MODE_WFOV_2X2BINNED
                                                        : K4A_DEPTH_MODE_NFOV_2X2BINNED);

                        k4a_calibration_t calibration;
                        if (K4A_RESULT_SUCCEEDED ==
                            k4a_device_get_calibration(g_device, selected_mode, K4A_COLOR_RESOLUTION_720P, &calibration)) {
                            DrawBody2DOnDepth(dvCopy, avgSkeleton, calibration, g_showHandInfo);
                        }

                        // 3D 스켈레톤 메쉬
                        std::vector<Eigen::Vector3d> j3d(kNumBodyJoints, {0, 0, 0});
                        for (int j = 0; j < kNumBodyJoints; j++) {
                            k4a_float3_t pos = avgSkeleton.joints[j].position;
                            double X = pos.v[0] / 1000.0;
                            double Y = -pos.v[1] / 1000.0;
                            double Z = -pos.v[2] / 1000.0;
                            j3d[j] = {X, Y, Z};
                        }
                        auto skeletonMesh = CreateSkeletonMesh(j3d, g_idToColor[id]);
                        if (g_usePointCloud && g_skelMesh) {
                            *g_skelMesh += *skeletonMesh;
                        }
                    }

                    cv::Mat showDepth;
                    cv::resize(dvCopy, showDepth, {g_fovWidth, g_fovHeight});
                    int rotationCode = GetRotationCode(g_sensorOrientation);
                    if (rotationCode != -1) {
                        cv::Mat rotatedColor, rotatedDepth;
                        cv::rotate(showColor, rotatedColor, rotationCode);
                        cv::rotate(showDepth, rotatedDepth, rotationCode);
                        showColor = rotatedColor;
                        showDepth = rotatedDepth;
                    }
                    cv::Mat combined;
                    if (!showColor.empty() && !showDepth.empty())
                        cv::hconcat(showColor, showDepth, combined);

                    if (!combined.empty()) {
                        cv::resizeWindow(g_combinedWin, combined.cols, combined.rows);
                        cv::imshow(g_combinedWin, combined);
                    }

                    // Open3D 상에 스켈레톤 메쉬 갱신
                    if (g_usePointCloud && g_skelMesh) {
                        g_vis.UpdateGeometry(g_skelMesh);
                        g_vis.UpdateRender();
                        if (!g_firstBodyDetected && !g_skelMesh->vertices_.empty()) {
                            g_firstBodyDetected = true;
                            g_vis.ResetViewPoint(true);
                        }
                    }

                    k4abt_frame_release(bodyFrame);
                }
            }
        }

        // 여기서부터 추가: depthImg를 이용해 point cloud 생성
        if (depthImg && g_usePointCloud && g_transform) {
            int dw = k4a_image_get_width_pixels(depthImg);
            int dh = k4a_image_get_height_pixels(depthImg);

            // Point Cloud용 이미지(pcImg) 생성
            k4a_image_t pcImg = nullptr;
            if (K4A_SUCCEEDED(k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
                                               dw, dh, dw * 3 * (int)sizeof(int16_t),
                                               &pcImg))) {
                // depth -> point cloud 변환
                if (K4A_SUCCEEDED(k4a_transformation_depth_image_to_point_cloud(
                                      g_transform, depthImg,
                                      K4A_CALIBRATION_TYPE_DEPTH, pcImg))) {
                    // 생성된 포인트 클라우드 업데이트
                    g_pcPtr->Clear();
                    g_pcPtr->points_.reserve(dw * dh);
                    g_pcPtr->colors_.reserve(dw * dh);

                    int16_t* buffer = (int16_t*)k4a_image_get_buffer(pcImg);
                    for (int i = 0; i < dw * dh; i++) {
                        int16_t xVal = buffer[3 * i + 0];
                        int16_t yVal = buffer[3 * i + 1];
                        int16_t zVal = buffer[3 * i + 2];
                        if (zVal == 0) // 유효 깊이 데이터가 없는 경우 무시
                            continue;

                        // Azure Kinect 기준 좌표계 -> 우리가 쓰는 변환
                        double X = (double)xVal / 1000.0;
                        double Y = -(double)yVal / 1000.0;
                        double Z = -(double)zVal / 1000.0;

                        g_pcPtr->points_.push_back({X, Y, Z});

                        // 예시: 거리 기반으로 간단하게 색상 매핑
                        double dist = std::sqrt(X * X + Y * Y + Z * Z);
                        double maxDist = 4.0; // 4m를 최대 표시 범위로 가정
                        if (dist > maxDist) dist = maxDist;
                        double t = dist / maxDist;
                        // 파랑(가까움) -> 빨강(멀리)
                        g_pcPtr->colors_.push_back({t, 0.0, 1.0 - t});
                    }

                    if (!g_pcPtr->points_.empty() && g_firstCloud) {
                        g_firstCloud = false;
                        g_vis.ResetViewPoint(true);
                    }
                    g_vis.UpdateGeometry(g_pcPtr);
                    g_vis.UpdateRender();
                }
                k4a_image_release(pcImg);
            }
        }

        if (depthImg)  k4a_image_release(depthImg);
        if (colorImg)  k4a_image_release(colorImg);
        k4a_capture_release(cap);

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - prevTime).count();
        int targetInterval = 1000 / 30;
        if (elapsed < targetInterval) {
            std::this_thread::sleep_for(std::chrono::milliseconds(targetInterval - elapsed));
        }
        prevTime = std::chrono::steady_clock::now();

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            quit = true;
            break;
        } else if (key == 'r' || key == 'R') {
            if (g_usePointCloud)
                RecreateVisualizerWindow(g_vis);
        } else if (key == 'd' || key == 'D') {
            FinalizeAll();
            goto RESTART_ALL;
        } else if (key == 'h' || key == 'H') {
            g_showHandInfo = !g_showHandInfo; // 손목 정보 표시 토글
        }
    }

    FinalizeAll();
    return 0;
}