#define NOMINMAX 
#include <k4a/k4a.h>
#include <open3d/visualization/visualizer/VisualizerWithKeyCallback.h>
#include <open3d/visualization/visualizer/ViewControl.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/TriangleMesh.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <cmath>
#include <deque>

//---------------------------------------------------------
// YOLO Pose 관련
//---------------------------------------------------------
static cv::dnn::Net g_poseNet;
static const int YOLO_INPUT_W = 640; // YOLO 모델 입력
static const int YOLO_INPUT_H = 640;
static const float kConfidenceThreshold = 0.5f; // 필요에 맞게 조정

// COCO 17 관절 (0~16)
static const int kNumKeypoints = 17;
static const std::vector<std::pair<int,int>> kSkeletonConnection = {
    {0,1}, {0,2}, {1,3}, {2,4}, {0,5}, {0,6},
    {5,7}, {7,9}, {6,8}, {8,10}, {5,11}, {6,12},
    {11,13}, {13,15}, {12,14}, {14,16}, {11,12}
};

//---------------------------------------------------------
class MyVisualizer : public open3d::visualization::VisualizerWithKeyCallback {
public:
    MyVisualizer() : exit_flag_(false) {}
    bool ShouldExit() const { return exit_flag_; }
    void RegisterExitKey() {
        RegisterKeyCallback(GLFW_KEY_ESCAPE,
            [this](open3d::visualization::Visualizer*) -> bool {
                exit_flag_ = true;
                return true;
            });
    }
private:
    bool exit_flag_;
};

static std::shared_ptr<open3d::geometry::LineSet> CreateGridLineSet(
    int gridSize = 10, float step = 1.0f)
{
    using namespace open3d::geometry;
    auto lineSet = std::make_shared<LineSet>();
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector2i> lines;
    int idx = 0;
    for (int x = -gridSize; x <= gridSize; ++x) {
        Eigen::Vector3d p1(x * step, 0.0, -gridSize * step);
        Eigen::Vector3d p2(x * step, 0.0, +gridSize * step);
        points.push_back(p1);
        points.push_back(p2);
        lines.push_back(Eigen::Vector2i(idx*2, idx*2+1));
        idx++;
    }
    for (int z = -gridSize; z <= gridSize; ++z){
        Eigen::Vector3d p1(-gridSize * step, 0.0, z * step);
        Eigen::Vector3d p2(+gridSize * step, 0.0, z * step);
        points.push_back(p1);
        points.push_back(p2);
        lines.push_back(Eigen::Vector2i(idx*2, idx*2+1));
        idx++;
    }
    lineSet->points_ = points;
    lineSet->lines_ = lines;
    lineSet->PaintUniformColor({0.6, 0.6, 0.6});
    return lineSet;
}

//---------------------------------------------------------
// 전역
//---------------------------------------------------------
static bool g_useColor = false;
static bool g_useDepth = false;
static bool g_usePointCloud = false;
static k4a_device_t g_device = nullptr;
static k4a_transformation_t g_transform = nullptr;

static MyVisualizer g_vis;
static std::shared_ptr<open3d::geometry::PointCloud> g_pcPtr;
static std::shared_ptr<open3d::geometry::LineSet> g_skelLineSet;

static std::string g_colorWin = "Color";
static std::string g_depthWin = "Depth";
static bool g_firstCloud = true;

// 여러 프레임 키포인트 평균화
static std::deque<std::vector<cv::Point2f>> g_kpHistory;
static const int kHistorySize = 5;

//---------------------------------------------------------
// 포즈 추론 구조
//---------------------------------------------------------
struct PoseKeypoint {
    cv::Point2f pt;
    float conf;
};
struct PersonPose {
    std::vector<PoseKeypoint> keypoints;
    float score;
};

//---------------------------------------------------------
// YOLO Pose 추론 (Aligned Color 이미지 기준)
//---------------------------------------------------------
static std::vector<PersonPose> InferPose(const cv::Mat &imgBGR)
{
    // 1) 모델 입력 준비
    // (주의) 여기서는 단순 resize(640x640) 사용
    // letterbox 미적용이므로 비율 왜곡 가능
    cv::Mat resized;
    cv::resize(imgBGR, resized, cv::Size(YOLO_INPUT_W, YOLO_INPUT_H));

    cv::Mat blob = cv::dnn::blobFromImage(
        resized, 1/255.f, cv::Size(YOLO_INPUT_W, YOLO_INPUT_H),
        cv::Scalar(0,0,0), true, false);

    g_poseNet.setInput(blob);
    cv::Mat output = g_poseNet.forward(); // (1, 56, 8400) 가정

    // 2) output 파싱 → 스켈레톤
    std::vector<PersonPose> results;
    int channels = output.size[1]; // 56
    int numPred = output.size[2];  // 8400
    float* data = (float*)output.data;

    // best conf 선택
    float bestConf = -1.f;
    int bestIdx = -1;
    for(int i=0; i<numPred; i++){
        float c = data[i*channels + 4]; // confidence
        if(c>bestConf){
            bestConf = c;
            bestIdx = i;
        }
    }
    if(bestIdx < 0 || bestConf < kConfidenceThreshold){
        return results;
    }

    PersonPose p;
    p.score = bestConf;
    p.keypoints.resize(kNumKeypoints);

    for(int i=0; i<kNumKeypoints; i++){
        int baseCh = 5 + i*3;
        float x = data[bestIdx*channels + baseCh];
        float y = data[bestIdx*channels + baseCh + 1];
        float c = data[bestIdx*channels + baseCh + 2];

        // 모델 입력(640x640) → 현재 Aligned BGR 크기
        float sx = (float)imgBGR.cols / (float)YOLO_INPUT_W;
        float sy = (float)imgBGR.rows / (float)YOLO_INPUT_H;
        x *= sx;
        y *= sy;

        // 범위 검사
        if(x<0 || x>=imgBGR.cols || y<0 || y>=imgBGR.rows){
            c = 0.f;
        }
        p.keypoints[i].pt = cv::Point2f(x,y);
        p.keypoints[i].conf = c;
    }
    results.push_back(p);
    return results;
}

//---------------------------------------------------------
// CreateAlignedColorImage: Depth 해상도에 맞춤
//---------------------------------------------------------
static k4a_image_t CreateAlignedColorImage(
    k4a_transformation_t transform,
    k4a_image_t depthImg,
    k4a_image_t colorImg)
{
    if(!transform || !depthImg || !colorImg) return nullptr;

    int dw = k4a_image_get_width_pixels(depthImg);
    int dh = k4a_image_get_height_pixels(depthImg);

    k4a_image_t alignedColor = nullptr;
    if(K4A_SUCCEEDED(k4a_image_create(
        K4A_IMAGE_FORMAT_COLOR_BGRA32, dw, dh,
        dw*4, &alignedColor)))
    {
        if(K4A_FAILED(k4a_transformation_color_image_to_depth_camera(
            transform, depthImg, colorImg, alignedColor)))
        {
            k4a_image_release(alignedColor);
            return nullptr;
        }
        return alignedColor;
    }
    return nullptr;
}

//---------------------------------------------------------
// 1픽셀 -> 3D 변환
//---------------------------------------------------------
static bool PixelTo3D(k4a_transformation_t transform,
    int px, int py, uint16_t depthVal,
    double &outX, double &outY, double &outZ)
{
    if(depthVal==0) return false;
    k4a_image_t d1 = nullptr;
    if(K4A_FAILED(k4a_image_create(
        K4A_IMAGE_FORMAT_DEPTH16,1,1,sizeof(uint16_t), &d1))){
        return false;
    }
    uint16_t *ptr = (uint16_t*)k4a_image_get_buffer(d1);
    ptr[0] = depthVal;

    k4a_image_t pcImg = nullptr;
    if(K4A_FAILED(k4a_image_create(
        K4A_IMAGE_FORMAT_CUSTOM, 1,1,3*sizeof(int16_t), &pcImg))){
        k4a_image_release(d1);
        return false;
    }

    if(K4A_FAILED(k4a_transformation_depth_image_to_point_cloud(
        transform, d1, K4A_CALIBRATION_TYPE_DEPTH, pcImg))){
        k4a_image_release(pcImg);
        k4a_image_release(d1);
        return false;
    }

    int16_t *b = (int16_t*)k4a_image_get_buffer(pcImg);
    double X = (double)(b[0])/1000.0;
    double Y = (double)(b[1])/1000.0;
    double Z = (double)(b[2])/1000.0;

    // Kinect -> Open3D (Y,Z 뒤집음)
    outX = X;
    outY = -Y;
    outZ = -Z;

    k4a_image_release(pcImg);
    k4a_image_release(d1);
    return true;
}

//---------------------------------------------------------
// 스켈레톤 라인셋 초기화
//---------------------------------------------------------
static void InitSkeletonLineSet(open3d::visualization::Visualizer &vis)
{
    g_skelLineSet = std::make_shared<open3d::geometry::LineSet>();
    g_skelLineSet->Clear();
    g_skelLineSet->PaintUniformColor({1.0,0.0,0.0});
    vis.AddGeometry(g_skelLineSet);
}

static void UpdateSkeleton3D(
    const std::vector<Eigen::Vector3d> &joints3D)
{
    g_skelLineSet->Clear();
    g_skelLineSet->points_.resize(joints3D.size());
    for(size_t i=0; i<joints3D.size(); i++){
        g_skelLineSet->points_[i] = joints3D[i];
    }
    for(auto &conn : kSkeletonConnection){
        int i1 = conn.first;
        int i2 = conn.second;
        if(i1<(int)joints3D.size() && i2<(int)joints3D.size()){
            g_skelLineSet->lines_.push_back(
                Eigen::Vector2i(i1,i2));
        }
    }
}

//---------------------------------------------------------
static bool RecreateVisualizerWindow(MyVisualizer &vis,
    std::shared_ptr<open3d::geometry::PointCloud> &pcPtr)
{
    using namespace open3d::visualization;
    vis.Close();
    vis.DestroyVisualizerWindow();

    if(!vis.CreateVisualizerWindow("3D Point Cloud",640,480)){
        return false;
    }
    vis.RegisterExitKey();

    pcPtr = std::make_shared<open3d::geometry::PointCloud>();
    vis.AddGeometry(pcPtr);

    auto axes = open3d::geometry::TriangleMesh::CreateCoordinateFrame(1.0,{0,0,0});
    vis.AddGeometry(axes);

    auto grid = CreateGridLineSet(10,0.2f);
    vis.AddGeometry(grid);

    vis.GetRenderOption().background_color_={0,0,0};
    vis.GetRenderOption().point_size_=1.0;
    // 스켈레톤 라인셋 다시 추가
    InitSkeletonLineSet(vis);

    return true;
}

static void DrainExtraFrames()
{
    while(true){
        k4a_capture_t ex=nullptr;
        auto r = k4a_device_get_capture(g_device, &ex,0);
        if(r==K4A_WAIT_RESULT_SUCCEEDED && ex){
            k4a_capture_release(ex);
        } else {
            break;
        }
    }
}

//---------------------------------------------------------
// Init / Fini
//---------------------------------------------------------
bool InitializeAll(){
    if(K4A_FAILED(k4a_device_open(K4A_DEVICE_DEFAULT, &g_device))){
        std::cerr<<"[Error] device_open.\n";
        return false;
    }
    k4a_device_configuration_t cfg = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    if(g_useColor){
        cfg.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
        cfg.color_resolution = K4A_COLOR_RESOLUTION_720P; // 예시
    }else{
        cfg.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    }
    if(g_useDepth){
        cfg.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED; // 640x576 등
    }else{
        cfg.depth_mode = K4A_DEPTH_MODE_OFF;
    }
    cfg.camera_fps = K4A_FRAMES_PER_SECOND_30;
    cfg.synchronized_images_only=true;

    if(K4A_FAILED(k4a_device_start_cameras(g_device,&cfg))){
        std::cerr<<"[Error] start_cameras.\n";
        k4a_device_close(g_device);
        g_device=nullptr;
        return false;
    }

    if(g_useColor||g_useDepth){
        k4a_calibration_t cal;
        if(K4A_FAILED(k4a_device_get_calibration(
            g_device,cfg.depth_mode,cfg.color_resolution,&cal))){
            std::cerr<<"[Error] get_calibration.\n";
            return false;
        }
        g_transform = k4a_transformation_create(&cal);
        if(!g_transform) return false;
    }

    if(g_useColor){
        cv::namedWindow(g_colorWin, cv::WINDOW_NORMAL);
        cv::resizeWindow(g_colorWin,640,480);
    }
    if(g_useDepth){
        cv::namedWindow(g_depthWin, cv::WINDOW_NORMAL);
        cv::resizeWindow(g_depthWin,640,480);
    }
    if(g_usePointCloud){
        if(!g_vis.CreateVisualizerWindow("3D Point Cloud",640,480)){
            return false;
        }
        g_vis.RegisterExitKey();

        g_pcPtr = std::make_shared<open3d::geometry::PointCloud>();
        g_vis.AddGeometry(g_pcPtr);

        auto axes = open3d::geometry::TriangleMesh::CreateCoordinateFrame(1.0,{0,0,0});
        g_vis.AddGeometry(axes);

        auto grid = CreateGridLineSet(10,0.2f);
        g_vis.AddGeometry(grid);

        g_vis.GetRenderOption().background_color_={0,0,0};
        g_vis.GetRenderOption().point_size_=1.0;

        InitSkeletonLineSet(g_vis);
    }

    g_firstCloud = true;
    return true;
}
void FinalizeAll(){
    if(g_usePointCloud){
        g_vis.Close();
        g_vis.DestroyVisualizerWindow();
    }
    if(g_transform){
        k4a_transformation_destroy(g_transform);
        g_transform=nullptr;
    }
    if(g_device){
        k4a_device_stop_cameras(g_device);
        k4a_device_close(g_device);
        g_device=nullptr;
    }
    if(g_useColor) cv::destroyWindow(g_colorWin);
    if(g_useDepth) cv::destroyWindow(g_depthWin);
}

//---------------------------------------------------------
// main
//---------------------------------------------------------
int main(){
    try{
        // ONNX 모델 로드 (실제 경로로 수정)
        g_poseNet = cv::dnn::readNetFromONNX("C:/project/obsbot/OBSBot_Test/src/cpp_orbbec/yolo11n-pose.onnx");
        g_poseNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        g_poseNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout<<"[INFO] YOLO model loaded.\n";
    }catch(std::exception &e){
        std::cerr<<"[ERROR] load onnx: "<<e.what()<<"\n";
        return -1;
    }

    // 사용자 입력
    std::cout<<"Use Color? (y/n): ";
    {
        std::string s; std::cin >> s;
        if(!s.empty() && (s[0]=='y'||s[0]=='Y')) g_useColor=true;
    }
    std::cout<<"Use Depth? (y/n): ";
    {
        std::string s; std::cin >> s;
        if(!s.empty() && (s[0]=='y'||s[0]=='Y')) g_useDepth=true;
    }
    std::cout<<"Use PointCloud? (y/n): ";
    {
        std::string s; std::cin >> s;
        if(!s.empty() && (s[0]=='y'||s[0]=='Y')){
            g_usePointCloud=true;
            if(!g_useDepth){
                std::cout<<"[INFO] enabling Depth.\n";
                g_useDepth=true;
            }
        }
    }

RESTART_ALL:
    if(!InitializeAll()){
        std::cerr<<"[ERROR] init.\n";
        return -1;
    }

    bool quit=false;
    while(!quit){
        if(g_usePointCloud){
            g_vis.PollEvents();
            if(g_vis.ShouldExit()){
                break;
            }
        }
        k4a_capture_t cap=nullptr;
        auto rs = k4a_device_get_capture(g_device,&cap,30);
        if(rs==K4A_WAIT_RESULT_TIMEOUT){
            continue;
        }else if(rs==K4A_WAIT_RESULT_FAILED){
            std::cerr<<"[ERROR] get_capture.\n";
            FinalizeAll();
            goto RESTART_ALL;
        }
        if(!cap) continue;

        // color / depth
        k4a_image_t colorImg = nullptr;
        if(g_useColor){
            colorImg = k4a_capture_get_color_image(cap);
        }
        k4a_image_t depthImg = nullptr;
        if(g_useDepth){
            depthImg = k4a_capture_get_depth_image(cap);
        }

        // depth 시각화
        cv::Mat depthMat;
        if(depthImg){
            int dw = k4a_image_get_width_pixels(depthImg);
            int dh = k4a_image_get_height_pixels(depthImg);
            uint16_t *dptr = (uint16_t*)k4a_image_get_buffer(depthImg);
            depthMat = cv::Mat(dh,dw,CV_16UC1,dptr);
            cv::Mat depthVis(dh,dw,CV_8UC1);
            for(int r=0;r<dh;r++){
                for(int c=0;c<dw;c++){
                    uint16_t v = depthMat.at<uint16_t>(r,c);
                    depthVis.at<uchar>(r,c) = (v>4000)? 255: (v/16);
                }
            }
            cv::Mat showDepth;
            cv::resize(depthVis, showDepth,{640,480});
            cv::imshow(g_depthWin, showDepth);
        }

        // 1) Aligned Color (Depth 해상도에 정렬)
        cv::Mat colorAlignBGR;
        if(colorImg && depthImg && g_transform){
            k4a_image_t alignColor = CreateAlignedColorImage(g_transform, depthImg, colorImg);
            if(alignColor){
                int aw = k4a_image_get_width_pixels(alignColor);
                int ah = k4a_image_get_height_pixels(alignColor);
                int stride = k4a_image_get_stride_bytes(alignColor);
                uint8_t *abuf = (uint8_t*)k4a_image_get_buffer(alignColor);
                cv::Mat ABGRA(ah,aw,CV_8UC4,(void*)abuf,stride);
                cv::cvtColor(ABGRA,colorAlignBGR,cv::COLOR_BGRA2BGR);

                // 화면 표시 (640x480)
                cv::Mat showAlign;
                cv::resize(colorAlignBGR, showAlign,{640,480});
                cv::imshow(g_colorWin, showAlign);

                k4a_image_release(alignColor);
            }
        }

        // 2) Pose on Aligned Color
        if(!colorAlignBGR.empty()){
            auto persons = InferPose(colorAlignBGR);
            if(!persons.empty()){
                auto &pMain = persons[0];
                if(pMain.score >= kConfidenceThreshold){
                    // 키포인트 히스토리
                    std::vector<cv::Point2f> nowKps(kNumKeypoints);
                    for(int i=0;i<kNumKeypoints;i++){
                        nowKps[i] = pMain.keypoints[i].pt;
                    }
                    g_kpHistory.push_back(nowKps);
                    if((int)g_kpHistory.size()>kHistorySize){
                        g_kpHistory.pop_front();
                    }
                    // 평균화
                    std::vector<cv::Point2f> avgKps(kNumKeypoints,cv::Point2f(0,0));
                    for(auto &frm: g_kpHistory){
                        for(int i=0;i<kNumKeypoints;i++){
                            avgKps[i]+=frm[i];
                        }
                    }
                    float inv = 1.f/g_kpHistory.size();
                    for(int i=0;i<kNumKeypoints;i++){
                        avgKps[i].x*=inv;
                        avgKps[i].y*=inv;
                    }

                    // 3D
                    if(!depthMat.empty() && g_usePointCloud){
                        int dw = depthMat.cols;
                        int dh = depthMat.rows;
                        std::vector<Eigen::Vector3d> skeleton3D(kNumKeypoints,{0,0,0});
                        for(int i=0;i<kNumKeypoints;i++){
                            float conf = pMain.keypoints[i].conf;
                            if(conf<0.5f) continue;
                            int px = (int)avgKps[i].x; 
                            int py = (int)avgKps[i].y;
                            if(px<0||px>=dw||py<0||py>=dh) continue;
                            uint16_t dVal = depthMat.at<uint16_t>(py,px);
                            double X=0,Y=0,Z=0;
                            if(PixelTo3D(g_transform,px,py,dVal,X,Y,Z)){
                                skeleton3D[i] = {X,Y,Z};
                            }
                        }
                        UpdateSkeleton3D(skeleton3D);
                        g_vis.UpdateGeometry(g_skelLineSet);
                        g_vis.UpdateRender();
                    }
                }
            }
        }

        // 3) PointCloud
        if(depthImg && g_usePointCloud && g_transform){
            // 포인트클라우드
            int dw = k4a_image_get_width_pixels(depthImg);
            int dh = k4a_image_get_height_pixels(depthImg);
            k4a_image_reference(depthImg);
            k4a_image_t pcImg=nullptr;
            if(K4A_SUCCEEDED(k4a_image_create(
                K4A_IMAGE_FORMAT_CUSTOM, dw,dh, dw*3*sizeof(int16_t), &pcImg))){
                if(K4A_SUCCEEDED(k4a_transformation_depth_image_to_point_cloud(
                    g_transform, depthImg,K4A_CALIBRATION_TYPE_DEPTH, pcImg))){
                    g_pcPtr->Clear();
                    g_pcPtr->points_.reserve(dw*dh);
                    g_pcPtr->colors_.reserve(dw*dh);

                    int16_t *b=(int16_t*)k4a_image_get_buffer(pcImg);
                    for(int i=0;i<dw*dh;i++){
                        int16_t xm=b[3*i+0];
                        int16_t ym=b[3*i+1];
                        int16_t zm=b[3*i+2];
                        if(zm==0) continue;
                        double X=(double)xm/1000.0;
                        double Y=(double)(-ym)/1000.0;
                        double Z=(double)(-zm)/1000.0;
                        g_pcPtr->points_.push_back({X,Y,Z});

                        double d = sqrt(X*X + Y*Y + Z*Z);
                        double maxD=4.0; 
                        if(d>maxD) d=maxD;
                        double t=d/maxD;
                        g_pcPtr->colors_.push_back({t,0.0,1.0-t});
                    }
                    if(g_firstCloud){
                        g_vis.ResetViewPoint(true);
                        g_firstCloud=false;
                    }
                    g_vis.UpdateGeometry(g_pcPtr);
                    g_vis.UpdateRender();
                }
                k4a_image_release(pcImg);
            }
        }

        // Cleanup
        if(colorImg) k4a_image_release(colorImg);
        if(depthImg) k4a_image_release(depthImg);
        k4a_capture_release(cap);
        DrainExtraFrames();

        int key = cv::waitKey(1);
        if(key==27 || key=='q'||key=='Q'){
            quit=true; break;
        }else if(key=='r'||key=='R'){
            if(g_usePointCloud){
                if(!RecreateVisualizerWindow(g_vis,g_pcPtr)){
                    quit=true; break;
                }
                g_firstCloud=true;
            }
        }else if(key=='d'||key=='D'){
            FinalizeAll();
            goto RESTART_ALL;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    FinalizeAll();
    return 0;
}