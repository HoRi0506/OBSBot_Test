import cv2
import numpy as np
import open3d as o3d
import sys
import threading
import queue
import time
from collections import deque
from ultralytics import YOLO

# pyorbbecsdk 설치 경로 추가
sys.path.append("C:/clone/pyorbbecsdk/install/lib")
from pyorbbecsdk import Pipeline, Config, OBSensorType

# YOLO pose 모델 로드 (사람 스켈레톤 용)
model_pose = YOLO("yolo11n-pose.pt")

# COCO 데이터셋의 (기본) 스켈레톤 연결 순서 예시
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (11, 12)
]

###############################################################################
# 2D 스켈레톤 관련 보조 함수
###############################################################################
def draw_skeleton_2d(image, keypoints_2d, skeleton_pairs, color=(0,0,255), radius=3, thickness=2):
    """
    2D 이미지에 스켈레톤을 그려주는 함수
    - image: numpy 배열(BGR)
    - keypoints_2d: shape = (17, 2) 등등 (COCO 17개 기준)
    - skeleton_pairs: 스켈레톤 연결 정보 (SKELETON 상수)
    """
    for (x, y) in keypoints_2d:
        if x is None or y is None:
            continue
        cv2.circle(image, (int(x), int(y)), radius, (0,255,0), -1)  # 관절점(초록)
    # 뼈대 연결(빨강)
    for (st, ed) in skeleton_pairs:
        if st < len(keypoints_2d) and ed < len(keypoints_2d):
            x1, y1 = keypoints_2d[st]
            x2, y2 = keypoints_2d[ed]
            if (x1 is not None and y1 is not None) and (x2 is not None and y2 is not None):
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def average_keypoints(keypoints_list):
    """
    큐에 모인 여러 프레임의 keypoints(각각 shape = (17,2) 가정)를 받아
    평균 2D 좌표를 구해 반환.
    - keypoints_list: [(17,2), (17,2), ...] 형태 (길이 = queue.maxlen)
    - 반환: (17,2) 형태의 평균 좌표
      * 단, 특정 관절이 유효하지 않은(None) 경우는 평균에서 제외
    """
    if not keypoints_list:
        return None
    
    # keypoints_list를 (N, 17, 2) 형태로 쌓기 위해 None->np.nan 처리
    np_list = []
    for kps in keypoints_list:
        arr = []
        for (x, y) in kps:
            if x is None or y is None:
                arr.append([np.nan, np.nan])
            else:
                arr.append([x, y])
        np_list.append(arr)
    np_array = np.array(np_list, dtype=np.float32)  # shape = (N, 17, 2)

    # 관절별로 NaN이 아닌 값만 평균
    mean_kps = np.nanmean(np_array, axis=0)  # shape = (17, 2)
    
    # 완전히 NaN인 경우(해당 관절 전부 유효 x)는 결과가 NaN -> None으로
    averaged_keypoints = []
    for (x, y) in mean_kps:
        if np.isnan(x) or np.isnan(y):
            averaged_keypoints.append((None, None))
        else:
            averaged_keypoints.append((x, y))
    return averaged_keypoints

def get_first_person_keypoints(result, image_w, image_h):
    """
    YOLO pose 결과에서 가장 첫 번째(또는 가장 확실한) 사람의 keypoints (17,2) 반환.
    - 사람 미검출 시 None 반환
    """
    if (len(result.boxes) == 0) or (result.keypoints is None):
        return None
    
    kp_all = result.keypoints.xy  # shape: (#detected, 17, 2)
    if kp_all is None or len(kp_all) == 0:
        return None

    # 필요 시, conf값 제일 높은 사람 등으로 고를 수도 있음. 여기서는 0번째
    idx = 0
    if idx >= len(kp_all):
        return None
    
    keypoints_2d = kp_all[idx].cpu().numpy().astype(np.float32)  # (17,2)
    
    # 이미지 범위 벗어나거나 (0,0)에 가까운 경우 등은 None 처리
    cleaned_kps = []
    for (x, y) in keypoints_2d:
        xi, yi = int(x), int(y)
        if xi < 0 or xi >= image_w or yi < 0 or yi >= image_h:
            cleaned_kps.append((None, None))
        elif (xi == 0 and yi == 0):
            cleaned_kps.append((None, None))
        else:
            cleaned_kps.append((x, y))
    
    return cleaned_kps

###############################################################################
# 3D 표시(스켈레톤) 관련 보조 함수
###############################################################################
def create_grid(size=1.0, divisions=10):
    """
    보기 좋게 바닥면에 깔 그리드 LineSet 생성
    """
    points = []
    lines = []
    step = size / divisions
    for i in range(divisions + 1):
        x = -size/2 + i * step
        points.append([x, 0, -size/2])
        points.append([x, 0, size/2])
        lines.append([2*i, 2*i+1])
    offset = (divisions+1)*2
    for j in range(divisions + 1):
        z = -size/2 + j * step
        points.append([-size/2, 0, z])
        points.append([size/2, 0, z])
        lines.append([offset+2*j, offset+2*j+1])
    colors = [[0.7, 0.7, 0.7] for _ in range(len(lines))]
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector(colors)
    return grid

def draw_skeleton_3d(vis, skeleton_3d, skeleton_pairs, line_set_handle, keypoint_cloud_handle):
    """
    3D 스켈레톤을 (LineSet + 키포인트 PointCloud) 형태로 새로 생성하여 Visualizer에 추가.
    - vis: o3d.visualization.Visualizer
    - skeleton_3d: [(x,y,z), ...] 길이 17 (COCO 기준)
    - skeleton_pairs: SKELETON 연결 정보
    - line_set_handle: 이전 프레임에 사용한 LineSet
    - keypoint_cloud_handle: 이전 프레임에 사용한 키포인트 PointCloud
    """
    # 1) 이전 라인셋과 관절점 구름 제거
    if line_set_handle is not None:
        vis.remove_geometry(line_set_handle, reset_bounding_box=False)
    if keypoint_cloud_handle is not None:
        vis.remove_geometry(keypoint_cloud_handle, reset_bounding_box=False)
    
    # 2) 새 LineSet(빨간 선)
    # 유효하지 않은 좌표는 (0,0,0)으로 대체하지만, 라인 생성 시 유효 여부를 확인함
    points_3d = []
    validity = []
    for (x, y, z) in skeleton_3d:
        if x is None or y is None or z is None:
            points_3d.append([0, 0, 0])
            validity.append(False)
        else:
            points_3d.append([x, y, z])
            validity.append(True)
    
    lines = []
    for (st, ed) in skeleton_pairs:
        if st < len(points_3d) and ed < len(points_3d):
            # 두 관절 모두 유효할 때만 라인을 추가
            if validity[st] and validity[ed]:
                lines.append([st, ed])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_3d)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    
    # 3) 새 키포인트 PointCloud(초록 점)
    keypoint_pc = o3d.geometry.PointCloud()
    keypoint_pc.points = o3d.utility.Vector3dVector(points_3d)
    key_colors = [[0, 1, 0] for _ in range(len(points_3d))]
    keypoint_pc.colors = o3d.utility.Vector3dVector(key_colors)
    
    # 4) vis 에 추가
    vis.add_geometry(line_set)
    vis.add_geometry(keypoint_pc)
    
    return line_set, keypoint_pc

###############################################################################
# 파이프라인 및 윈도우 관련 함수
###############################################################################
def frame_acquisition(pipeline, frame_queue, running_event):
    while running_event.is_set():
        try:
            frames = pipeline.wait_for_frames(1000)
            if frames is not None:
                try:
                    frame_queue.put(frames, timeout=1)
                except queue.Full:
                    pass
        except Exception as e:
            print("프레임 획득 중 예외 발생:", e)
            continue

def get_user_window_selection():
    print("실행 시에 표시할 창을 선택하세요. (복수 선택 가능, 예: 1,3,4)")
    print("1. Color Window")
    print("2. Depth Window")
    print("3. 3D Point Cloud (Open3D)")
    print("4. 3D Point Cloud with Skeleton (Open3D)")
    selection = input("선택 (쉼표로 구분): ")
    selections = {s.strip() for s in selection.split(",")}
    show_color = "1" in selections
    show_depth = "2" in selections
    show_point_cloud = "3" in selections
    show_point_cloud_skeleton = "4" in selections
    return show_color, show_depth, show_point_cloud, show_point_cloud_skeleton

def reinitialize_windows(show_color, show_depth, show_point_cloud, show_point_cloud_skeleton):
    cv2.destroyAllWindows()
    vis_pc = None
    pcd_pc = None
    vis_pcs = None
    pcd_pcs = None

    if show_point_cloud:
        vis_pc = o3d.visualization.Visualizer()
        vis_pc.create_window(window_name="3D Point Cloud", width=640, height=576)
        pcd_pc = o3d.geometry.PointCloud()
        vis_pc.add_geometry(pcd_pc)
        
        grid = create_grid(size=2.0, divisions=20)
        vis_pc.add_geometry(grid)
        
        render_opt = vis_pc.get_render_option()
        render_opt.point_size = 1.0
        render_opt.background_color = np.asarray([0.05, 0.05, 0.05])
        
        view_control = vis_pc.get_view_control()
        view_control.set_zoom(0.3)
    
    if show_point_cloud_skeleton:
        vis_pcs = o3d.visualization.Visualizer()
        vis_pcs.create_window(window_name="3D Point Cloud with Skeleton", width=640, height=576)
        pcd_pcs = o3d.geometry.PointCloud()
        vis_pcs.add_geometry(pcd_pcs)
        
        grid = create_grid(size=2.0, divisions=20)
        vis_pcs.add_geometry(grid)
        
        render_opt_s = vis_pcs.get_render_option()
        render_opt_s.point_size = 1.0
        render_opt_s.background_color = np.asarray([0.05, 0.05, 0.05])
        
        view_control_s = vis_pcs.get_view_control()
        view_control_s.set_zoom(0.3)
    
    return vis_pc, pcd_pc, vis_pcs, pcd_pcs

###############################################################################
# 메인
###############################################################################
def main():
    show_color, show_depth, show_point_cloud, show_point_cloud_skeleton = get_user_window_selection()
    
    sample_interval = 0.1
    window_watchdog_timeout = 3.0
    last_window_update_time = time.time()
    
    pipeline = Pipeline()
    config = Config()
    
    # 1) Color/Depth 프로필 지정
    depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)\
                            .get_default_video_stream_profile()
    config.enable_stream(depth_profile)
    
    color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)\
                            .get_default_video_stream_profile()
    config.enable_stream(color_profile)
    
    pipeline.start(config)

    # 2) 카메라 파라미터
    cam_param = pipeline.get_camera_param()
    
    # [Depth Intrinsics]
    depth_intr = cam_param.depth_intrinsic
    depth_fx, depth_fy = depth_intr.fx, depth_intr.fy
    depth_cx, depth_cy = depth_intr.cx, depth_intr.cy
    depth_width, depth_height = depth_intr.width, depth_intr.height
    
    # [Color Intrinsics]
    color_intr = cam_param.rgb_intrinsic
    color_fx_orig = color_intr.fx
    color_fy_orig = color_intr.fy
    color_cx_orig = color_intr.cx
    color_cy_orig = color_intr.cy
    color_width_orig = color_intr.width
    color_height_orig = color_intr.height
    
    # (Depth 해상도)에 맞춰 Color 프레임 리사이즈
    dw, dh = depth_width, depth_height
    scale_x = dw / float(color_width_orig)
    scale_y = dh / float(color_height_orig)
    
    color_fx_scaled = color_fx_orig * scale_x
    color_fy_scaled = color_fy_orig * scale_y
    color_cx_scaled = color_cx_orig * scale_x
    color_cy_scaled = color_cy_orig * scale_y
    
    # Depth 해상도 기준 (u,v) 좌표
    u_coords = np.tile(np.arange(dw), dh)
    v_coords = np.repeat(np.arange(dh), dw)
    
    # Open3D 창 초기화
    vis_pc, pcd_pc, vis_pcs, pcd_pcs = reinitialize_windows(
        show_color, show_depth, show_point_cloud, show_point_cloud_skeleton
    )
    
    frame_queue = queue.Queue(maxsize=5)
    running_event = threading.Event()
    running_event.set()
    acquisition_thread = threading.Thread(target=frame_acquisition, args=(pipeline, frame_queue, running_event))
    acquisition_thread.start()
    
    last_process_time = 0
    print("Streaming started... Press 'q' or ESC to quit.")

    # 스켈레톤 좌표(2D)를 모아둘 큐
    skeleton_queue = deque(maxlen=5)

    # 3D 스켈레톤용 이전 프레임 라인셋 & 키포인트
    skeleton_lineset = None
    skeleton_keypoints_pc = None

    try:
        while True:
            # 윈도우 감시(watchdog)
            current_time = time.time()
            if current_time - last_window_update_time > window_watchdog_timeout:
                print("Window unresponsive, reinitializing windows...")
                if vis_pc is not None:
                    vis_pc.destroy_window()
                if vis_pcs is not None:
                    vis_pcs.destroy_window()
                vis_pc, pcd_pc, vis_pcs, pcd_pcs = reinitialize_windows(
                    show_color, show_depth, show_point_cloud, show_point_cloud_skeleton
                )
                last_window_update_time = current_time

            # 프레임 가져오기
            frames = None
            try:
                while True:
                    frames = frame_queue.get_nowait()
            except queue.Empty:
                pass
            if frames is None:
                continue
            
            # 처리 간격
            current_time = time.time()
            if current_time - last_process_time < sample_interval:
                time.sleep(0.005)
                continue
            last_process_time = current_time

            # 가져온 프레임에서 Color / Depth 추출
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue
            
            # ======================
            #  Color Frame
            # ======================
            w, h = color_frame.get_width(), color_frame.get_height()
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            fmt = color_frame.get_format().name
            
            if fmt.startswith("MJPG"):
                color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
            elif fmt.startswith("YUYV"):
                color_yuv = color_data.reshape(h, w, 2)
                color_image = cv2.cvtColor(color_yuv, cv2.COLOR_YUV2BGR_YUYV)
            else:
                color_image = color_data.reshape(h, w, 3)
                if fmt == "RGB":
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            
            color_image = cv2.flip(color_image, 1)
            if (color_image.shape[1] != dw) or (color_image.shape[0] != dh):
                color_image = cv2.resize(color_image, (dw, dh), interpolation=cv2.INTER_AREA)
            
            # ======================
            #  Depth Frame
            # ======================
            depth_scale = depth_frame.get_depth_scale()
            d_w, d_h = depth_frame.get_width(), depth_frame.get_height()
            if d_w != dw or d_h != dh:
                print("경고: Depth 해상도가 예상과 다릅니다.", d_w, d_h, "vs expected:", dw, dh)
            
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(dh, dw)
            depth_mm = depth_data.astype(np.float32) * depth_scale
            # 센서별 보정 - 필요시 수정
            depth_m = depth_mm / 4000.0

            # depth 시각화 (COLORMAP_JET)
            depth_norm = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            depth_colormap = cv2.flip(depth_colormap, 1)
            
            # ======================
            #  3D Point Cloud
            # ======================
            Z_raw = depth_m.flatten()  # (dw*dh,)
            X = (u_coords - depth_cx) / depth_fx * Z_raw
            Y = -(v_coords - depth_cy) / depth_fy * Z_raw
            Z = -Z_raw
            points_all = np.vstack((-X, Y, Z)).T  # open3d 좌표계 (x: -X, y: Y, z: Z)
            
            valid_idx = (Z_raw > 0)
            points_valid = points_all[valid_idx, :]

            # --> 공통 color mapping
            depth_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
            colors_all = depth_rgb.reshape(-1, 3).astype(np.float32) / 255.0
            colors_valid = colors_all[valid_idx, :]

            # Point Cloud Window
            if show_point_cloud and vis_pc is not None and pcd_pc is not None:
                pcd_pc.points = o3d.utility.Vector3dVector(points_valid)
                pcd_pc.colors = o3d.utility.Vector3dVector(colors_valid)
                vis_pc.update_geometry(pcd_pc)
                vis_pc.poll_events()
                vis_pc.update_renderer()
                last_window_update_time = time.time()

            # ======================
            #  스켈레톤 (YOLO Pose)
            # ======================
            results = model_pose.predict(color_image, verbose=False)
            
            keypoints_2d = None
            if len(results) > 0:
                keypoints_2d = get_first_person_keypoints(results[0], dw, dh)
            
            if keypoints_2d is not None:
                skeleton_queue.append(keypoints_2d)
            else:
                skeleton_queue.clear()

            # 큐가 가득 차면 평균 2D 좌표 계산
            if len(skeleton_queue) == skeleton_queue.maxlen:
                avg_kps_2d = average_keypoints(list(skeleton_queue))

                # Color / Depth 윈도우 2D 스켈레톤 표시
                if show_color:
                    draw_skeleton_2d(color_image, avg_kps_2d, SKELETON, color=(0,0,255))
                if show_depth:
                    # depth_colormap 에 2D 스켈레톤
                    draw_skeleton_2d(depth_colormap, avg_kps_2d, SKELETON, color=(0,0,255))

                # 3D 스켈레톤 계산
                skeleton_3d = []
                for (x2d, y2d) in avg_kps_2d:
                    if x2d is None or y2d is None:
                        skeleton_3d.append((None, None, None))
                        continue
                    u_pix = int(round(x2d))
                    v_pix = int(round(y2d))
                    z_val = depth_m[v_pix, u_pix]
                    if z_val <= 0:
                        skeleton_3d.append((None, None, None))
                    else:
                        # color intrinsics(스케일된)로 2D->3D
                        X3d = (u_pix - color_cx_scaled) / color_fx_scaled * z_val
                        Y3d = -((v_pix - color_cy_scaled) / color_fy_scaled * z_val)
                        Z3d = -z_val
                        # Open3D 기준 (-X, Y, Z)로 보정
                        skeleton_3d.append((-X3d, Y3d, Z3d))

                # Point Cloud with Skeleton
                if show_point_cloud_skeleton and vis_pcs is not None and pcd_pcs is not None:
                    # 배경 포인트 클라우드 표시 (동일 컬러)
                    pcd_pcs.points = o3d.utility.Vector3dVector(points_valid)
                    pcd_pcs.colors = o3d.utility.Vector3dVector(colors_valid)
                    vis_pcs.update_geometry(pcd_pcs)

                    # 스켈레톤 라인/키포인트
                    skeleton_lineset, skeleton_keypoints_pc = draw_skeleton_3d(
                        vis_pcs,
                        skeleton_3d,
                        SKELETON,
                        skeleton_lineset,
                        skeleton_keypoints_pc
                    )
                    vis_pcs.poll_events()
                    vis_pcs.update_renderer()
                    last_window_update_time = time.time()

            else:
                # 큐 미충족 시에는 우선 Color/Depth 창만 업데이트
                if show_color:
                    cv2.imshow("Color", color_image)
                if show_depth:
                    cv2.imshow("Depth", depth_colormap)
                
                # Point Cloud with Skeleton 윈도우에서 사람 스켈레톤은 이전 상태 그대로
                # (단, 매 프레임 포인트 업데이트)
                if show_point_cloud_skeleton and vis_pcs is not None and pcd_pcs is not None:
                    pcd_pcs.points = o3d.utility.Vector3dVector(points_valid)
                    pcd_pcs.colors = o3d.utility.Vector3dVector(colors_valid)
                    vis_pcs.update_geometry(pcd_pcs)
                    vis_pcs.poll_events()
                    vis_pcs.update_renderer()
                    last_window_update_time = time.time()

                # 종료키 체크
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
                time.sleep(0.005)
                continue

            # 최종 윈도우 표시
            if show_color:
                cv2.imshow("Color", color_image)
            if show_depth:
                cv2.imshow("Depth", depth_colormap)

            # 종료키
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
            
            time.sleep(0.005)

    finally:
        running_event.clear()
        acquisition_thread.join()
        if vis_pc is not None:
            vis_pc.destroy_window()
        if vis_pcs is not None:
            vis_pcs.destroy_window()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
