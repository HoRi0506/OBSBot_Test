# import cv2
# import numpy as np
# import open3d as o3d
# import sys

# sys.path.append("C:/project/pyorbbecsdk/install/lib")
# import pyorbbecsdk

# # -----------------------------
# # Open3D 시각화 관련 함수들
# # -----------------------------
# def create_visualizer():
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()

#     render_opt = vis.get_render_option()
#     render_opt.background_color = np.asarray([0, 0, 0])  # 배경은 검정
#     render_opt.point_size = 5.0
#     render_opt.light_on = True

#     # 카메라 뷰 초기 설정
#     view_control = vis.get_view_control()
#     view_control.set_zoom(0.8)
#     view_control.set_front([0.4257, -0.2125, -0.8795])
#     view_control.set_lookat([0, 0, 0])
#     view_control.set_up([-0.0694, -0.9768, 0])

#     return vis

# def update_visualizer(vis, points, voxel_size=0.005):
#     """
#     새 포인트 정보(points)를 받아서 PointCloud를 갱신.
#     points shape: (N, 3) 또는 (N, 6)
#       - [:, :3] = x, y, z
#       - [:, 3:6] = r, g, b (0~255 범위)
#     """
#     if points.size == 0:
#         return  # 아무 점이 없으면 패스

#     # 1) Open3D의 PointCloud 생성
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points[:, :3])
#     if points.shape[1] == 6:
#         pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)

#     # 2) 유효하지 않은 포인트(NaN 등) 제거
#     pcd.remove_non_finite_points()

#     # 3) y축 뒤집기(상하 반전) 적용
#     flip_transform = np.eye(4)
#     flip_transform[1, 1] = -1.0  # y축만 뒤집기
#     pcd.transform(flip_transform)

#     # 4) 다운샘플 (voxel_down_sample) 적용
#     #    voxel_size 값은 상황에 맞게 조절하세요.
#     if voxel_size > 0:
#         pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

#     # 좌표축은 한 번만 추가
#     if not hasattr(update_visualizer, "coord_added"):
#         coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
#         vis.add_geometry(coord)
#         update_visualizer.coord_added = True

#     # 최초 한 번만 add, 이후는 업데이트
#     if not hasattr(update_visualizer, "pcd_added"):
#         vis.add_geometry(pcd)
#         update_visualizer.pcd_added = pcd
#     else:
#         update_visualizer.pcd_added.points = pcd.points
#         update_visualizer.pcd_added.colors = pcd.colors
#         vis.update_geometry(update_visualizer.pcd_added)

#     # Open3D 이벤트 처리
#     vis.poll_events()
#     vis.update_renderer()


# # -----------------------------
# # 메인 실행부
# # -----------------------------
# def main():
#     # 1) 파이프라인 생성
#     pipeline = pyorbbecsdk.Pipeline()
#     config = pyorbbecsdk.Config()

#     # -------------------------
#     # 스트림 프로파일 "명시적" 설정
#     #  - Color: 1280 x 720 @ 30, MJPG
#     #  - Depth: 640 x 576 @ 30, Y16
#     # -------------------------
#     try:
#         color_profiles = pipeline.get_stream_profile_list(pyorbbecsdk.OBSensorType.COLOR_SENSOR)
#         color_profile = color_profiles.get_video_stream_profile(
#             1280, 720, pyorbbecsdk.OBFormat.MJPG, 30
#         )
#         print("Selected Color Profile: {}x{}@{} [{}]".format(
#             color_profile.get_width(),
#             color_profile.get_height(),
#             color_profile.get_fps(),
#             color_profile.get_format()))
#         config.enable_stream(color_profile)

#         depth_profiles = pipeline.get_stream_profile_list(pyorbbecsdk.OBSensorType.DEPTH_SENSOR)
#         depth_profile = depth_profiles.get_video_stream_profile(
#             640, 576, pyorbbecsdk.OBFormat.Y16, 30
#         )
#         print("Selected Depth Profile: {}x{}@{} [{}]".format(
#             depth_profile.get_width(),
#             depth_profile.get_height(),
#             depth_profile.get_fps(),
#             depth_profile.get_format()))
#         config.enable_stream(depth_profile)

#     except Exception as e:
#         print("Error enabling stream profiles:", e)
#         return

#     # -------------------------
#     # 2) 파이프라인 시작
#     # -------------------------
#     try:
#         pipeline.enable_frame_sync()
#         pipeline.start(config)
#     except Exception as e:
#         print("Error starting pipeline:", e)
#         return

#     # 3) 카메라 파라미터 (PointCloud 생성을 위해 필요)
#     camera_param = pipeline.get_camera_param()

#     # -------------------------
#     # 4) GPU 사용 설정 (SDK 버전에 따라 다를 수 있음)
#     #    - 정렬 시 GPU 사용
#     #    - PointCloud 생성 시 GPU 사용
#     # -------------------------
#     align_filter = pyorbbecsdk.AlignFilter(align_to_stream=pyorbbecsdk.OBStreamType.COLOR_STREAM)
#     # 정렬 시 GPU 사용
#     try:
#         align_filter.set_engine_type(pyorbbecsdk.OBEngineType.ENGINE_GPU)
#     except:
#         print("[Warning] align_filter.set_engine_type(OBEngineType.ENGINE_GPU) is not supported in this SDK version.")

#     point_cloud_filter = pyorbbecsdk.PointCloudFilter()
#     point_cloud_filter.set_camera_param(camera_param)
#     # PointCloud 생성 시 GPU 사용
#     try:
#         point_cloud_filter.set_cloud_engine(pyorbbecsdk.OBCloudEngineType.ENGINE_GPU)
#     except:
#         print("[Warning] point_cloud_filter.set_cloud_engine(OBCloudEngineType.ENGINE_GPU) is not supported in this SDK version.")

#     # -------------------------
#     # Open3D 시각화 창 생성
#     # -------------------------
#     vis = create_visualizer()
#     pyorbbecsdk.Context().set_logger_level(pyorbbecsdk.OBLogLevel.NONE)

#     print("Start streaming... Press ESC to exit.")

#     try:
#         while True:
#             # -------------------------
#             # 프레임 획득
#             # -------------------------
#             frames = pipeline.wait_for_frames(100)
#             if frames is None:
#                 continue

#             color_frame = frames.get_color_frame()
#             depth_frame = frames.get_depth_frame()
#             if color_frame is None or depth_frame is None:
#                 continue

#             # -------------------------
#             # (A) OpenCV로 Color/Depth 출력
#             # -------------------------
#             color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
#             color_img = cv2.imdecode(color_data, cv2.IMREAD_UNCHANGED)
#             if color_img is None:
#                 print("Failed to decode color frame")
#                 continue

#             w_d, h_d = depth_frame.get_width(), depth_frame.get_height()
#             depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
#             try:
#                 depth_map = depth_data.reshape((h_d, w_d))
#             except Exception as ex:
#                 print("Depth reshape error:", ex)
#                 continue

#             depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#             depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

#             cv2.imshow("Orbbec Color", color_img)
#             cv2.imshow("Orbbec Depth", depth_vis)

#             # ESC 키 감지
#             if cv2.waitKey(1) == 27:
#                 break

#             # -------------------------
#             # (B) Point Cloud 생성 (Open3D 시각화)
#             # -------------------------
#             aligned_frames = align_filter.process(frames)
#             if aligned_frames is None:
#                 continue

#             color_frame_aligned = aligned_frames.get_color_frame()
#             depth_frame_aligned = aligned_frames.get_depth_frame()
#             if color_frame_aligned is None or depth_frame_aligned is None:
#                 continue

#             scale = depth_frame_aligned.get_depth_scale()
#             point_cloud_filter.set_position_data_scaled(scale)

#             # 컬러가 정상적으로 있으면 RGB_POINT, 없으면 POINT
#             if color_frame_aligned is not None:
#                 point_cloud_filter.set_create_point_format(pyorbbecsdk.OBFormat.RGB_POINT)
#             else:
#                 point_cloud_filter.set_create_point_format(pyorbbecsdk.OBFormat.POINT)

#             pcd_frame = point_cloud_filter.process(aligned_frames)
#             if pcd_frame is None:
#                 continue

#             points_data = point_cloud_filter.calculate(pcd_frame)
#             if points_data is None:
#                 continue

#             # numpy 배열 → Open3D 시각화
#             points_array = np.array(points_data)
            
#             # voxel_down_sample 적용 포함
#             update_visualizer(vis, points_array, voxel_size=0.005)

#     except Exception as e:
#         print("Exception:", e)
#     finally:
#         pipeline.stop()
#         vis.destroy_window()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import open3d as o3d
import sys

sys.path.append("C:/project/pyorbbecsdk/install/lib")
import pyorbbecsdk

def colorize_by_depth(pcd):
    """
    pcd의 모든 점에 대해,
    가까운 점 => 파랑(0,0,1),
    먼 점 => 빨강(1,0,0) 으로 선형 보간하여 색 지정
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return
    # 센서 원점(0,0,0) 기준 거리
    dist = np.linalg.norm(pts, axis=1)
    min_d, max_d = dist.min(), dist.max()
    if max_d - min_d < 1e-9:
        # 모든 점의 거리가 거의 같으면 단색(파랑)
        colors = np.zeros_like(pts, dtype=np.float32)
        colors[:, 2] = 1.0  # 파랑
    else:
        # [0~1]로 정규화
        norm = (dist - min_d) / (max_d - min_d)  # 가까우면 0, 멀수록 1
        # 파랑(0,0,1) ~ 빨강(1,0,0)
        r = norm
        g = np.zeros_like(norm)
        b = 1.0 - norm
        colors = np.stack([r, g, b], axis=-1)

    pcd.colors = o3d.utility.Vector3dVector(colors)


def create_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    render_opt = vis.get_render_option()
    # 배경: 검정색
    render_opt.background_color = np.asarray([0, 0, 0])
    # 점 크기
    render_opt.point_size = 1.0
    render_opt.light_on = True

    # 카메라 뷰 초기 설정 (원하는 위치로 조정 가능)
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    view_control.set_front([0.4257, -0.2125, -0.8795])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([-0.0694, -0.9768, 0])

    return vis

def update_visualizer(vis, points, voxel_size=0.005):
    """
    (A) NumPy -> PointCloud
    (B) y축 뒤집기
    (C) voxel_down_sample
    (D) 거리 기반 색상(colorize_by_depth)
    (E) 한 번만 add, 이후 update
    """
    if points.size == 0:
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # 유효하지 않은 포인트 제거
    pcd.remove_non_finite_points()

    # y축 뒤집기
    flip = np.eye(4)
    flip[1, 1] = -1.0
    pcd.transform(flip)

    # 다운샘플 (필요하다면 주석 처리 가능)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    # 깊이 기반으로 파랑 ~ 빨강 색상 매핑
    colorize_by_depth(pcd)

    # 매 프레임마다 갱신
    if not hasattr(update_visualizer, "pcd_added"):
        # 좌표축 1회만 추가
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        vis.add_geometry(coord)

        vis.add_geometry(pcd)
        update_visualizer.pcd_added = pcd
    else:
        # 기존 pcd에 좌표/색만 덮어쓰기
        update_visualizer.pcd_added.points = pcd.points
        update_visualizer.pcd_added.colors = pcd.colors
        vis.update_geometry(update_visualizer.pcd_added)

    vis.poll_events()
    vis.update_renderer()


def main():
    # 1) 파이프라인
    pipeline = pyorbbecsdk.Pipeline()
    config = pyorbbecsdk.Config()

    # 스트림 설정
    try:
        color_profiles = pipeline.get_stream_profile_list(pyorbbecsdk.OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(
            1280, 720, pyorbbecsdk.OBFormat.MJPG, 30
        )
        config.enable_stream(color_profile)
        print(f"Selected Color Profile: {color_profile.get_width()}x{color_profile.get_height()}@{color_profile.get_fps()} [{color_profile.get_format()}]")

        depth_profiles = pipeline.get_stream_profile_list(pyorbbecsdk.OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_video_stream_profile(
            640, 576, pyorbbecsdk.OBFormat.Y16, 30
        )
        config.enable_stream(depth_profile)
        print(f"Selected Depth Profile: {depth_profile.get_width()}x{depth_profile.get_height()}@{depth_profile.get_fps()} [{depth_profile.get_format()}]")
    except Exception as e:
        print("Error enabling stream profiles:", e)
        return

    # 파이프라인 시작
    try:
        pipeline.enable_frame_sync()
        pipeline.start(config)
    except Exception as e:
        print("Error starting pipeline:", e)
        return

    # 카메라 파라미터
    camera_param = pipeline.get_camera_param()

    # 필터 설정
    align_filter = pyorbbecsdk.AlignFilter(align_to_stream=pyorbbecsdk.OBStreamType.COLOR_STREAM)
    try:
        align_filter.set_engine_type(pyorbbecsdk.OBEngineType.ENGINE_GPU)
    except:
        pass

    point_cloud_filter = pyorbbecsdk.PointCloudFilter()
    point_cloud_filter.set_camera_param(camera_param)
    try:
        point_cloud_filter.set_cloud_engine(pyorbbecsdk.OBCloudEngineType.ENGINE_GPU)
    except:
        pass

    # Open3D 시각화
    vis = create_visualizer()
    pyorbbecsdk.Context().set_logger_level(pyorbbecsdk.OBLogLevel.NONE)

    print("Start streaming... Press ESC to exit.")

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            color_frame = frames.get_color_frame()
            # 아래는 미리보기용, 실제로는 color_frame를 쓰지 않고
            # depth만으로 포인트 색을 입힘
            if color_frame is not None:
                color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                color_img = cv2.imdecode(color_data, cv2.IMREAD_UNCHANGED)
                if color_img is not None:
                    cv2.imshow("Orbbec Color", color_img)

            # Depth 미리보기
            w_d, h_d = depth_frame.get_width(), depth_frame.get_height()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            try:
                depth_map = depth_data.reshape((h_d, w_d))
            except:
                continue

            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow("Orbbec Depth", depth_vis)

            if cv2.waitKey(1) == 27:  # ESC
                break

            # Point Cloud 생성 (Depth→Color 정렬)
            aligned = align_filter.process(frames)
            if aligned is None:
                continue

            depth_aligned = aligned.get_depth_frame()
            if depth_aligned is None:
                continue

            scale = depth_aligned.get_depth_scale()
            point_cloud_filter.set_position_data_scaled(scale)

            # 컬러가 필요 없으므로 POINT 포맷으로 고정
            point_cloud_filter.set_create_point_format(pyorbbecsdk.OBFormat.POINT)

            pcd_frame = point_cloud_filter.process(aligned)
            if pcd_frame is None:
                continue

            points_data = point_cloud_filter.calculate(pcd_frame)
            if points_data is None:
                continue

            points_array = np.array(points_data)
            # 3D 시각화 업데이트
            update_visualizer(vis, points_array, voxel_size=0.005)

    except Exception as e:
        print("Exception:", e)
    finally:
        pipeline.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
