## color version
# import cv2
# import numpy as np
# import open3d as o3d
# import sys

# sys.path.append("C:/project/pyorbbecsdk/install/lib")
# # import pyorbbecsdk

# from pyorbbecsdk import *


# def create_visualizer():
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     render_opt = vis.get_render_option()
#     render_opt.background_color = np.asarray([0, 0, 0])
#     render_opt.point_size = 1.0
#     return vis


# def update_visualizer(vis, points):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
#     # y축 뒤집기
#     flip = np.eye(4)
#     flip[1, 1] = -1.0
#     pcd.transform(flip)
#     if points.shape[1] == 6:
#         pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)
#     vis.clear_geometries()
#     vis.add_geometry(pcd)
#     vis.poll_events()
#     vis.update_renderer()


# def main():
#     context = Context()
#     pipeline = Pipeline()
#     config = Config()
#     depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
#     if depth_profile_list is None:
#         print("No proper depth profile, can not generate point cloud")
#         return
#     depth_profile = depth_profile_list.get_default_video_stream_profile()
#     config.enable_stream(depth_profile)
#     has_color_sensor = False
#     try:
#         profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
#         if profile_list is not None:
#             color_profile = profile_list.get_default_video_stream_profile()
#             config.enable_stream(color_profile)
#             has_color_sensor = True
#     except OBError as e:
#         print(e)
#     pipeline.enable_frame_sync()
#     pipeline.start(config)
#     camera_param = pipeline.get_camera_param()
#     align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
#     point_cloud_filter = PointCloudFilter()
#     point_cloud_filter.set_camera_param(camera_param)

#     vis = create_visualizer()
#     context.set_logger_level(OBLogLevel.NONE)
#     print("start to show point cloud, press 'ctrl+c' to exit")
#     try:
#         while True:
#             frames = pipeline.wait_for_frames(100)
#             if frames is None:
#                 continue
#             depth_frame = frames.get_depth_frame()
#             if depth_frame is None:
#                 continue
#             color_frame = frames.get_color_frame()
#             if has_color_sensor and color_frame is None:
#                 continue
#             frame = align_filter.process(frames)
#             scale = depth_frame.get_depth_scale()
#             point_cloud_filter.set_position_data_scaled(scale)

#             point_cloud_filter.set_create_point_format(
#                 OBFormat.RGB_POINT if has_color_sensor and color_frame is not None else OBFormat.POINT)
#             point_cloud_frame = point_cloud_filter.process(frame)
#             points = point_cloud_filter.calculate(point_cloud_frame)
#             update_visualizer(vis, np.array(points))

#     except Exception as e:
#         print(e)

#     finally:
#         pipeline.stop()
#         vis.destroy_window()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()


########################################################################################
## depth version
########################################################################################
import cv2
import numpy as np
import open3d as o3d
import sys

sys.path.append("C:/project/pyorbbecsdk/install/lib")
from pyorbbecsdk import *


def create_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([0, 0, 0])
    render_opt.point_size = 1.0
    return vis


def update_visualizer(vis, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # y축 뒤집기 (좌표계 보정)
    flip = np.eye(4)
    flip[1, 1] = -1.0
    pcd.transform(flip)
    
    # 컬러 정보가 없으면 색상 설정은 생략
    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()


def main():
    context = Context()
    pipeline = Pipeline()
    config = Config()
    
    # depth 스트림 프로파일만 사용합니다.
    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    if depth_profile_list is None:
        print("No proper depth profile, cannot generate point cloud")
        return
    depth_profile = depth_profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)
    
    pipeline.enable_frame_sync()
    pipeline.start(config)
    
    # 카메라 파라미터 획득 (depth 내재 행렬 등)
    camera_param = pipeline.get_camera_param()
    point_cloud_filter = PointCloudFilter()
    point_cloud_filter.set_camera_param(camera_param)
    
    vis = create_visualizer()
    context.set_logger_level(OBLogLevel.NONE)
    print("Start to show depth point cloud, press 'ctrl+c' to exit")
    
    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue
            
            scale = depth_frame.get_depth_scale()
            point_cloud_filter.set_position_data_scaled(scale)
            
            # 항상 depth만 사용하므로 포인트 포맷은 OBFormat.POINT로 설정
            point_cloud_filter.set_create_point_format(OBFormat.POINT)
            
            # depth frame 만으로 포인트 클라우드 계산
            point_cloud_frame = point_cloud_filter.process(frames)
            points = point_cloud_filter.calculate(point_cloud_frame)
            update_visualizer(vis, np.array(points))
    except Exception as e:
        print(e)
    finally:
        pipeline.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()