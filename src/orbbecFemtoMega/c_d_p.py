import cv2
import numpy as np
import open3d as o3d
import sys

# pyorbbecsdk 설치 경로 추가
sys.path.append("C:/project/pyorbbecsdk/install/lib")
from pyorbbecsdk import Pipeline, Config, OBSensorType

def create_grid(size=1.0, divisions=10):
    """
    XZ 평면에 일정한 간격의 그리드를 생성하여 LineSet 형태로 반환합니다.
    """
    points = []
    lines = []
    step = size / divisions
    # X 방향 선들
    for i in range(divisions + 1):
        x = -size/2 + i * step
        points.append([x, 0, -size/2])
        points.append([x, 0, size/2])
        lines.append([2*i, 2*i+1])
    offset = (divisions + 1) * 2
    # Z 방향 선들
    for j in range(divisions + 1):
        z = -size/2 + j * step
        points.append([-size/2, 0, z])
        points.append([size/2, 0, z])
        lines.append([offset + 2*j, offset + 2*j+1])
    colors = [[0.7, 0.7, 0.7] for _ in range(len(lines))]
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector(colors)
    return grid

def main():
    # 1. Initialize pipeline and choose stream profiles
    pipeline = Pipeline()
    config = Config()
    
    depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)\
                            .get_default_video_stream_profile()
    config.enable_stream(depth_profile)
    
    color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)\
                            .get_default_video_stream_profile()
    config.enable_stream(color_profile)
    
    pipeline.start(config)
    
    # 2. Get camera intrinsics for depth
    cam_param = pipeline.get_camera_param()
    intr = cam_param.depth_intrinsic
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.cx, intr.cy
    depth_width, depth_height = intr.width, intr.height
    
    # Prepare pixel coordinate grid
    u_coords = np.tile(np.arange(depth_width), depth_height)
    v_coords = np.repeat(np.arange(depth_height), depth_width)
    
    # 3. Setup Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Point Cloud", width=1280, height=720)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    # 그리드 추가 (바닥)
    grid = create_grid(size=2.0, divisions=20)
    vis.add_geometry(grid)
    # Set render option: adjust point size (예: 1.0)
    render_opt = vis.get_render_option()
    render_opt.point_size = 5.0  # 작게 조절
    render_opt.background_color = np.asarray([0.05, 0.05, 0.05])
    
    print("Streaming started... Press 'q' or ESC to quit.")
    
    scale = 1.0
    
    try:
        while True:
            frames = pipeline.wait_for_frames(2000)
            if frames is None:
                continue
            
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue
            
            # 4. Process color frame
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
            cv2.imshow("Color", color_image)
            
            # 5. Process depth frame
            depth_scale = depth_frame.get_depth_scale()
            dw, dh = depth_frame.get_width(), depth_frame.get_height()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(dh, dw)
            depth_mm = depth_data.astype(np.float32) * depth_scale  # depth in mm
            
            depth_norm = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            depth_colormap = cv2.flip(depth_colormap, 1)
            cv2.imshow("Depth", depth_colormap)
            
            # 6. Compute 3D points (in meters)
            depth_m = depth_mm / 1000.0  # convert to meters
            Z_raw = depth_m.flatten()
            # 사용: 원래 카메라 좌표계 : X = (u-cx)/fx*Z, Y = (v-cy)/fy*Z, Z = depth.
            # Open3D 시각화에서는 일반적으로 카메라가 원점에서 -Z를 바라보므로 변환해줍니다.
            X = (u_coords - cx) / fx * Z_raw
            Y = -(v_coords - cy) / fy * Z_raw
            Z = -Z_raw  # Z를 반전
            points = np.vstack((-X, Y, Z)).T  # shape (N,3)
            
            # 7. Filter out points with zero depth
            valid_idx = (Z_raw > 0)
            num_valid = np.count_nonzero(valid_idx)
            if num_valid == 0:
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
                continue
            points_valid = points[valid_idx, :]
            # Scale the point cloud for easier visualization
            points_valid *= scale
            
            # 8. Map corresponding colors (convert to RGB)
            depth_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
            colors_all = depth_rgb.reshape(-1, 3).astype(np.float32) / 255.0
            colors_valid = colors_all[valid_idx, :]
            
            # 9. Update Open3D point cloud
            pcd.points = o3d.utility.Vector3dVector(points_valid)
            pcd.colors = o3d.utility.Vector3dVector(colors_valid)
            
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
    finally:
        vis.destroy_window()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()