import cv2
import numpy as np
import sys
sys.path.append("C:/project/pyorbbecsdk/install/lib")
import pyorbbecsdk

def main():
    pipeline = pyorbbecsdk.Pipeline()
    config = pyorbbecsdk.Config()
    
    try:
        # Femto Mega의 경우 기본 컬러 스트림 프로파일 사용 (MJPG)
        color_profiles = pipeline.get_stream_profile_list(pyorbbecsdk.OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_default_video_stream_profile()
        print("color profile : {}x{}@{}_{}".format(
            color_profile.get_width(),
            color_profile.get_height(),
            color_profile.get_fps(),
            color_profile.get_format()))
        config.enable_stream(color_profile)
        
        # 뎁스 스트림 프로파일 기본값 사용 (Y16)
        depth_profiles = pipeline.get_stream_profile_list(pyorbbecsdk.OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        print("depth profile : {}x{}@{}_{}".format(
            depth_profile.get_width(),
            depth_profile.get_height(),
            depth_profile.get_fps(),
            depth_profile.get_format()))
        config.enable_stream(depth_profile)
    except Exception as e:
        print("Error enabling stream profiles:", e)
        return

    try:
        pipeline.start(config)
    except Exception as e:
        print("Error starting pipeline:", e)
        return

    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            continue

        # 컬러 프레임: MJPG 압축 데이터 디코딩 시 IMREAD_UNCHANGED 사용
        color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        color_img = cv2.imdecode(color_data, cv2.IMREAD_UNCHANGED)
        if color_img is None:
            print("Failed to decode color frame")
            continue

        # 뎁스 프레임 처리 (raw Y16 데이터)
        w_d, h_d = depth_frame.get_width(), depth_frame.get_height()
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        try:
            depth_map = depth_data.reshape((h_d, w_d))
        except Exception as ex:
            print("Depth reshape error:", ex)
            continue
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cv2.imshow("Orbbec Color", color_img)
        cv2.imshow("Orbbec Depth", depth_vis)
        if cv2.waitKey(1) == 27:  # ESC키로 종료
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
