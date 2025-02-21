import cv2
import subprocess
import time
import collections
import numpy as np
from pywinauto import Desktop
from pythonosc.udp_client import SimpleUDPClient
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n-pose.pt")
# model = YOLO("yolo11n.pt")

obsbot_center_apppath = r"C:\Program Files\OBSBOT Center\bin\OBSBOT_Main.exe"
obsbot_center_working_dir = r"C:\Program Files\OBSBOT Center\bin"

ip = "127.0.0.1"
port = 16284

client = SimpleUDPClient(ip, port)

# OBSBOT Center 실행
print("OBSBOT App 실행 중...")
pg = subprocess.Popen(obsbot_center_apppath, cwd=obsbot_center_working_dir)
time.sleep(2)

# OBSBOT Device setting
print("장치 초기화 중...")
client.send_message("/OBSBOT/WebCam/General/WakeSleep", 1)     # OBSBOT 깨우기
time.sleep(2)
client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)   # OBSBOT motor 원위치
time.sleep(2)
client.send_message("/OBSBOT/WebCam/General/SetZoom", 0)       # OBSBOT zoom 0배
time.sleep(2)
client.send_message("/OBSBOT/WebCam/Tiny/SetTrackingMode", 1)  # OBSBOT standard mode
time.sleep(2)
client.send_message("/OBSBOT/WebCam/Tiny/SetAiMode", 0)        # OBSBOT no tracking mode
time.sleep(2)
client.send_message("/OBSBOT/WebCam/General/SetView", 0)       # OBSBOT set view 86
print("비디오 실행")

cap = cv2.VideoCapture(1)

# --- 파라미터 설정 ---
deadzone_ratio = 0.2      # 객체 bounding box 크기의 20% 이내면 중앙으로 간주
command_delay = 0.4       # 명령 전송 최소 간격 (초)
# 객체의 중앙값의 margin 오프셋 (필요에 따라 조정)
margin_offset_x = 20      # 좌우 오프셋 (픽셀 단위)
margin_offset_y = 40      # 상하 오프셋 (픽셀 단위)

# 추론 빈도 감소 설정: 전체 프레임 중 1/skip_frame만 추론
skip_frame = 4
frame_counter = 0
last_results = None      # 마지막 추론 결과 저장

# 객체 중심 좌표를 위한 큐 (최근 10프레임)
center_x_queue = collections.deque(maxlen=10)
center_y_queue = collections.deque(maxlen=10)

# 각 축의 모터 상태를 관리 ("moving" 또는 "stopped")
state_x = "stopped"
state_y = "stopped"
last_command_time_x = time.time()
last_command_time_y = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        client.send_message("/OBSBOT/WebCam/General/WakeSleep", 0)
        break

    # 프레임 리사이즈
    reframe = cv2.resize(frame, None, fx=3.2, fy=3.2, interpolation=cv2.INTER_AREA)
    
    # --- 추론 빈도 감소 ---
    if frame_counter % skip_frame == 0:
        results = model.track(reframe, persist=True, tracker="bytetrack.yaml", verbose=False)
        last_results = results
    else:
        results = last_results if last_results is not None else model.track(reframe, persist=True, tracker="bytetrack.yaml", verbose=False)
        last_results = results
    frame_counter += 1
    # print(f"results: {results}\n")
    # break

    # --- detection 결과 처리 ---
    xy = results[0].boxes.xyxy      # 각 객체의 [x1, y1, x2, y2]
    ids = results[0].boxes.id       # 각 객체의 id
    confs = results[0].boxes.conf   # 각 객체의 confidence
    
    # print(f"results.boxes: {results[0].boxes}\n")
    # print(f"results.keypoints: {results[0].keypoints}\n")
    # break

    # confs가 0.6 미만인 detection은 제거
    valid_mask = confs >= 0.6
    if valid_mask.sum() > 0:
        results[0].boxes.data = results[0].boxes.data[valid_mask]
        
        # print(f"results.keypoints: {results[0].keypoints}\n")
        # print(f"results.keypoints: {results[0].keypoints.data}\n")
        # print(f"results.keypoints: {results[0].keypoints.xy[0]}\n")
        # print(f"results.keypoints: {results[0].keypoints.xy[0][0]}\n")
        # break
        xy = results[0].boxes.xyxy
        ids = results[0].boxes.id
        confs = results[0].boxes.conf
        
        
        left_eyes_y = results[0].keypoints.xy[0][1][1]
        right_eyes_y = results[0].keypoints.xy[0][2][1]
        # left_shoulder_x = results[0].keypoints.xy[0][5][0]
        left_shoulder_y = results[0].keypoints.xy[0][5][1]
        # right_shoulder_x = results[0].keypoints.xy[0][6][0]
        right_shoulder_y = results[0].keypoints.xy[0][6][1]
        
        # print(f"left shoulder x: {left_shoulder_x}\n")
        # print(f"left shoulder y: {left_shoulder_y}\n")
        # print(f"right shoulder_x: {right_shoulder_x}\n")
        # print(f"right shoulder_y: {right_shoulder_y}\n")
        # break
        
        try:
            # 여러 detection 중 id가 가장 낮은 객체 선택
            min_idx = int(ids.argmin().item())
            x1, y1, x2, y2 = [round(float(v), 4) for v in xy[min_idx]]
        except Exception as e:
            client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
            continue

        # 객체 중심 좌표 계산
        obj_center_x = (x1 + x2) / 2.0
        obj_center_y = (y1 + y2) / 2.0

        # 동적 deadzone 계산: bounding box 크기의 일정 비율
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        dynamic_deadzone_x = deadzone_ratio * bbox_width
        dynamic_deadzone_y = deadzone_ratio * bbox_height

        # --- 객체 중심 좌표 큐 (최근 프레임 평균) ---
        center_x_queue.append(obj_center_x)
        center_y_queue.append(obj_center_y)
        avg_obj_center_x = np.mean(center_x_queue)
        avg_obj_center_y = np.mean(center_y_queue)

        # 프레임 중앙 좌표 계산
        frame_height, frame_width = reframe.shape[:2]
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0

        # **객체의 중앙 margin 좌표 계산**  
        # (객체의 중심에 margin offset을 적용한 값)
        target_x = avg_obj_center_x + margin_offset_x
        target_y = avg_obj_center_y + margin_offset_y

        # 프레임 중앙과의 오차 계산 (각 축)
        error_x = target_x - frame_center_x
        error_y = target_y - frame_center_y

        current_time = time.time()
        # --- 수평 (좌우) 제어: 수평 정렬이 완전히 이루어질 때까지 ---
        if current_time - last_command_time_x > command_delay:
            if abs(error_x) > dynamic_deadzone_x:
                move_amount_x = abs(error_x)  # 오차 크기만큼 이동 (비례 제어)
                if error_x > 0:
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", (move_amount_x / 3.2) / 2.0)
                else:
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", (move_amount_x / 3.2) / 2.0)
                state_x = "moving"
                # 수평 정렬 중에는 수직 움직임은 중단
                if state_y == "moving":
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", 0)
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", 0)
                    state_y = "stopped"
            else:
                # 수평 오차가 허용 범위 내면 수평 모터 정지
                if state_x == "moving":
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", 0)
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", 0)
                    state_x = "stopped"
            last_command_time_x = current_time

        # --- 수직 (상하) 제어: 수평 정렬이 완료된 경우에만 ---
        if abs(error_x) <= dynamic_deadzone_x:
            if current_time - last_command_time_y > command_delay:
                if abs(error_y) > dynamic_deadzone_y:
                    move_amount_y = abs(error_y)
                    if error_y > 0:
                        client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", (move_amount_y / 3.2) / 2.0)
                    else:
                        client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", (move_amount_y / 3.2) / 2.0)
                    state_y = "moving"
                elif left_shoulder_y <= 0 or right_shoulder_y <= 0:
                    # print(f"left shoulder_y: {left_shoulder_y}\n")
                    # print(f"right shoulder_y: {right_shoulder_y}\n")
                    if left_eyes_y > 0 or right_eyes_y > 0:
                        client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", (move_amount_y / 3.2) / 2.0)
                    else:
                        client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", (move_amount_y / 3.2) / 2.0)
                    state_y = "moving"
                else:
                    if state_y == "moving":
                        client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", 0)
                        client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", 0)
                        state_y = "stopped"
                last_command_time_y = current_time
    else:
        client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)

    # 시각화: 필터링된 결과를 기반으로 박스 그림
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO11 Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
        time.sleep(2)
        client.send_message("/OBSBOT/WebCam/General/WakeSleep", 0)
        preview_win = Desktop(backend="uia").window(title_re=".*(Preview|미리보기|OBSBOT_M).*")
        preview_win.close()
        pg.terminate()
        print('프로그램을 종료합니다.')
        break

    time.sleep(0.005)

cap.release()
cv2.destroyAllWindows()