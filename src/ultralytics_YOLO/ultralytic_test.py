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

obsbot_center_apppath = r"C:\Program Files\OBSBOT Center\bin\OBSBOT_Main.exe"
obsbot_center_working_dir = r"C:\Program Files\OBSBOT Center\bin"

ip = "127.0.0.1"
port = 16284

client = SimpleUDPClient(ip, port)
pg = subprocess.Popen(obsbot_center_apppath, cwd=obsbot_center_working_dir)
client.send_message("/OBSBOT/WebCam/General/WakeSleep", 1)
client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
client.send_message("/OBSBOT/WebCam/General/SetZoom", 0)
print("비디오 실행")

cap = cv2.VideoCapture(1)

# Tracking 설정 값
deadzone = 150           # 이 범위 내에서는 움직임 무시 (픽셀)
base_move_step = 30      # 기본 이동량
hysteresis_margin = 40   # 오차 변화가 이 값보다 작으면 새로운 명령을 내리지 않음

# 에러 평활화를 위한 버퍼 (최근 프레임의 오차를 기록)
error_x_buffer = collections.deque(maxlen=40)
error_y_buffer = collections.deque(maxlen=40)

# 이전 평균 오차값
prev_avg_error_x = 0
prev_avg_error_y = 0

# 명령 전송 간격
last_command_time = time.time()
command_delay = 0.2

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 프레임 리사이즈
    reframe = cv2.resize(frame, None, fx=3.2, fy=3.2, interpolation=cv2.INTER_AREA)

    # YOLO11 tracking 수행
    results = model.track(reframe, persist=True, tracker="bytetrack.yaml", verbose=False)
    
    # 원래 결과의 박스 정보
    xy = results[0].boxes.xyxy      # 각 객체의 [x1, y1, x2, y2]
    ids = results[0].boxes.id       # 각 객체의 id
    confs = results[0].boxes.conf   # 각 객체의 confidence

    # confs가 0.5 미만인 detection을 결과에서 제거하여 시각화 시 표시하지 않음
    valid_mask = confs >= 0.5
    if valid_mask.sum() > 0:
        # 필터링한 결과로 박스 데이터를 업데이트합니다.
        results[0].boxes.data = results[0].boxes.data[valid_mask]
        # 업데이트한 데이터를 기반으로 다시 추출
        xy = results[0].boxes.xyxy
        ids = results[0].boxes.id
        confs = results[0].boxes.conf

        try:
            # id가 가장 낮은 객체 선택
            min_idx = int(ids.argmin().item())
            x1, y1, x2, y2 = xy[min_idx]
        except Exception as e:
            client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
            continue
        
        # 객체의 중심 좌표 계산 (평균값)
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # 프레임 중앙 좌표 계산
        frame_height, frame_width = reframe.shape[:2]
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        
        # 오차 계산 (객체 중심 - 프레임 중앙)
        error_x = bbox_center_x - frame_center_x
        error_y = bbox_center_y - frame_center_y
        
        # 버퍼에 오차 저장 (평활화)
        error_x_buffer.append(float(error_x))
        error_y_buffer.append(float(error_y))
        
        avg_error_x = np.mean(error_x_buffer)
        avg_error_y = np.mean(error_y_buffer)
        
        # 이전 평균 오차와의 차이가 일정 값 이상일 때만 업데이트
        delta_x = abs(avg_error_x - prev_avg_error_x)
        delta_y = abs(avg_error_y - prev_avg_error_y)
        
        current_time = time.time()
        if current_time - last_command_time > command_delay:
            # X축 이동 처리
            if abs(avg_error_x) > deadzone and delta_x > hysteresis_margin:
                ratio_x = min(1.0, (abs(avg_error_x) - deadzone) / abs(avg_error_x))
                move_amount_x = base_move_step * ratio_x
                if avg_error_x > 0:
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", move_amount_x)
                else:
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", move_amount_x)
                prev_avg_error_x = avg_error_x
            
            # Y축 이동 처리
            if abs(avg_error_y) > deadzone and delta_y > hysteresis_margin:
                ratio_y = min(1.0, (abs(avg_error_y) - deadzone) / abs(avg_error_y))
                move_amount_y = base_move_step * ratio_y
                if avg_error_y > 0:
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", move_amount_y)
                else:
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", move_amount_y)
                prev_avg_error_y = avg_error_y
            
            last_command_time = current_time
    else:
        client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
        
    # 수정된 결과를 기반으로 시각화
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO11 Tracking", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        client.send_message("/OBSBOT/WebCam/General/SetZoom", 0)
        client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
        client.send_message("/OBSBOT/WebCam/General/WakeSleep", 0)
        preview_win = Desktop(backend="uia").window(title_re=".*(Preview|미리보기|OBSBOT_M).*")
        preview_win.close()
        pg.terminate()
        print('프로그램을 종료합니다.')
        break

cap.release()
cv2.destroyAllWindows()