import cv2
import subprocess
import time
import collections
import numpy as np
from pywinauto import Desktop
from pythonosc.udp_client import SimpleUDPClient
from ultralytics import YOLO
import atexit  # 프로그램 종료 시 정리 작업을 위한 모듈 추가
import psutil  # 프로세스 확인을 위한 모듈 추가
import os  # 파일 경로 확인을 위한 모듈 추가

# 정리 작업을 위한 함수 정의
def cleanup_resources():
    global cap, client, pg
    print("정리 작업을 수행합니다...")
        
    # 카메라 리소스 해제
    if 'cap' in globals() and cap is not None:
        cap.release()
        
    # 미리보기 창 닫기 - 창이 존재하는지 먼저 확인
    try:
        # 창이 존재하는지 확인하는 함수
        def window_exists(title_pattern):
            try:
                windows = Desktop(backend="uia").windows()
                for w in windows:
                    if any(pattern in w.window_text() for pattern in ["Preview", "미리보기", "OBSBOT_M"]):
                        return True
                return False
            except:
                return False
        
        # 창이 존재하는 경우에만 닫기 시도
        if window_exists(["Preview", "미리보기", "OBSBOT_M"]):
            preview_win = Desktop(backend="uia").window(title_re=".*(Preview|미리보기|OBSBOT_M).*")
            preview_win.close()
            print("미리보기 창을 닫았습니다.")
        else:
            print("미리보기 창이 이미 닫혀 있습니다.")
    except Exception as e:
        print(f"미리보기 창 처리 중 오류: {e}")
    
    # OBSBOT 명령 전송
    if 'client' in globals() and client is not None:
        try:
            client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
            time.sleep(1)
            client.send_message("/OBSBOT/WebCam/General/WakeSleep", 0)
        except Exception as e:
            print(f"OBSBOT 명령 전송 실패: {e}")
    
    # 프로세스 종료
    if 'pg' in globals() and pg is not None:
        try:
            pg.terminate()
        except Exception as e:
            print(f"프로세스 종료 실패: {e}")
    
    # OpenCV 창 닫기
    cv2.destroyAllWindows()
    print('프로그램이 안전하게 종료되었습니다.')

def custom_plot(results, valid_indices, img):
    """
    valid_indices에 해당하는 detection만 사용하여,
    - bounding box (내부 좌측 상단에 id와 conf 텍스트 포함),
    - 17개 keypoint를 원으로 표시,
    - keypoints를 연결하는 skeleton 선을 그립니다.
    """
    annotated = img.copy()
    boxes = results[0].boxes
    keypoints = results[0].keypoints

    skeleton = [
        (0, 1), (0, 2),      # 코에서 눈으로
        (1, 3), (2, 4),      # 눈에서 각 귀로
        (3, 5),              # 귀 -> 어깨 연결
        (5, 6),              # 양쪽 어깨 연결
        (5, 7), (7, 9),      # 왼쪽 어깨 -> 팔꿈치 -> 손목
        (6, 8), (8, 10),     # 오른쪽 어깨 -> 팔꿈치 -> 손목
        (5, 11), (6, 12),    # 어깨에서 엉덩이로
        (11, 12),            # 양쪽 엉덩이 연결
        (11, 13), (13, 15),  # 왼쪽 엉덩이 -> 무릎 -> 발목
        (12, 14), (14, 16)   # 오른쪽 엉덩이 -> 무릎 -> 발목
    ]
    
    for i in valid_indices:
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
        cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 4)

        id_val = boxes.id[i].item() if boxes.id is not None else -1
        conf_val = boxes.conf[i].item() if boxes.conf is not None else 0.0
        text = f"ID:{id_val} Conf:{conf_val:.2f}"
        cv2.putText(annotated, text, (xyxy[0]+5, xyxy[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

        kp = keypoints.xy[i].cpu().numpy().astype(int)  # shape: (17, 2)
        for point in kp:
            if point[0] != 0 or point[1] != 0:
                cv2.circle(annotated, (point[0], point[1]), 3, (0, 255, 0), 3)

        for conn in skeleton:
            pt1_idx, pt2_idx = conn
            pt1 = kp[pt1_idx]
            pt2 = kp[pt2_idx]
            if (pt1[0] != 0 or pt1[1] != 0) and (pt2[0] != 0 or pt2[1] != 0):
                cv2.line(annotated, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 4)
    return annotated

def kill_process_by_name(name):
    try:
        subprocess.call(f'taskkill /f /fi "IMAGENAME eq *{name}*"', shell=True)
    except Exception as ex:
        print(f"프로세스 종료 중 오류 발생: {ex}")

# Load model
model = YOLO("yolo11n-pose.pt")

# ArUco 마커 검출을 위한 dictionary, 파라미터 생성
aruco_dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_dict_original = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
aruco_params = cv2.aruco.DetectorParameters()

# 마커 검출 성능 향상을 위한 파라미터 조정
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 10
aruco_params.adaptiveThreshConstant = 7

obsbot_center_apppath = r"C:\Program Files\OBSBOT Center\bin\OBSBOT_Main.exe"
obsbot_center_working_dir = r"C:\Program Files\OBSBOT Center\bin"
ip = "127.0.0.1"
port = 16284
client = SimpleUDPClient(ip, port)

print("OBSBOT App 실행 중...")
while True:
    try:
        pg = subprocess.Popen(obsbot_center_apppath, cwd=obsbot_center_working_dir)
        time.sleep(2)
        print("프로그램 실행 성공!")
        break
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {e}")
        kill_process_by_name("OBSBOT")
        time.sleep(2)

print("장치 초기화 중...")
client.send_message("/OBSBOT/WebCam/General/WakeSleep", 1)
time.sleep(1)
client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
time.sleep(1)
# client.send_message("/OBSBOT/WebCam/General/SetZoom", 0)
client.send_message("/OBSBOT/WebCam/General/SetZoomMin", 0)
# time.sleep(1)
# client.send_message("/OBSBOT/WebCam/Tiny/SetTrackingMode", 1)
time.sleep(1)
client.send_message("/OBSBOT/WebCam/General/SetAutoWhiteBalance", 1)
time.sleep(1)
client.send_message("/OBSBOT/WebCam/Tiny/SetAiMode", 0)
print("비디오 실행")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Windows의 경우 DirectShow 사용 예시
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1792)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1344)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 30)

command_delay = 0.2
margin_offset_x = 10  # 객체 중앙 인식 범위의 x축 margin
margin_offset_y = 20  # 객체 중앙 인식 범위의 y축 margin
motor_speed_factor = 0.5

# 연속 이동에 따른 이동량 감쇠를 위한 변수들
consecutive_x_moves = 0
consecutive_y_moves = 0
last_direction_x = None
last_direction_y = None
decay_factor = 0.7  # 연속 이동 시 이동량을 줄여줄 감쇠 계수 (0 < decay_factor < 1)

# ArUco 마커 검출 관련 변수
aruco_detected = False
aruco_detection_count = 0
aruco_detection_threshold = 1  # 즉시 인식

# 마커 위치 추적을 위한 변수
marker_positions = {0: None, 1: None}  # 마커 ID를 키로 하는 위치 딕셔너리
person_in_gate = False  # 사람이 마커 사이에 있는지 여부
gate_frame = None  # 사람이 마커 사이를 지나갈 때의 프레임
show_gate_frame = False  # 서브윈도우 표시 여부
gate_frame_time = 0  # 서브윈도우 표시 시작 시간
gate_display_duration = 5  # 서브윈도우 표시 지속 시간(초)

skip_frame = 2
frame_counter = 0
last_results = None

center_x_queue = collections.deque(maxlen=10)
center_y_queue = collections.deque(maxlen=10)

# detection persistence (debounce) 시간 (초)
detection_timeout = 1.5
last_valid_detection_time = time.time()

reset_sent = False

state_x = "stopped"
state_y = "stopped"
last_command_time_x = time.time()
last_command_time_y = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
        time.sleep(2)
        client.send_message("/OBSBOT/WebCam/General/WakeSleep", 0)
        try:
            preview_win = Desktop(backend="uia").window(title_re=".*(Preview|미리보기|OBSBOT_M).*")
            preview_win.close()
        except Exception as e:
            pass
        pg.terminate()
        print('프로그램을 종료합니다.')
        break

    reframe = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

    if frame_counter % skip_frame == 0:
        results = model.track(reframe, persist=True, tracker="bytetrack.yaml", verbose=False)
        last_results = results
    else:
        results = last_results if last_results is not None else model.track(reframe, persist=True, tracker="bytetrack.yaml", verbose=False)
        last_results = results
    frame_counter += 1

    confs = results[0].boxes.conf
    valid_indices = [i for i, conf in enumerate(confs) if conf.item() >= 0.6]

    current_time = time.time()
    if len(valid_indices) == 0 or results[0].boxes.id is None:
        # 트래킹하는 사람이 없을 경우
        if current_time - last_valid_detection_time < detection_timeout:
            cv2.imshow("YOLO11 Tracking", reframe)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        if not reset_sent:
            client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
            time.sleep(0.3)
            client.send_message("/OBSBOT/WebCam/General/SetGimMotorDegree", [40, 0, -20])  # 속도, 좌우, 상하
            reset_sent = True
        cv2.imshow("YOLO11 Tracking", reframe)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue
    else:
        last_valid_detection_time = current_time
        reset_sent = False

    # 객체가 있을 때 처리
    ids = results[0].boxes.id
    valid_ids = [ids[i].item() for i in valid_indices]
    try:
        min_id = min(valid_ids)
        min_idx = valid_ids.index(min_id)
            
        x1, y1, x2, y2 = [round(float(v), 4) for v in results[0].boxes.xyxy[valid_indices[min_idx]]]
    except Exception as e:
        client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
        time.sleep(0.3)
        client.send_message("/OBSBOT/WebCam/General/SetGimMotorDegree", [40, 0, -20])
        cv2.imshow("YOLO11 Tracking", reframe)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    left_eyes_y = results[0].keypoints.xy[ valid_indices[min_idx] ][1][1]
    right_eyes_y = results[0].keypoints.xy[ valid_indices[min_idx] ][2][1]
    left_body_y = results[0].keypoints.xy[ valid_indices[min_idx] ][11][1]
    right_body_y = results[0].keypoints.xy[ valid_indices[min_idx] ][12][1]
    
    # 객체 중앙값
    obj_center_x = (x1 + x2) / 2.0
    obj_center_y = (y1 + y2) / 2.0

    # 큐에 넣어서 평균값 사용
    center_x_queue.append(obj_center_x)
    center_y_queue.append(obj_center_y)
    avg_obj_center_x = np.mean(center_x_queue)
    avg_obj_center_y = np.mean(center_y_queue)

    # 윈도우 중앙값
    frame_height, frame_width = reframe.shape[:2]
    frame_center_x = frame_width / 2.0
    frame_center_y = frame_height / 2.0

    # 객체 중앙 인식 범위
    obj_center_min_x = avg_obj_center_x - margin_offset_x
    obj_center_max_x = avg_obj_center_x + margin_offset_x
    obj_center_min_y = avg_obj_center_y - margin_offset_y
    obj_center_max_y = avg_obj_center_y + margin_offset_y

    # 윈도우 중앙이 객체 중앙 범위 안에 있는지 체크
    x_in_range = obj_center_min_x <= frame_center_x <= obj_center_max_x
    y_in_range = obj_center_min_y <= frame_center_y <= obj_center_max_y

    # 오차 계산
    error_x = avg_obj_center_x - frame_center_x
    error_y = avg_obj_center_y - frame_center_y
    abs_error_x = abs(error_x)
    abs_error_y = abs(error_y)

    # -----------------------------
    # 1) X축 vs Y축 중 더 큰 오차를 먼저 보정
    # -----------------------------
    if abs_error_x > abs_error_y:
        # 우선 X축 처리
        if state_y == "moving":
            # Y축 움직임 중이었다면 정지
            client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", 0)
            client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", 0)
            state_y = "stopped"

        if current_time - last_command_time_x > command_delay:
            if not x_in_range:
                # ---------------------------
                # (A) 연속 이동 감쇠 로직
                # ---------------------------
                direction_x = "right" if (error_x > 0) else "left"
                
                # 연속 방향 체크
                if direction_x == last_direction_x:
                    consecutive_x_moves += 1
                else:
                    consecutive_x_moves = 1
                    last_direction_x = direction_x
                
                # 이동량 = 기본계수 × 오차 × (감쇠계수 ^ (연속횟수-1))
                move_amount_x = abs_error_x * motor_speed_factor * (decay_factor ** (consecutive_x_moves - 1))
                
                # 실제 모터 명령
                if error_x > 0:  # 객체가 윈도우 중앙보다 오른쪽
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", move_amount_x)
                else:            # 객체가 윈도우 중앙보다 왼쪽
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", move_amount_x)

                state_x = "moving"
            else:
                # 범위 안이면 정지 명령
                if state_x == "moving":
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", 0)
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", 0)
                state_x = "stopped"

            last_command_time_x = current_time

    else:
        # Y축 처리
        if state_x == "moving":
            # X축 움직임 중이었다면 정지
            client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", 0)
            client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", 0)
            state_x = "stopped"

        if current_time - last_command_time_y > command_delay:
            if not y_in_range:
                # ---------------------------
                # (A) 연속 이동 감쇠 로직
                # ---------------------------
                direction_y = "down" if (error_y > 0) else "up"
                
                # 연속 방향 체크
                if direction_y == last_direction_y:
                    consecutive_y_moves += 1
                else:
                    consecutive_y_moves = 1
                    last_direction_y = direction_y

                # 이동량 = 기본계수 × 오차 × (감쇠계수 ^ (연속횟수-1))
                move_amount_y = abs_error_y * motor_speed_factor * (decay_factor ** (consecutive_y_moves - 1))

                # 실제 모터 명령
                if error_y > 0:  # 객체가 윈도우 중앙보다 아래
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", move_amount_y)
                else:            # 객체가 윈도우 중앙보다 위
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", move_amount_y)

                state_y = "moving"
            
            # 사람 객체 skeleton에서 몸체 하단이 보이지 않을 때 보정
            elif left_body_y <= 0 or right_body_y <= 0:
                # 눈이 보이긴 하는데 몸이 제대로 안 잡혔을 경우
                direction_y = "up" if (left_eyes_y > 0 or right_eyes_y > 0) else "down"
                
                # 연속 방향 체크
                if direction_y == last_direction_y:
                    consecutive_y_moves += 1
                else:
                    consecutive_y_moves = 1
                    last_direction_y = direction_y

                move_amount_y = abs_error_y * motor_speed_factor * (decay_factor ** (consecutive_y_moves - 1))

                if direction_y == "up":
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", move_amount_y)
                else:
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", move_amount_y)

                state_y = "moving"
            else:
                # 범위 안이면 정지
                if state_y == "moving":
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", 0)
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", 0)
                state_y = "stopped"

            last_command_time_y = current_time

    # 시각화
    annotated_frame = custom_plot(results, valid_indices, reframe)

    # ----- [추가 기능] ArUco Marker 인식 -----
    marker_positions = {0: None, 1: None}
    
    detector_4x4 = cv2.aruco.ArucoDetector(aruco_dict_4x4, aruco_params)
    detector_original = cv2.aruco.ArucoDetector(aruco_dict_original, aruco_params)
    
    aruco_corners, aruco_ids, _ = detector_original.detectMarkers(reframe)
    if aruco_ids is None:
        aruco_corners, aruco_ids, _ = detector_4x4.detectMarkers(reframe)
    
    if aruco_ids is not None:
        aruco_detection_count += 1
        if aruco_detection_count >= aruco_detection_threshold:
            aruco_detected = True
            annotated_frame = cv2.aruco.drawDetectedMarkers(annotated_frame, aruco_corners, aruco_ids)
            
            for i, corner in enumerate(aruco_corners):
                corners = corner.reshape((4, 2))
                corners = corners.astype(int)
                
                x_min = np.min(corners[:, 0])
                y_min = np.min(corners[:, 1])
                x_max = np.max(corners[:, 0])
                y_max = np.max(corners[:, 1])
                
                marker_center_x = (x_min + x_max) // 2
                marker_center_y = (y_min + y_max) // 2
                
                marker_id = int(aruco_ids[i][0])
                
                if marker_id in [0, 1]:
                    marker_positions[marker_id] = (marker_center_x, marker_center_y)
                
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 4)
                cv2.putText(annotated_frame, f"ID:{marker_id}", (x_min, y_min - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        aruco_detection_count = 0
        aruco_detected = False
    
    if marker_positions[0] is not None and marker_positions[1] is not None:
        cv2.line(annotated_frame, marker_positions[0], marker_positions[1], (0, 255, 0), 6)
        gate_center_x = (marker_positions[0][0] + marker_positions[1][0]) // 2
        gate_center_y = (marker_positions[0][1] + marker_positions[1][1]) // 2
        cv2.circle(annotated_frame, (gate_center_x, gate_center_y), 20, (255, 0, 255), -1)
        cv2.putText(annotated_frame, "GATE", (gate_center_x - 40, gate_center_y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        if len(valid_indices) > 0 and results[0].boxes.id is not None:
            ids = results[0].boxes.id
            valid_ids = [ids[i].item() for i in valid_indices]
            min_id = min(valid_ids)
            min_idx = valid_ids.index(min_id)
            
            x1, y1, x2, y2 = [round(float(v), 4) for v in results[0].boxes.xyxy[valid_indices[min_idx]]]
            person_center_x = (x1 + x2) / 2.0
            person_center_y = (y1 + y2) / 2.0
            
            marker_distance = np.sqrt((marker_positions[1][0] - marker_positions[0][0])**2 + 
                                      (marker_positions[1][1] - marker_positions[0][1])**2)
            
            person_to_gate_distance = np.sqrt((person_center_x - gate_center_x)**2 + 
                                              (person_center_y - gate_center_y)**2)
            
            gate_threshold = marker_distance * 0.3
            
            current_person_in_gate = person_to_gate_distance < gate_threshold
            
            if current_person_in_gate and not person_in_gate:
                gate_frame = annotated_frame.copy()
                show_gate_frame = True
                gate_frame_time = time.time()
                print("사람이 게이트를 통과했습니다!")
            
            person_in_gate = current_person_in_gate
            cv2.line(annotated_frame, 
                     (int(person_center_x), int(person_center_y)), 
                     (gate_center_x, gate_center_y), 
                     (255, 255, 0), 2)
    
    if show_gate_frame:
        if time.time() - gate_frame_time < gate_display_duration:
            cv2.imshow("Gate Passage", gate_frame)
        else:
            show_gate_frame = False
            try:
                cv2.destroyWindow("Gate Passage")
            except:
                pass
    
    cv2.imshow("YOLO11 Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("종료 키가 감지되었습니다. 프로그램을 종료합니다...")
        break

    time.sleep(0.005)

cleanup_resources()
cap.release()
cv2.destroyAllWindows()
