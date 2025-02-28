import cv2
import subprocess
import time
import collections
import numpy as np
from pywinauto import Desktop
from pythonosc.udp_client import SimpleUDPClient
from ultralytics import YOLO
import atexit
import psutil
import os

def cleanup_resources():
    global cap, client, pg
    print("정리 작업을 수행합니다...")
    
    if 'client' in globals() and client is not None:
        try:
            client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
            time.sleep(1)
            client.send_message("/OBSBOT/WebCam/General/WakeSleep", 0)
        except Exception as e:
            print(f"OBSBOT 명령 전송 실패: {e}")
    
    if 'cap' in globals() and cap is not None:
        cap.release()
        
    try:
        def window_exists(title_pattern):
            try:
                windows = Desktop(backend="uia").windows()
                for w in windows:
                    if any(pattern in w.window_text() for pattern in ["Preview", "미리보기", "OBSBOT_M"]):
                        return True
                return False
            except:
                return False
        
        if window_exists(["Preview", "미리보기", "OBSBOT_M"]):
            preview_win = Desktop(backend="uia").window(title_re=".*(Preview|미리보기|OBSBOT_M).*")
            preview_win.close()
            print("미리보기 창을 닫았습니다.")
        else:
            print("미리보기 창이 이미 닫혀 있습니다.")
    except Exception as e:
        print(f"미리보기 창 처리 중 오류: {e}")
    
    if 'pg' in globals() and pg is not None:
        try:
            pg.terminate()
        except Exception as e:
            print(f"프로세스 종료 실패: {e}")
    
    cv2.destroyAllWindows()
    print('프로그램이 안전하게 종료되었습니다.')

def custom_plot(results, valid_indices, img):
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
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 4)

        id_val = boxes.id[i].item() if boxes.id is not None else -1
        conf_val = boxes.conf[i].item() if boxes.conf is not None else 0.0
        text = f"ID:{id_val} Conf:{conf_val:.2f}"
        cv2.putText(annotated, text, (xyxy[0]+5, xyxy[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

        kp = keypoints.xy[i].cpu().numpy().astype(int)
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

# ---------------------------
#       메인 설정 시작
# ---------------------------
model = YOLO("yolo11n-pose.pt")

aruco_dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_dict_original = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
aruco_params = cv2.aruco.DetectorParameters()
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
client.send_message("/OBSBOT/WebCam/General/SetZoomMin", 0)
time.sleep(1)
client.send_message("/OBSBOT/WebCam/General/SetAutoWhiteBalance", 1)
time.sleep(1)
client.send_message("/OBSBOT/WebCam/Tiny/SetAiMode", 0)
print("비디오 실행")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
default_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
default_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
scale_factor = 3
new_width = int(default_width * scale_factor)
new_height = int(default_height * scale_factor)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 30)

command_delay = 0.2
margin_offset_x = 5
margin_offset_y = 5
motor_speed_factor = 0.5

aruco_detected = False
aruco_detection_count = 0
aruco_detection_threshold = 1
marker_positions = {0: None, 1: None}
person_in_gate = False
gate_frame = None
show_gate_frame = False
gate_frame_time = 0
gate_display_duration = 5

skip_frame = 2
frame_counter = 0
last_results = None

# EMA를 위한 변수
ema_obj_center_x = None
ema_obj_center_y = None

detection_timeout = 1.5
last_valid_detection_time = time.time()
reset_sent = False
just_reset = False

# motor 명령 간 지연 체크용 시간
last_command_time_x = time.time()
last_command_time_y = time.time()

# 같은 프레임에서 바로 stop을 보내지 않기 위해,
# move를 보냈는지 여부를 임시로 저장
move_sent_x = False
move_sent_y = False

def is_valid_pt(pt):
    return (pt[0] != 0 or pt[1] != 0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
        time.sleep(0.3)
        client.send_message("/OBSBOT/WebCam/General/WakeSleep", 0)
        try:
            preview_win = Desktop(backend="uia").window(title_re=".*(Preview|미리보기|OBSBOT_M).*")
            preview_win.close()
        except:
            pass
        pg.terminate()
        print('프로그램을 종료합니다.')
        break

    reframe = frame.copy()

    if frame_counter % skip_frame == 0:
        results = model.track(reframe, persist=True, tracker="bytetrack.yaml", verbose=False)
        last_results = results
    else:
        if last_results is None:
            results = model.track(reframe, persist=True, tracker="bytetrack.yaml", verbose=False)
            last_results = results
        else:
            results = last_results
    frame_counter += 1

    confs = results[0].boxes.conf
    valid_indices = [i for i, conf in enumerate(confs) if conf.item() >= 0.6]
    current_time = time.time()

    # 1) 객체 없으면 Reset
    if len(valid_indices) == 0 or results[0].boxes.id is None:
        if current_time - last_valid_detection_time < detection_timeout:
            cv2.imshow("YOLO11 Tracking", reframe)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        if not reset_sent:
            client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
            time.sleep(0.3)
            client.send_message("/OBSBOT/WebCam/General/SetGimMotorDegree", [40, 0, -20])
            reset_sent = True
            just_reset = True
        cv2.imshow("YOLO11 Tracking", reframe)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue
    else:
        last_valid_detection_time = current_time
        reset_sent = False

    if just_reset:
        # EMA 변수 초기화
        ema_obj_center_x = None
        ema_obj_center_y = None
        last_command_time_x = current_time - command_delay
        last_command_time_y = current_time - command_delay
        just_reset = False

    ids = results[0].boxes.id
    valid_ids = [ids[i].item() for i in valid_indices]
    try:
        min_id = min(valid_ids)
        min_idx = valid_ids.index(min_id)
        x1, y1, x2, y2 = [round(float(v), 4) for v in results[0].boxes.xyxy[valid_indices[min_idx]]]
    except:
        cv2.imshow("YOLO11 Tracking", reframe)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # 스켈레톤 5,6,11,12 -> 4점, 아니면 BBox 사용
    keyp = results[0].keypoints.xy[ valid_indices[min_idx] ].cpu().numpy().astype(int)
    s5 = keyp[5]
    s6 = keyp[6]
    s11 = keyp[11]
    s12 = keyp[12]

    if is_valid_pt(s5) and is_valid_pt(s6) and is_valid_pt(s11) and is_valid_pt(s12):
        arr = np.array([s5, s6, s11, s12], dtype=float)
        center_x = np.mean(arr[:, 0])
        center_y = np.mean(arr[:, 1])
    else:
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

    # 윈도우 크기와 중앙 계산
    frame_h, frame_w = reframe.shape[:2]
    center_wx = frame_w / 2.0
    center_wy = frame_h / 2.0

    # EMA 기반 스무딩 적용 (α=0.75, 오차가 100 이상이면 바로 현재 좌표 사용)
    alpha = 0.75
    error_threshold = 100
    if ema_obj_center_x is None or ema_obj_center_y is None:
        ema_obj_center_x = center_x
        ema_obj_center_y = center_y
    else:
        if abs(center_x - center_wx) > error_threshold:
            ema_obj_center_x = center_x
        else:
            ema_obj_center_x = alpha * center_x + (1 - alpha) * ema_obj_center_x
        if abs(center_y - center_wy) > error_threshold:
            ema_obj_center_y = center_y
        else:
            ema_obj_center_y = alpha * center_y + (1 - alpha) * ema_obj_center_y

    avg_obj_center_x = ema_obj_center_x
    avg_obj_center_y = ema_obj_center_y

    # 윈도우 중앙 시각화(십자선)
    cv2.line(reframe, (int(center_wx), 0), (int(center_wx), frame_h), (0, 255, 255), 2)
    cv2.line(reframe, (0, int(center_wy)), (frame_w, int(center_wy)), (0, 255, 255), 2)

    # margin 범위
    obj_min_y = avg_obj_center_y - margin_offset_y
    obj_max_y = avg_obj_center_y + margin_offset_y
    obj_min_x = avg_obj_center_x - margin_offset_x
    obj_max_x = avg_obj_center_x + margin_offset_x

    y_in_range = (obj_min_y <= center_wy <= obj_max_y)
    x_in_range = (obj_min_x <= center_wx <= obj_max_x)

    # 오차 계산
    error_y = avg_obj_center_y - center_wy
    error_x = avg_obj_center_x - center_wx
    abs_error_x = abs(error_x)
    abs_error_y = abs(error_y)

    # -------------------------------------------
    # 우선순위에 따라 한 축씩 이동시키기
    # -------------------------------------------
    move_sent_x = False
    move_sent_y = False

    if abs_error_y >= abs_error_x:
        # y축 오차가 더 크므로 먼저 y축 이동
        if not y_in_range and (current_time - last_command_time_y) >= command_delay:
            half_dist_y = abs_error_y / 2.0
            move_amount_y = half_dist_y * motor_speed_factor
            if error_y > 0:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", move_amount_y)
            else:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", move_amount_y)
            move_sent_y = True
            last_command_time_y = current_time
        # y축이 보정된 후, x축 이동
        elif y_in_range and not x_in_range and (current_time - last_command_time_x) >= command_delay:
            half_dist_x = abs_error_x / 2.0
            move_amount_x = half_dist_x * motor_speed_factor
            if error_x > 0:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", move_amount_x)
            else:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", move_amount_x)
            move_sent_x = True
            last_command_time_x = current_time
    else:
        # x축 오차가 더 크므로 먼저 x축 이동
        if not x_in_range and (current_time - last_command_time_x) >= command_delay:
            half_dist_x = abs_error_x / 2.0
            move_amount_x = half_dist_x * motor_speed_factor
            if error_x > 0:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", move_amount_x)
            else:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", move_amount_x)
            move_sent_x = True
            last_command_time_x = current_time
        # x축이 보정된 후, y축 이동
        elif x_in_range and not y_in_range and (current_time - last_command_time_y) >= command_delay:
            half_dist_y = abs_error_y / 2.0
            move_amount_y = half_dist_y * motor_speed_factor
            if error_y > 0:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", move_amount_y)
            else:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", move_amount_y)
            move_sent_y = True
            last_command_time_y = current_time

    # -------------------------------------------
    # Motor 정지 명령: 이동하지 않은 축에 대해 보내기
    # -------------------------------------------
    if (current_time - last_command_time_y) >= command_delay:
        if y_in_range and (not move_sent_y):
            client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", 0)
            client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", 0)
            last_command_time_y = current_time

    if (current_time - last_command_time_x) >= command_delay:
        if x_in_range and (not move_sent_x):
            client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", 0)
            client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", 0)
            last_command_time_x = current_time

    # -------------------------------------------
    # 시각화 & ArUco 로직
    # -------------------------------------------
    annotated_frame = custom_plot(results, valid_indices, reframe)
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
                corners = corner.reshape((4, 2)).astype(int)
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
