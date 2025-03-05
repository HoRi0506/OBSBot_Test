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

# ---------------------------
#  리소스 정리 및 종료 함수
# ---------------------------
def cleanup_resources():
    """
    프로그램 종료 전 장치, 프로세스, 창 등을 안전하게 종료하는 함수.
    OBSBOT 카메라의 각종 상태를 초기화하고, 캡처 객체와 미리보기 창을 종료합니다.
    """
    global cap, client, pg
    print("정리 작업을 수행합니다...")
    
    # OBSBOT 관련 명령 전송 (리셋 및 슬립 해제)
    if 'client' in globals() and client is not None:
        try:
            client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
            time.sleep(1)
            client.send_message("/OBSBOT/WebCam/General/WakeSleep", 0)
        except Exception as e:
            print(f"OBSBOT 명령 전송 실패: {e}")
    
    # VideoCapture 객체 해제
    if 'cap' in globals() and cap is not None:
        cap.release()
        
    # 미리보기 창 종료 (pywinauto 이용)
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
    
    # OBSBOT 관련 프로세스 종료
    if 'pg' in globals() and pg is not None:
        try:
            pg.terminate()
        except Exception as e:
            print(f"프로세스 종료 실패: {e}")
    
    cv2.destroyAllWindows()
    print('프로그램이 안전하게 종료되었습니다.')

# ---------------------------
#  사람 객체 검출 결과 시각화 함수
# ---------------------------
def custom_plot(results, valid_indices, img):
    """
    검출된 객체에 대해 bounding box, keypoint(스켈레톤) 및 연결선을 그리는 함수.
    
    Args:
        results: YOLO 모델의 검출 결과
        valid_indices: 신뢰도 임계값을 넘는 검출 결과의 인덱스 리스트
        img: 원본 이미지
        
    Returns:
        주석이 그려진 이미지.
    """
    annotated = img.copy()
    boxes = results[0].boxes
    keypoints = results[0].keypoints

    # 스켈레톤 연결선 정보 (예: 코에서 눈, 눈에서 귀, 어깨, 팔, 다리 등)
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
    
    # 유효한 각 검출 결과에 대해 처리
    for i in valid_indices:
        # 객체의 bounding box 좌표
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 4)

        # 객체 ID와 신뢰도 텍스트 출력
        id_val = boxes.id[i].item() if boxes.id is not None else -1
        conf_val = boxes.conf[i].item() if boxes.conf is not None else 0.0
        text = f"ID:{id_val} Conf:{conf_val:.2f}"
        cv2.putText(annotated, text, (xyxy[0]+5, xyxy[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

        # 각 keypoint (스켈레톤의 관절) 표시
        kp = keypoints.xy[i].cpu().numpy().astype(int)
        for point in kp:
            if point[0] != 0 or point[1] != 0:
                cv2.circle(annotated, (point[0], point[1]), 3, (0, 255, 0), 3)

        # 스켈레톤 연결선 그리기
        for conn in skeleton:
            pt1_idx, pt2_idx = conn
            pt1 = kp[pt1_idx]
            pt2 = kp[pt2_idx]
            if (pt1[0] != 0 or pt1[1] != 0) and (pt2[0] != 0 or pt2[1] != 0):
                cv2.line(annotated, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 4)
    return annotated

# ---------------------------
#  프로세스 종료 도우미 함수
# ---------------------------
def kill_process_by_name(name):
    """
    윈도우에서 주어진 이름을 포함하는 프로세스를 강제 종료하는 함수.
    """
    try:
        subprocess.call(f'taskkill /f /fi "IMAGENAME eq *{name}*"', shell=True)
    except Exception as ex:
        print(f"프로세스 종료 중 오류 발생: {ex}")

# ---------------------------
#  OBSBOT 앱 및 장치 초기화 함수
# ---------------------------
def start_obsbots_app():
    """
    OBSBOT 앱을 실행하고 정상 실행될 때까지 대기하는 함수.
    OBSBOT 앱이 정상 실행되면 프로세스 핸들을 반환.
    """
    print("OBSBOT App 실행 중...")
    while True:
        try:
            pg = subprocess.Popen(obsbot_center_apppath, cwd=obsbot_center_working_dir)
            time.sleep(2)
            print("프로그램 실행 성공!")
            return pg
        except Exception as e:
            print(f"프로그램 실행 중 오류 발생: {e}")
            kill_process_by_name("OBSBOT")
            time.sleep(2)

def initialize_device():
    """
    OBSBOT 장치를 초기화하기 위한 초기 명령들을 전송하는 함수.
    """
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

# ---------------------------
#  카메라 초기화 함수
# ---------------------------
def initialize_camera(scale_factor=3):
    """
    cv2.VideoCapture를 이용해 카메라를 초기화하고 해상도 및 FPS를 설정하는 함수.
    
    Args:
        scale_factor: 기본 해상도에 곱할 배수
        
    Returns:
        초기화된 VideoCapture 객체.
    """
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    default_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    default_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    new_width = int(default_width * scale_factor)
    new_height = int(default_height * scale_factor)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

# ---------------------------
#  객체 중심 계산 함수
# ---------------------------
def get_object_center(results, valid_indices):
    """
    검출된 객체들 중 신뢰도 임계값을 넘는 결과들 중
    최소 id를 가지는 객체의 스켈레톤 또는 bbox 중심 좌표를 계산.
    
    Returns:
        center_x, center_y, bbox 좌표 (x1, y1, x2, y2)
    """
    ids = results[0].boxes.id
    valid_ids = [ids[i].item() for i in valid_indices]
    min_id = min(valid_ids)
    min_idx = valid_ids.index(min_id)
    # bbox 좌표
    x1, y1, x2, y2 = [round(float(v), 4) for v in results[0].boxes.xyxy[valid_indices[min_idx]]]
    
    # 스켈레톤 포인트 사용 (5, 6, 11, 12)
    keyp = results[0].keypoints.xy[valid_indices[min_idx]].cpu().numpy().astype(int)
    s5 = keyp[5]
    s6 = keyp[6]
    s11 = keyp[11]
    s12 = keyp[12]
    
    # 네 개의 포인트가 유효하면 스켈레톤 중심, 아니면 bbox 중심 사용
    if is_valid_pt(s5) and is_valid_pt(s6) and is_valid_pt(s11) and is_valid_pt(s12):
        center_x = np.mean([s5[0], s6[0], s11[0], s12[0]])
        center_y = np.mean([s5[1], s6[1], s11[1], s12[1]])
    else:
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
    return center_x, center_y, x1, y1, x2, y2

# ---------------------------
#  EMA 스무딩 적용 함수
# ---------------------------
def apply_ema_smoothing(center_x, center_y, center_wx, center_wy, ema_x, ema_y, alpha=0.75, error_threshold=100):
    """
    현재 객체 중심 좌표에 대해 지수 이동평균(EMA)을 적용하여 평활화.
    만약 객체 중심과 윈도우 중심의 오차가 error_threshold 이상이면
    평활화 대신 현재 좌표를 바로 사용.
    
    Returns:
        업데이트된 ema_x, ema_y 값.
    """
    if ema_x is None or ema_y is None:
        ema_x = center_x
        ema_y = center_y
    else:
        if abs(center_x - center_wx) > error_threshold:
            ema_x = center_x
        else:
            ema_x = alpha * center_x + (1 - alpha) * ema_x
        if abs(center_y - center_wy) > error_threshold:
            ema_y = center_y
        else:
            ema_y = alpha * center_y + (1 - alpha) * ema_y
    return ema_x, ema_y

# ---------------------------
#  모터 이동 명령 처리 함수
# ---------------------------
def process_motor_movement(client, error_x, error_y, x_in_range, y_in_range,
                           current_time, last_command_time_x, last_command_time_y,
                           motor_speed_factor, command_delay):
    """
    x, y축의 오차를 비교하여 우선 오차가 큰 축의 모터를 먼저 움직이고,
    해당 축이 보정되면 다른 축의 이동 명령을 전송.
    
    Returns:
        업데이트된 last_command_time_x, last_command_time_y,
        그리고 해당 프레임에서 모터 명령을 보냈는지 여부 (move_sent_x, move_sent_y).
    """
    move_sent_x = False
    move_sent_y = False
    abs_error_x = abs(error_x)
    abs_error_y = abs(error_y)
    
    # y축 오차가 더 크거나 같으면 y축 우선
    if abs_error_y >= abs_error_x:
        if not y_in_range and (current_time - last_command_time_y) >= command_delay:
            half_dist_y = abs_error_y / 2.0
            move_amount_y = half_dist_y * motor_speed_factor
            if error_y > 0:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", move_amount_y)
            else:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", move_amount_y)
            move_sent_y = True
            last_command_time_y = current_time
        # y축이 보정되었으면 x축 이동
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
        # x축 오차가 더 크면 x축 우선
        if not x_in_range and (current_time - last_command_time_x) >= command_delay:
            half_dist_x = abs_error_x / 2.0
            move_amount_x = half_dist_x * motor_speed_factor
            if error_x > 0:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", move_amount_x)
            else:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", move_amount_x)
            move_sent_x = True
            last_command_time_x = current_time
        # x축 보정 후 y축 이동
        elif x_in_range and not y_in_range and (current_time - last_command_time_y) >= command_delay:
            half_dist_y = abs_error_y / 2.0
            move_amount_y = half_dist_y * motor_speed_factor
            if error_y > 0:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", move_amount_y)
            else:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", move_amount_y)
            move_sent_y = True
            last_command_time_y = current_time
            
    return last_command_time_x, last_command_time_y, move_sent_x, move_sent_y

# ---------------------------
#  모터 정지 명령 전송 함수
# ---------------------------
def send_stop_commands(client, current_time, last_command_time_x, last_command_time_y, x_in_range, y_in_range, command_delay):
    """
    오차가 margin 범위 내에 있을 경우 모터 정지 명령을 전송.
    
    Returns:
        업데이트된 last_command_time_x, last_command_time_y.
    """
    if (current_time - last_command_time_y) >= command_delay:
        if y_in_range:
            client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", 0)
            client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", 0)
            last_command_time_y = current_time
    if (current_time - last_command_time_x) >= command_delay:
        if x_in_range:
            client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", 0)
            client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", 0)
            last_command_time_x = current_time
    return last_command_time_x, last_command_time_y

# ---------------------------
#  ArUco 마커 검출 처리 함수
# ---------------------------
def process_aruco_markers(annotated_frame, reframe, marker_positions, aruco_dict_4x4, aruco_dict_original, aruco_params, aruco_detection_count, aruco_detection_threshold):
    """
    ArUco 마커를 검출하여 검출된 마커의 위치와 ID를 업데이트하고,
    주석 이미지를 반환하는 함수.
    
    Returns:
        업데이트된 annotated_frame, aruco_detection_count, marker_positions.
    """
    # 먼저 원본 사전으로 검출, 없으면 4x4 사전으로 검출
    detector_original = cv2.aruco.ArucoDetector(aruco_dict_original, aruco_params)
    detector_4x4 = cv2.aruco.ArucoDetector(aruco_dict_4x4, aruco_params)
    aruco_corners, aruco_ids, _ = detector_original.detectMarkers(reframe)
    if aruco_ids is None:
        aruco_corners, aruco_ids, _ = detector_4x4.detectMarkers(reframe)
    
    if aruco_ids is not None:
        aruco_detection_count += 1
        if aruco_detection_count >= aruco_detection_threshold:
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
                # 마커 ID가 0 또는 1이면 해당 위치 저장
                if marker_id in [0, 1]:
                    marker_positions[marker_id] = (marker_center_x, marker_center_y)
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 4)
                cv2.putText(annotated_frame, f"ID:{marker_id}", (x_min, y_min - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        aruco_detection_count = 0
    
    return annotated_frame, aruco_detection_count, marker_positions

# ---------------------------
#  게이트 통과 처리 함수 (ArUco 마커 기반)
# ---------------------------
def process_gate(annotated_frame, marker_positions, results, valid_indices,
                 person_in_gate, gate_frame, show_gate_frame, gate_frame_time, gate_display_duration):
    """
    두 개의 ArUco 마커(0,1)를 이용해 게이트 중앙을 계산하고,
    사람 객체의 중심과의 거리를 비교하여 게이트 통과 여부를 판단하는 함수.
    
    Returns:
        업데이트된 annotated_frame, person_in_gate, gate_frame, show_gate_frame, gate_frame_time.
    """
    if marker_positions[0] is not None and marker_positions[1] is not None:
        # 게이트 선과 중앙 계산
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
            
            # 마커 간 거리와 사람-게이트 중앙 간 거리 계산
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
    return annotated_frame, person_in_gate, gate_frame, show_gate_frame, gate_frame_time

# ---------------------------
#  유효 좌표 판별 함수
# ---------------------------
def is_valid_pt(pt):
    """
    좌표 pt가 (0,0)이 아니면 유효한 좌표로 판단.
    """
    return (pt[0] != 0 or pt[1] != 0)

# ============================================================
# 메인 설정 및 초기화
# ============================================================
# YOLO 모델 초기화
model = YOLO("yolo11n-pose.pt")

# ArUco 관련 사전 및 파라미터 설정
aruco_dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_dict_original = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 10
aruco_params.adaptiveThreshConstant = 7

# OBSBOT 관련 설정 (앱 경로, 작업 디렉토리, 통신 IP/Port)
obsbot_center_apppath = r"C:\Program Files\OBSBOT Center\bin\OBSBOT_Main.exe"
obsbot_center_working_dir = r"C:\Program Files\OBSBOT Center\bin"
ip = "127.0.0.1"
port = 16284
client = SimpleUDPClient(ip, port)

# OBSBOT 앱 실행
pg = start_obsbots_app()

# 장치 초기화 (모터, 줌, 화이트밸런스 등)
initialize_device()

print("비디오 실행")
# 카메라 초기화
cap = initialize_camera(scale_factor=3)

# -------------------------------------------
# 초기 파라미터 및 변수 설정
# -------------------------------------------
command_delay = 0.5         # 모터 명령 간 최소 딜레이 시간 (초)
margin_offset_x = 5         # x축 오차 허용 범위
margin_offset_y = 5         # y축 오차 허용 범위
motor_speed_factor = 0.5    # 모터 이동 속도 계수

aruco_detected = False
aruco_detection_count = 0
aruco_detection_threshold = 1
marker_positions = {0: None, 1: None}
person_in_gate = False
gate_frame = None
show_gate_frame = False
gate_frame_time = 0
gate_display_duration = 5   # 게이트 통과 시 게이트 창 표시 시간

skip_frame = 2              # 프레임 스킵 (성능 조절)
frame_counter = 0
last_results = None

# EMA(지수 이동평균) 변수 (객체 중심 좌표 평활화)
ema_obj_center_x = None
ema_obj_center_y = None

detection_timeout = 1.5       # 검출 timeout 시간 (초)
last_valid_detection_time = time.time()
reset_sent = False
just_reset = False

# 모터 명령 간 지연 체크용 시간 초기화
last_command_time_x = time.time()
last_command_time_y = time.time()

# 같은 프레임 내에 중복 stop 명령을 피하기 위해, 명령 전송 여부 플래그
move_sent_x = False
move_sent_y = False

# ============================================================
# 메인 루프 (프레임별 처리)
# ============================================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        # 프레임 읽기 실패 시, OBSBOT 상태 리셋 후 종료
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

    # 원본 프레임 복사 (후에 주석 및 표시 용)
    reframe = frame.copy()

    # 프레임 스킵을 적용하여 YOLO 추적 결과 갱신
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
    # 신뢰도가 0.6 이상인 검출 결과만 선택
    valid_indices = [i for i, conf in enumerate(confs) if conf.item() >= 0.6]
    current_time = time.time()

    # -------------------------------
    # 1) 객체 미검출 시 Reset 처리
    # -------------------------------
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

    # 검출이 재시작되면 EMA 변수 초기화
    if just_reset:
        ema_obj_center_x = None
        ema_obj_center_y = None
        last_command_time_x = current_time - command_delay
        last_command_time_y = current_time - command_delay
        just_reset = False

    # -------------------------------
    # 2) 사람 객체 중심 좌표 계산
    # -------------------------------
    try:
        center_x, center_y, x1, y1, x2, y2 = get_object_center(results, valid_indices)
    except Exception as e:
        cv2.imshow("YOLO11 Tracking", reframe)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # 윈도우 크기 및 중앙 좌표 계산
    frame_h, frame_w = reframe.shape[:2]
    center_wx = frame_w / 2.0
    center_wy = frame_h / 2.0

    # -------------------------------
    # 3) EMA 스무딩 적용 (노이즈 억제 및 빠른 추적)
    # -------------------------------
    alpha = 0.75         # EMA 가중치
    error_threshold = 100  # 큰 오차인 경우 EMA 적용 없이 현재 좌표 사용
    ema_obj_center_x, ema_obj_center_y = apply_ema_smoothing(center_x, center_y, center_wx, center_wy,
                                                               ema_obj_center_x, ema_obj_center_y,
                                                               alpha, error_threshold)
    avg_obj_center_x = ema_obj_center_x
    avg_obj_center_y = ema_obj_center_y

    # 윈도우 중앙 시각화 (십자선)
    cv2.line(reframe, (int(center_wx), 0), (int(center_wx), frame_h), (0, 255, 255), 2)
    cv2.line(reframe, (0, int(center_wy)), (frame_w, int(center_wy)), (0, 255, 255), 2)

    # margin 범위 (허용 오차)
    obj_min_y = avg_obj_center_y - margin_offset_y
    obj_max_y = avg_obj_center_y + margin_offset_y
    obj_min_x = avg_obj_center_x - margin_offset_x
    obj_max_x = avg_obj_center_x + margin_offset_x

    y_in_range = (obj_min_y <= center_wy <= obj_max_y)
    x_in_range = (obj_min_x <= center_wx <= obj_max_x)

    # x, y축 오차 계산 (객체 중심과 윈도우 중앙 간 차이)
    error_y = avg_obj_center_y - center_wy
    error_x = avg_obj_center_x - center_wx
    abs_error_x = abs(error_x)
    abs_error_y = abs(error_y)

    # -------------------------------
    # 4) 모터 이동 명령 전송 (우선순위: 오차 큰 축 우선)
    # -------------------------------
    move_sent_x = False
    move_sent_y = False
    last_command_time_x, last_command_time_y, move_sent_x, move_sent_y = process_motor_movement(
        client, error_x, error_y, x_in_range, y_in_range,
        current_time, last_command_time_x, last_command_time_y,
        motor_speed_factor, command_delay)

    # -------------------------------
    # 5) 정지 명령 전송 (해당 축 오차가 허용범위 내이면)
    # -------------------------------
    last_command_time_x, last_command_time_y = send_stop_commands(
        client, current_time, last_command_time_x, last_command_time_y, x_in_range, y_in_range, command_delay)

    # -------------------------------
    # 6) ArUco 마커 검출 처리
    # -------------------------------
    annotated_frame = custom_plot(results, valid_indices, reframe)
    annotated_frame, aruco_detection_count, marker_positions = process_aruco_markers(
        annotated_frame, reframe, marker_positions, aruco_dict_4x4, aruco_dict_original, aruco_params,
        aruco_detection_count, aruco_detection_threshold)

    # -------------------------------
    # 7) 게이트 통과(ArUco 마커 기반) 처리
    # -------------------------------
    annotated_frame, person_in_gate, gate_frame, show_gate_frame, gate_frame_time = process_gate(
        annotated_frame, marker_positions, results, valid_indices,
        person_in_gate, gate_frame, show_gate_frame, gate_frame_time, gate_display_duration)

    # 게이트 통과 시 별도 창에 표시
    if show_gate_frame:
        if time.time() - gate_frame_time < gate_display_duration:
            cv2.imshow("Gate Passage", gate_frame)
        else:
            show_gate_frame = False
            try:
                cv2.destroyWindow("Gate Passage")
            except:
                pass

    # -------------------------------
    # 8) 최종 결과 이미지 출력 및 종료 키 처리
    # -------------------------------
    cv2.imshow("YOLO11 Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("종료 키가 감지되었습니다. 프로그램을 종료합니다...")
        break

    time.sleep(0.005)

# ============================================================
# 종료 전 리소스 정리
# ============================================================
cleanup_resources()
cap.release()
cv2.destroyAllWindows()