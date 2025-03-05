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

# ---------------------------
#  사람 객체 검출 결과 시각화 함수
# ---------------------------
def custom_plot(results, valid_indices, img):
    """
    사람(스켈레톤) 검출용 pose 모델 결과를 바탕으로,
    bounding box와 keypoint(스켈레톤)을 그리는 함수.
    """
    annotated = img.copy()
    boxes = results[0].boxes
    keypoints = results[0].keypoints

    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (3, 5), (5, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
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

def start_obsbots_app():
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

def initialize_camera(scale_factor=3):
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # default_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # default_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # new_width = int(default_width * scale_factor)
    # new_height = int(default_height * scale_factor)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FPS, 60)
    return cap

def get_object_center(results, valid_indices):
    """
    pose 모델 결과(스켈레톤) 중 ID가 가장 작은 사람의 중심 좌표를 구함.
    (여기서 중심은 5,6,11,12번 keypoints의 평균값)
    """
    ids = results[0].boxes.id
    valid_ids = [ids[i].item() for i in valid_indices]
    min_id = min(valid_ids)
    min_idx = valid_ids.index(min_id)

    x1, y1, x2, y2 = [round(float(v), 4) for v in results[0].boxes.xyxy[valid_indices[min_idx]]]
    keyp = results[0].keypoints.xy[valid_indices[min_idx]].cpu().numpy().astype(int)

    s5 = keyp[5]
    s6 = keyp[6]
    s11 = keyp[11]
    s12 = keyp[12]
    
    if is_valid_pt(s5) and is_valid_pt(s6) and is_valid_pt(s11) and is_valid_pt(s12):
        center_x = np.mean([s5[0], s6[0], s11[0], s12[0]])
        center_y = np.mean([s5[1], s6[1], s11[1], s12[1]])
    else:
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
    return center_x, center_y, x1, y1, x2, y2

def apply_ema_smoothing(center_x, center_y, center_wx, center_wy, ema_x, ema_y, alpha=0.75, error_threshold=100):
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

def process_motor_movement(client, error_x, error_y, x_in_range, y_in_range,
                           current_time, last_command_time_x, last_command_time_y,
                           motor_speed_factor, command_delay):
    move_sent_x = False
    move_sent_y = False
    abs_error_x = abs(error_x)
    abs_error_y = abs(error_y)
    
    # y축 오차가 더 크거나 같으면 y축 우선
    if abs_error_y >= abs_error_x:
        if not y_in_range and (current_time - last_command_time_y) >= command_delay:
            move_amount_y = abs_error_y * motor_speed_factor
            if error_y > 0:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", move_amount_y)
            else:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", move_amount_y)
            move_sent_y = True
            last_command_time_y = current_time
        elif y_in_range and not x_in_range and (current_time - last_command_time_x) >= command_delay:
            move_amount_x = abs_error_x * motor_speed_factor
            if error_x > 0:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", move_amount_x)
            else:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", move_amount_x)
            move_sent_x = True
            last_command_time_x = current_time
    else:
        # x축 오차가 더 큰 경우 x축 우선
        if not x_in_range and (current_time - last_command_time_x) >= command_delay:
            move_amount_x = abs_error_x * motor_speed_factor
            if error_x > 0:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", move_amount_x)
            else:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", move_amount_x)
            move_sent_x = True
            last_command_time_x = current_time
        elif x_in_range and not y_in_range and (current_time - last_command_time_y) >= command_delay:
            move_amount_y = abs_error_y * motor_speed_factor
            if error_y > 0:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", move_amount_y)
            else:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", move_amount_y)
            move_sent_y = True
            last_command_time_y = current_time
            
    return last_command_time_x, last_command_time_y, move_sent_x, move_sent_y

def send_stop_commands(client, current_time, last_command_time_x, last_command_time_y, x_in_range, y_in_range, command_delay):
    if (current_time - last_command_time_y) >= command_delay and y_in_range:
        client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", 0)
        client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", 0)
        last_command_time_y = current_time
    
    if (current_time - last_command_time_x) >= command_delay and x_in_range:
        client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", 0)
        client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", 0)
        last_command_time_x = current_time
    return last_command_time_x, last_command_time_y

def is_valid_pt(pt):
    return (pt[0] != 0 or pt[1] != 0)

# ============================================================
# 메인 설정 및 초기화
# ============================================================
model_pose = YOLO("yolo11n-pose.pt")  # 사람 스켈레톤 용
model_object = YOLO("yolo11n.pt")    # 사람=0, 사물=1~

obsbot_center_apppath = r"C:\Program Files\OBSBOT Center\bin\OBSBOT_Main.exe"
obsbot_center_working_dir = r"C:\Program Files\OBSBOT Center\bin"
ip = "127.0.0.1"
port = 16284
client = SimpleUDPClient(ip, port)

pg = start_obsbots_app()
initialize_device()

cap = initialize_camera(scale_factor=3)

command_delay = 0.5
margin_offset_x = 5
margin_offset_y = 5
motor_speed_factor = 0.5

skip_frame = 2
frame_counter = 0
last_results = None

ema_obj_center_x = None
ema_obj_center_y = None

detection_timeout = 1.5
last_valid_detection_time = time.time()
reset_sent = False
just_reset = False

last_command_time_x = time.time()
last_command_time_y = time.time()

# zoom 및 wrist 모드 관련 변수
zoomed_in = False
zoom_window_name = "Zoomed Hand"
zoomed_hand_window = False
subwindow_open = False
wrist_mode = False          # wrist 모드 활성화 여부
target_wrist = None         # 선택된 손목 좌표 (예: (x,y))
wrist_error_threshold = 20  # 손목 좌표가 윈도우 중심과 얼마나 가까워야 zoom할지 (픽셀 단위)
wrist_radius = 50           # object detection 판별용

# ============================================================
# 메인 루프
# ============================================================
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

    # --- 사람(스켈레톤) 추적 ---
    if frame_counter % skip_frame == 0:
        results_pose = model_pose.track(reframe, persist=True, tracker="bytetrack.yaml", verbose=False)
        last_results = results_pose
    else:
        if last_results is None:
            results_pose = model_pose.track(reframe, persist=True, tracker="bytetrack.yaml", verbose=False)
            last_results = results_pose
        else:
            results_pose = last_results
    frame_counter += 1

    confs = results_pose[0].boxes.conf
    valid_indices = [i for i, conf in enumerate(confs) if conf.item() >= 0.6]
    current_time = time.time()

    # 사람 미검출 시 일정 시간 후 Reset
    if len(valid_indices) == 0 or results_pose[0].boxes.id is None:
        if current_time - last_valid_detection_time < detection_timeout:
            cv2.imshow("YOLO11 Tracking", reframe)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        if not reset_sent:
            client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
            time.sleep(0.5)
            client.send_message("/OBSBOT/WebCam/General/SetGimMotorDegree", [40, 0, -20])
            reset_sent = True
            just_reset = True
        cv2.imshow("YOLO11 Tracking", reframe)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        continue
    else:
        last_valid_detection_time = current_time
        reset_sent = False

    if just_reset:
        ema_obj_center_x = None
        ema_obj_center_y = None
        last_command_time_x = current_time - command_delay
        last_command_time_y = current_time - command_delay
        just_reset = False
        wrist_mode = False
        target_wrist = None

    # 사람 중심 좌표 (기본: 5,6,11,12번 중심)
    try:
        center_x, center_y, x1, y1, x2, y2 = get_object_center(results_pose, valid_indices)
    except:
        cv2.imshow("YOLO11 Tracking", reframe)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        continue

    frame_h, frame_w = reframe.shape[:2]
    center_wx = frame_w / 2.0
    center_wy = frame_h / 2.0

    # EMA 스무딩 및 error 계산
    # 기본적으로 error는 사람 중심(5,6,11,12) 기준
    avg_obj_center_x, avg_obj_center_y = center_x, center_y
    if wrist_mode:
        # wrist_mode가 활성화되면 매 프레임 감지된 손목 좌표로 target_wrist 업데이트
        if object_in_left_hand:
            target_wrist = left_wrist
        elif object_in_right_hand:
            target_wrist = right_wrist
        error_x = target_wrist[0] - center_wx
        error_y = target_wrist[1] - center_wy
    else:
        error_x = avg_obj_center_x - center_wx
        error_y = avg_obj_center_y - center_wy

    cv2.line(reframe, (int(center_wx), 0), (int(center_wx), frame_h), (0, 255, 255), 2)
    cv2.line(reframe, (0, int(center_wy)), (frame_w, int(center_wy)), (0, 255, 255), 2)

    # 모터 이동
    last_command_time_x, last_command_time_y, move_sent_x, move_sent_y = process_motor_movement(
        client, error_x, error_y, (abs(error_x) < margin_offset_x), (abs(error_y) < margin_offset_y),
        current_time, last_command_time_x, last_command_time_y,
        motor_speed_factor, command_delay
    )

    # 모터 정지
    last_command_time_x, last_command_time_y = send_stop_commands(
        client, current_time, last_command_time_x, last_command_time_y,
        (abs(error_x) < margin_offset_x), (abs(error_y) < margin_offset_y), command_delay
    )

    annotated_frame = custom_plot(results_pose, valid_indices, reframe)

    # 객체(사물) 검출 → reframe 사용
    obj_results = model_object.predict(reframe, conf=0.5, iou=0.45, verbose=False)
    object_boxes = obj_results[0].boxes

    # 사람(클래스0)은 제외하고 사물만 표시
    for box in object_boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0:
            continue
        bx1, by1, bx2, by2 = box.xyxy[0]
        conf_val = float(box.conf[0])
        cv2.rectangle(annotated_frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{model_object.names[cls_id]} {conf_val:.2f}",
                    (int(bx1), int(by1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # 손목 좌표 추출
    kp = results_pose[0].keypoints.xy[valid_indices[0]].cpu().numpy().astype(int)
    left_wrist = kp[9]
    right_wrist = kp[10]

    # 사물이 손목 내부에 있는지 확인 (BBox 내부 여부)
    object_in_left_hand = False
    object_in_right_hand = False

    if is_valid_pt(left_wrist):
        lx, ly = left_wrist
        for box in object_boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:
                continue
            bx1, by1, bx2, by2 = box.xyxy[0]
            if (lx >= bx1 and lx <= bx2) and (ly >= by1 and ly <= by2):
                object_in_left_hand = True
                break

    if is_valid_pt(right_wrist):
        rx, ry = right_wrist
        for box in object_boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:
                continue
            bx1, by1, bx2, by2 = box.xyxy[0]
            if (rx >= bx1 and rx <= bx2) and (ry >= by1 and ry <= by2):
                object_in_right_hand = True
                break

    # 키 입력 처리: Space, ESC, q
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("종료 키가 감지되었습니다. 프로그램을 종료합니다...")
        break
    # ESC: subwindow 닫고 zoom 해제, wrist_mode 해제 → 기본 tracking 복귀
    elif key == 27:
        if subwindow_open:
            cv2.destroyWindow(zoom_window_name)
            subwindow_open = False
        if zoomed_in:
            client.send_message("/OBSBOT/WebCam/General/SetZoom", 0)
            zoomed_in = False
        if wrist_mode:
            wrist_mode = False
            target_wrist = None
    # Space: wrist mode 전환 (기존 중심을 손목 좌표로 변경)
    elif key == ord(' '):
        if not wrist_mode:
            if object_in_left_hand or object_in_right_hand:
                target_wrist = left_wrist if object_in_left_hand else right_wrist
                wrist_mode = True
                print("[INFO] Wrist mode 활성화: target_wrist =", target_wrist)
            else:
                print("[INFO] 손목 근처에 사물이 없어 wrist mode 전환하지 않습니다.")

    # 만약 wrist_mode가 활성화되어 있고, motor가 target_wrist에 도달했으면 zoom 실행
    if wrist_mode and (not zoomed_in) and (abs(target_wrist[0] - center_wx) < wrist_error_threshold) and (abs(target_wrist[1] - center_wy) < wrist_error_threshold):
        # Zoom 실행: target_wrist를 중심으로 crop 후 subwindow 표시
        client.send_message("/OBSBOT/WebCam/General/SetZoom", 100)
        zoomed_in = True
        time.sleep(0.5)
        for _ in range(50):
            ret_zoom, zoom_frame=cap.read()

        if ret_zoom:
            # 여기서 "확대한 화면에서 손목 부분을 Crop"하여 subwindow에 표시
            crop_size=200  # 원하는 만큼 크기 조절
            cx1=max(0, target_wrist[0]-crop_size)
            cy1=max(0, target_wrist[1]-crop_size)
            cx2=min(frame_w, target_wrist[0]+crop_size)
            cy2=min(frame_h, target_wrist[1]+crop_size)
            cropped_zoom=zoom_frame[cy1:cy2, cx1:cx2].copy()
            
            re_cropped_zoom = cv2.resize(cropped_zoom, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_AREA)

            cv2.imshow(zoom_window_name, re_cropped_zoom)
            subwindow_open=True
            print("[INFO] Zoom된 프레임에서 손목 부근 Crop -> subwindow 표시")
        else:
            print("[ERROR] Zoom된 프레임을 읽지 못했습니다.")

    cv2.imshow("YOLO11 Tracking", annotated_frame)
    time.sleep(0.005)

cleanup_resources()
cap.release()
cv2.destroyAllWindows()