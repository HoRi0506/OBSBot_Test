import cv2
import subprocess
import time
import collections
import numpy as np
from pywinauto import Desktop
from pythonosc.udp_client import SimpleUDPClient
from ultralytics import YOLO

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

# 모델 로드
model_full = YOLO("yolo11n.pt")      # 객체(사람, refrigerator 등) 탐지 모델
model_pose = YOLO("yolo11n-pose.pt")   # 사람 포즈(17 keypoints) 탐지 모델

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
time.sleep(2)
client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
time.sleep(2)
client.send_message("/OBSBOT/WebCam/General/SetZoom", 0)
time.sleep(2)
client.send_message("/OBSBOT/WebCam/Tiny/SetTrackingMode", 1)
time.sleep(2)
client.send_message("/OBSBOT/WebCam/Tiny/SetAiMode", 0)
time.sleep(2)
client.send_message("/OBSBOT/WebCam/General/SetView", 0)
print("비디오 실행")

cap = cv2.VideoCapture(1)

deadzone_ratio = 0.2
command_delay = 0.4
margin_offset_x = 20
margin_offset_y = 40

skip_frame = 4
frame_counter = 0
last_results_pose = None
last_results_full = None

center_x_queue = collections.deque(maxlen=12)
center_y_queue = collections.deque(maxlen=12)

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

    reframe = cv2.resize(frame, None, fx=3.2, fy=3.2, interpolation=cv2.INTER_AREA)
    
    # 매 skip_frame마다 두 모델로 detection 수행
    if frame_counter % skip_frame == 0:
        results_full = model_full.track(reframe, persist=True, tracker="bytetrack.yaml", verbose=False)
        results_pose = model_pose.track(reframe, persist=True, tracker="bytetrack.yaml", verbose=False)
        last_results_full = results_full
        last_results_pose = results_pose
    else:
        results_full = last_results_full if last_results_full is not None else model_full.track(reframe, persist=True, tracker="bytetrack.yaml", verbose=False)
        results_pose = last_results_pose if last_results_pose is not None else model_pose.track(reframe, persist=True, tracker="bytetrack.yaml", verbose=False)
        last_results_full = results_full
        last_results_pose = results_pose
    frame_counter += 1

    # 포즈 모델로부터 사람 detection 처리 (confidence 0.6 이상)
    confs = results_pose[0].boxes.conf
    valid_indices = [i for i, conf in enumerate(confs) if conf.item() >= 0.6]

    current_time = time.time()
    if len(valid_indices) == 0 or results_pose[0].boxes.id is None:
        if current_time - last_valid_detection_time < detection_timeout:
            cv2.imshow("YOLO11 Tracking", reframe)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        if not reset_sent:
            client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
            reset_sent = True
        cv2.imshow("YOLO11 Tracking", reframe)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue
    else:
        last_valid_detection_time = current_time
        reset_sent = False

    # tracking을 위한 사람 detection 처리 (포즈 모델 사용)
    ids = results_pose[0].boxes.id
    valid_ids = [ids[i].item() for i in valid_indices]
    try:
        min_id = min(valid_ids)
        min_idx = valid_ids.index(min_id)
        x1, y1, x2, y2 = [round(float(v), 4) for v in results_pose[0].boxes.xyxy[ valid_indices[min_idx] ]]
    except Exception as e:
        client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)
        cv2.imshow("YOLO11 Tracking", reframe)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    left_eyes_y = results_pose[0].keypoints.xy[ valid_indices[min_idx] ][1][1]
    right_eyes_y = results_pose[0].keypoints.xy[ valid_indices[min_idx] ][2][1]
    left_shoulder_y = results_pose[0].keypoints.xy[ valid_indices[min_idx] ][5][1]
    right_shoulder_y = results_pose[0].keypoints.xy[ valid_indices[min_idx] ][6][1]
    
    obj_center_x = (x1 + x2) / 2.0
    obj_center_y = (y1 + y2) / 2.0

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    dynamic_deadzone_x = deadzone_ratio * bbox_width
    dynamic_deadzone_y = deadzone_ratio * bbox_height

    center_x_queue.append(obj_center_x)
    center_y_queue.append(obj_center_y)
    avg_obj_center_x = np.mean(center_x_queue)
    avg_obj_center_y = np.mean(center_y_queue)

    frame_height, frame_width = reframe.shape[:2]
    frame_center_x = frame_width / 2.0
    frame_center_y = frame_height / 2.0

    target_x = avg_obj_center_x + margin_offset_x
    target_y = avg_obj_center_y + margin_offset_y

    error_x = target_x - frame_center_x
    error_y = target_y - frame_center_y

    if current_time - last_command_time_x > command_delay:
        move_amount_x = abs(error_x)
        if abs(error_x) > dynamic_deadzone_x:
            if error_x > 0:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", (move_amount_x / 3.2) / 3.0)
            else:
                client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", (move_amount_x / 3.2) / 3.0)
            state_x = "moving"
            if state_y == "moving":
                client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", 0)
                client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", 0)
                state_y = "stopped"
        else:
            if state_x == "moving":
                client.send_message("/OBSBOT/WebCam/General/SetGimbalRight", 0)
                client.send_message("/OBSBOT/WebCam/General/SetGimbalLeft", 0)
                state_x = "stopped"
        last_command_time_x = current_time

    if abs(error_x) <= dynamic_deadzone_x:
        move_amount_y = abs(error_y)
        if current_time - last_command_time_y > command_delay:
            if abs(error_y) > dynamic_deadzone_y:
                if error_y > 0:
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", (move_amount_y / 3.2) / 3.0)
                else:
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", (move_amount_y / 3.2) / 3.0)
                state_y = "moving"
            elif left_shoulder_y <= 0 or right_shoulder_y <= 0:
                if left_eyes_y > 0 or right_eyes_y > 0:
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", (move_amount_y / 3.2) / 3.0)
                else:
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", (move_amount_y / 3.2) / 3.0)
                state_y = "moving"
            else:
                if state_y == "moving":
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalDown", 0)
                    client.send_message("/OBSBOT/WebCam/General/SetGimbalUp", 0)
                    state_y = "stopped"
            last_command_time_y = current_time

    annotated_frame = custom_plot(results_pose, valid_indices, reframe)

    # ----- 추가 기능: refrigerator와 사람 손목의 근접 여부 체크 -----
    trigger_fridge = False
    # 객체 탐지 결과에서 refrigerator 탐지 (confidence 0.6 이상)
    fridge_indices = []
    for i, conf in enumerate(results_full[0].boxes.conf):
        if conf.item() >= 0.6:
            cls_index = int(results_full[0].boxes.cls[i].item())
            # if results_full[0].names[cls_index] == "refrigerator":
            if results_full[0].names[cls_index] == "tv":
                fridge_indices.append(i)
    # refrigerator가 감지되었으면, 포즈 모델에서 검출한 사람의 손목(keypoints 9, 10)이 해당 refrigerator의 영역 내에 있는지 확인
    if fridge_indices:
        margin = 20  # refrigerator 영역에 약간의 여유(margin)
        for person_idx in valid_indices:
            kp = results_pose[0].keypoints.xy[person_idx].cpu().numpy().astype(int)
            wrist_left = kp[9]   # 왼쪽 손목
            wrist_right = kp[10] # 오른쪽 손목
            for fridge_idx in fridge_indices:
                fridge_box = results_full[0].boxes.xyxy[fridge_idx].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
                fridge_x1 = fridge_box[0] - margin
                fridge_y1 = fridge_box[1] - margin
                fridge_x2 = fridge_box[2] + margin
                fridge_y2 = fridge_box[3] + margin
                # 손목이 확장된 refrigerator 영역 안에 있는지 확인
                if ((fridge_x1 <= wrist_left[0] <= fridge_x2 and fridge_y1 <= wrist_left[1] <= fridge_y2) or
                    (fridge_x1 <= wrist_right[0] <= fridge_x2 and fridge_y1 <= wrist_right[1] <= fridge_y2)):
                    trigger_fridge = True
                    break
            if trigger_fridge:
                break

    # 조건 만족 시 subwindow에 프레임 표시, 아니면 subwindow 닫기
    if trigger_fridge:
        cv2.imshow("Subwindow", reframe)
    else:
        try:
            cv2.destroyWindow("Subwindow")
        except:
            pass
    # ---------------------------------------------------------

    cv2.imshow("YOLO11 Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
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

    time.sleep(0.005)

cap.release()
cv2.destroyAllWindows()

### z축 인식이 안됨