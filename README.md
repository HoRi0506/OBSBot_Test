# OBSBOT Tracking System

이 프로젝트는 OBSBOT Tiny 2 카메라와 YOLO 객체 추적 기술을 결합하여 고급 카메라 추적 시스템을 구현합니다.

<br>

<h2>📚 설치 및 종속성</h2>

<details>
<summary>setup</summary>

### 필요 라이브러리

```bash
# OSC 라이브러리 설치
pip install python-osc

# Ultralytics YOLO 설치
pip install ultralytics

# OpenCV 설치
pip install opencv-python
```

### 필요 하드웨어
- OBSBOT Tiny 2 카메라
- USB 연결이 가능한 컴퓨터

</details>

<br>

<h2>🔄 OSC 설정 방법</h2>

<details>
<summary>OBSBOT setup</summary>

### OBSBOT Center 앱 설정
1. OBSBOT Center 앱 실행
2. 톱니바퀴 모양(설정) 클릭
3. 목록에서 OSC 선택
4. 연결방식을 UDP Server로 설정
5. host를 `127.0.0.1`로 설정
6. 수신 포트 번호 확인
7. 앱에서 '>' 클릭 후 OSC 활성화

### 코드에서 OSC 클라이언트 설정

```python
from pythonosc.udp_client import SimpleUDPClient

ip = "127.0.0.1"
port = 16284  # OBSBOT Center에서 확인한 포트 번호

client = SimpleUDPClient(ip, port)
```

### 명령어 전송 예시

```python
# 짐벌 리셋
client.send_message("/OBSBOT/WebCam/General/ResetGimbal", 0)

# 카메라 깨우기
client.send_message("/OBSBOT/WebCam/General/WakeSleep", 1)

# 줌 최소화
client.send_message("/OBSBOT/WebCam/General/SetZoomMin", 0)

# 자동 화이트밸런스 설정
client.send_message("/OBSBOT/WebCam/General/SetAutoWhiteBalance", 1)

# AI 모드 비활성화
client.send_message("/OBSBOT/WebCam/Tiny/SetAiMode", 0)
```

reference: [OBSBOT OSC 명령어 참조](https://www.obsbot.co.kr/kr/explore/obsbot-center/osc)

</details>

<br>

<h2>🤖 Ultralytics YOLO</h2>

<details>
<summary>Ultralytics YOLO demo setup</summary>

### 기본 사용법

```python
import cv2
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("yolo11n-pose.pt")  # 포즈 추정 모델

# 비디오 캡처 설정
cap = cv2.VideoCapture(1)  # 카메라 인덱스에 맞게 조정

# 비디오 프레임 처리 루프
while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        # YOLO 추적 실행
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # 결과 시각화
        annotated_frame = results[0].plot()
        
        # 화면에 표시
        cv2.imshow("YOLO Tracking", annotated_frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
```

### 주요 기능
- 객체 감지 및 추적
- 포즈 추정
- 실시간 처리
- 다양한 추적 알고리즘 지원

</details>

<br>

<h2>📁 테스트 코드</h2>

<details>

<summary>ultralytic_test_obj.py</summary>

### 주요 기능
- OBSBOT Tiny 2 카메라 제어 및 초기화
- YOLO 포즈 추정 모델을 사용한 인체 추적
- 스켈레톤 중심점 기반 카메라 모터 제어
- EMA(지수 이동 평균) 기반 추적 안정화
- ArUco 마커 감지 및 게이트 통과 이벤트 처리

### 주요 구성 요소
1. **리소스 관리**
   - 프로그램 시작/종료 시 장치 초기화 및 정리
   - OBSBOT Center 앱 실행 및 연결 관리

2. **객체 추적 알고리즘**
   - ByteTrack 알고리즘 활용
   - 스켈레톤 완전성 검사 (어깨, 엉덩이 등 주요 관절 확인)
   - 객체 중심 계산 및 추적

3. **모터 제어 로직**
   - 오차 크기에 따른 동적 모터 속도 조절
   - X축/Y축 우선순위 기반 이동 명령
   - 마진 범위 내 정지 명령 처리

4. **시각화 및 디버깅**
   - 스켈레톤 및 바운딩 박스 표시
   - 추적 상태 및 오차 정보 화면 표시
   - 모터 상태 및 속도 시각화

### 주요 매개변수
- `margin_offset_x`, `margin_offset_y`: 중심 마진 오프셋 (5픽셀)
- `motor_speed_factor`: 기본 모터 속도 계수 (0.5)
- `min_motor_speed_factor`, `max_motor_speed_factor`: 모터 속도 범위 (0.3-0.8)
- `alpha`: EMA 가중치 (0.8)
- `large_error_threshold`: 큰 오차 임계값 (100픽셀)
- `speed_adjust_threshold`: 속도 조절 임계값 (200픽셀)

</details>

<details>
<summary>ultralytic_human_tracking.py</summary>

### 주요 기능
- 인체 추적에 특화된 YOLO 구현
- 사람 객체 식별 및 ID 할당
- 추적 지속성 유지

### 주요 구성 요소
1. **객체 감지 및 필터링**
   - 사람 클래스 필터링
   - 신뢰도 기반 결과 필터링

2. **추적 알고리즘**
   - 프레임 간 ID 유지
   - 객체 이동 예측

3. **시각화**
   - 추적 결과 시각화
   - ID 및 신뢰도 표시

</details>

<br>

<h2>🎯 프로젝트 진행 상황</h2>

<details>
<summary>asd</summary>

### 완료된 작업 ✅
- [X] 프로그램 시작 시 카메라 초기 세팅 설정
- [X] 프로그램 실행 및 종료 시 로직 개선
- [X] 추론 빈도 최적화
- [X] 중심에서 더 먼 축 우선 모터 이동 구현
- [X] BBox의 confidence 값 0.6 이상 필터링

### 진행 중인 작업 🔄
- 카메라 움직임 개선
- EMA 방식으로 최근 프레임에 가중치를 두어 모터 작동
- 두 개의 ArUco 마커 통과 감지 및 서브 윈도우 출력

### 예정된 작업 📋
- ArUco 마커 통과 후 객체 스켈레톤 중 손목 부분 확대 촬영
- 라우터를 통한 멀티 카메라 출력 구현

</details>

<br>

<h2>🔧 VISCA 지원 상태</h2>

<details>
<summary>OBSBOT VISCA Setting</summary>

현재 OBSBOT Tiny 2는 VISCA 프로토콜을 지원하지 않습니다. 대신 OSC 프로토콜을 통해 제어가 가능합니다.

</details>

<h2>📝 참고 자료</h2>

<details>
<summary>references</summary>

- [OBSBOT OSC 명령어 참조](https://www.obsbot.co.kr/kr/explore/obsbot-center/osc)
- [Ultralytics YOLO 문서](https://docs.ultralytics.com/ko/modes/track/#why-choose-ultralytics-yolo-for-object-tracking)
- [Python-OSC 라이브러리](https://pypi.org/project/python-osc/)

</details>