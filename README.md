# 카메라 추적 및 3D 이미징 시스템

<br>

## OBSBOT Tracking System

OBSBOT Tiny 2 카메라와 YOLO 객체 추적 기술을 결합하여 고급 카메라 추적 시스템을 구현합니다.

<br>

<h3>📚 설치 및 종속성</h3>

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

<h3>🔄 OSC 설정 방법</h3>

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

<h3>🤖 Ultralytics YOLO</h3>

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

<h3>🔧 VISCA 지원 상태</h3>

<details>
<summary>OBSBOT VISCA Setting</summary>

현재 OBSBOT Tiny 2는 VISCA 프로토콜을 지원하지 않습니다. 대신 OSC 프로토콜을 통해 제어가 가능합니다.

</details>

<br>

<h3>📁 OBSBOT 테스트 코드</h3>

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

<details>
<summary>ultralytic_hand_tracking.py</summary>

### 주요 기능
- 손 추적 및 제스처 인식
- 손목 위치 기반 확대 촬영
- 스켈레톤 포인트 분석

### 주요 구성 요소
1. **손 감지 및 추적**
   - 손목 키포인트 추출
   - 제스처 패턴 인식

2. **카메라 제어**
   - 손 위치 기반 줌 제어
   - 제스처 기반 명령 실행

</details>

<br>

## Orbbec 3D 이미징 시스템

Orbbec Femto Mega 카메라를 활용한 3D 포인트 클라우드 처리 및 깊이 이미지 분석 시스템입니다.

<br>

<h3>📚 설치 및 종속성</h3>

<details>
<summary>setup</summary>

### 필요 라이브러리

```bash
# Orbbec SDK 관련 라이브러리
pip install open3d
pip install pyorbbecsdk

# 이미지 처리용 라이브러리
pip install opencv-python
pip install numpy
```

### 필요 하드웨어
- Orbbec Femto Mega 카메라
- USB 3.0 연결이 가능한 컴퓨터

### Orbbec SDK 설치 및 설정 과정

#### 1. SDK 다운로드 및 설치
1. [OrbbecSDK_v2 GitHub](https://github.com/orbbec/OrbbecSDK_v2) 또는 [릴리즈 페이지](https://github.com/orbbec/OrbbecSDK_v2/releases)에서 Windows용 SDK 패키지 다운로드
2. 다운로드한 `.exe` 파일을 실행하여 설치 (기본 경로: `C:\Program Files\Orbbec\OrbbecSDK`)
3. 또는 `.zip` 파일을 원하는 경로에 압축 해제

#### 2. 환경 변수 설정
- Orbbec SDK의 `bin` 폴더를 시스템 PATH에 추가
  ```
  시스템 속성 > 고급 > 환경 변수 > Path에 추가:
  C:\Program Files\Orbbec\OrbbecSDK\bin
  ```
- 이는 Python에서 `.dll` 파일을 찾을 수 있도록 하기 위함

#### 3. Python SDK (pyorbbecsdk) 설치
1. **개발 환경 준비**
   - Python 3.6 이상
   - CMake 설치 ([CMake 공식 웹사이트](https://cmake.org/download/))
   - Visual Studio 설치 (Desktop development with C++ 포함)

2. **pyorbbecsdk 빌드 및 설치**
   ```bash
   # pyorbbecsdk 저장소 클론
   git clone https://github.com/orbbec/pyorbbecsdk
   cd pyorbbecsdk
   
   # 가상환경 생성 및 활성화
   python -m venv ./venv
   .\venv\Scripts\activate  # Windows PowerShell
   
   # 필요 패키지 설치
   pip install -r requirements.txt
   
   # 빌드 디렉토리 생성 및 이동
   mkdir build
   cd build
   
   # CMake 구성
   cmake -Dpybind11_DIR=$(pybind11-config --cmakedir) ..
   
   # 빌드 및 설치
   cmake --build . --config Release
   cmake --install .
   ```

3. **설치 확인**
   - 설치 후 `site-packages` 디렉토리에 `pyorbbecsdk.cp3XX-win_amd64.pyd` 파일이 생성되어야 함(혹은 유사한)
   - 가상환경의 site-packages 경로는 일반적으로 `venv/Lib/site-packages`

#### 4. 일반적인 문제 해결
1. **모듈을 찾을 수 없는 경우**
   - `.pyd` 파일이 Python의 검색 경로에 있는지 확인
   - 다음 방법 중 하나로 해결:
     - `CMAKE_INSTALL_PREFIX`를 가상환경의 site-packages로 지정하여 재빌드
     - `.pyd` 파일을 수동으로 site-packages 디렉토리에 복사
     - `PYTHONPATH` 환경변수에 `.pyd` 파일 경로 추가

2. **DLL 오류 발생 시**
   - `OrbbecSDK.dll` 또는 `depthengine_2_0.dll` 등이 PATH에 있는지 확인
   - 필요시 DLL 파일을 Python 실행 경로에 복사

#### 5. 카메라 연결
- USB 3.0 포트에 Femto Mega 카메라 연결
- 별도 드라이버 없이 Windows에서 UVC 카메라로 인식됨
- Orbbec Viewer(`tools` 폴더)로 카메라 연결 테스트 가능

</details>

<br>

<h3>📁 Orbbec 테스트 코드</h3>

<details>
<summary>c_d_p.py (컬러, 깊이, 포인트 클라우드)</summary>

### 주요 기능
- 컬러, 깊이, 포인트 클라우드 처리
- Open3D 기반 3D 시각화
- 깊이 기반 색상화

### 주요 구성 요소
1. **데이터 처리**
   - 깊이 맵 처리
   - 포인트 클라우드 생성
   - 다운샘플링 및 필터링

2. **시각화**
   - 3D 포인트 클라우드 렌더링
   - 깊이 기반 색상 매핑
   - 실시간 뷰 업데이트

</details>

<details>
<summary>color_depth.py</summary>

### 주요 기능
- 컬러 및 깊이 이미지 동시 처리
- 깊이 정보 시각화
- 컬러-깊이 정렬

### 주요 구성 요소
1. **이미지 처리**
   - 깊이 맵 컬러화
   - 컬러-깊이 이미지 동기화

2. **시각화**
   - 깊이 정보 히트맵 표시
   - 실시간 이미지 표시

</details>

<details>
<summary>orbbec_official_pointcloud.py</summary>

### 주요 기능
- Orbbec 공식 SDK 기반 포인트 클라우드 생성
- 고성능 포인트 클라우드 처리
- GPU 가속 지원

### 주요 구성 요소
1. **데이터 처리**
   - SDK 기반 포인트 클라우드 생성
   - 고성능 필터링

2. **시각화**
   - 포인트 클라우드 렌더링
   - 실시간 뷰 제어

</details>

<br>

## 프로젝트 공통 정보

<br>

<h3>📁 프로젝트 구조</h3>

<details>
<summary>디렉토리 구조</summary>

```
OBSBOT_Test/
├── src/
│   ├── orbbecFemtoMega/        # Orbbec Femto Mega 카메라 관련 코드
│   │   ├── c_d_p.py            # 컬러, 깊이, 포인트 클라우드 처리
│   │   ├── color_depth.py      # 컬러 및 깊이 이미지 처리
│   │   └── orbbec_official_pointcloud.py  # 공식 포인트 클라우드 처리
│   │
│   ├── osc/                    # OSC 프로토콜 관련 코드
│   │   └── osc_test.py         # OSC 테스트 및 기본 기능
│   │
│   ├── ultralytics_YOLO/       # YOLO 객체 감지 및 추적 관련 코드
│   │   ├── ultralytic_hand_tracking.py    # 손 추적 구현
│   │   ├── ultralytic_human_tracking.py   # 인체 추적 구현
│   │   ├── ultralytic_test_normal.py      # 기본 테스트 코드
│   │   ├── ultralytic_test_obj.py         # 객체 추적 메인 코드
│   │   └── yolo11n-pose.pt                # 포즈 추정 모델 파일
│   │
│   └── visca/                  # VISCA 프로토콜 관련 코드
│       └── visca_test.py       # VISCA 테스트 코드
```

</details>

<br>

<h3>🎯 프로젝트 진행 상황</h3>

<details>
<summary>진행 상황</summary>

### 완료된 작업 ✅
- [X] 프로그램 시작 시 카메라 초기 세팅 설정
- [X] 프로그램 실행 및 종료 시 로직 개선
- [X] 추론 빈도 최적화
- [X] 중심에서 더 먼 축 우선 모터 이동 구현
- [X] BBox의 confidence 값 0.6 이상 필터링
- [X] EMA 방식으로 최근 프레임에 가중치를 두어 모터 작동
- [X] 특수 키 입력 시 화면 확대 후 손목 부분을 crop하여 촬영
- [X] 카메라 움직임 안정성 개선

### 진행 중인 작업 🔄
- Orbbec Femto Mega, OBSBOT 카메라 통합 및 기능 통합
- Orbbec Femto Mega color, depth, pointcloud를 viewer와 유사하게 출력되도록 수정
- Orbbec Femto Mega program 실행 시 최적화 작업

### 예정된 작업 📋
- 라우터를 통한 멀티 카메라 출력 구현
- Orbbec Femto Mega color, depth, pointcloud를 C or C++로 구현

</details>

<br>

<h3>📝 참고 자료</h3>

<details>
<summary>references</summary>

- [OBSBOT OSC 명령어 참조](https://www.obsbot.co.kr/kr/explore/obsbot-center/osc)
- [Ultralytics YOLO 문서](https://docs.ultralytics.com/ko/modes/track/#why-choose-ultralytics-yolo-for-object-tracking)
- [Python-OSC 라이브러리](https://pypi.org/project/python-osc/)
- [Open3D 문서](http://www.open3d.org/docs/release/)
- [Orbbec SDK 문서](https://orbbec3d.com/index/download.html)

</details>