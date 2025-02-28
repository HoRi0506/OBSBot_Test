# OBSBot_Test

## OSC
- OSC를 활용해 Tiny 2를 작동

<br>

  - library install
    ```py
    # osc library install
    pip install python-osc
    # reference: https://pypi.org/project/python-osc/
    
    # keyboard event library install
    pip install keyboard
    ```

<br>

- OBSBOT App(OBSBOT Center)에서 설정 <br>
1. App을 실행한 후 톱니바퀴 모양(설정)을 클릭
2. 목록에서 OSC를 선택 
3. 연결방식을 UDP Server로 설정
4. host를 127.0.0.1로 설정
5. 수신 포트 확인
6. App에서 '>'를 클릭
7. osc를 클릭(osc로 컨트롤하기 위해서는 해당 기능을 켜줘야 함)
   
<br>

- 설정 창에서 host와 수신 포트를 아래와 같이 작성하여 사용
    ```py
    from pythonosc.udp_client improt SimpleUDPClient

    ip = "127.0.0.1"
    port = {port_number}

    client = SimpleUDPClient(ip, port)
    ```

<br>

- 아래와 같이 정해진 규칙에 따라 명령어를 보내는 것으로 Tiny 2를 컨트롤
    ```py
    client.send_message({rule}, {number})
    # reference: https://www.obsbot.co.kr/kr/explore/obsbot-center/osc
    ```

<br>

---

## VISCA
- VISCA를 활용해 Tiny 2를 작동

<br>

- Tiny 2는 VISCA에서 지원되지 않음.

<br>

---

## ultralytics YOLO
- 라이브러리 설치
  ```py
    # Install the ultralytics package from PyPI
    pip install ultralytics
    # reference: https://docs.ultralytics.com/ko/modes/track/#why-choose-ultralytics-yolo-for-object-tracking
  ```

<br>

- 테스트 코드
  ```py
    import cv2

    from ultralytics import YOLO

    # Load the YOLO11 model
    # model을 선택할 수 있음(human-tracking은 yolo11n-pose.pt)
    model = YOLO("yolo11n.pt")

    # Open the video file
    # 비디오를 선택하거나 카메라를 선택할 수 있음(경로를 지정해주거나, video_path 대신 장치 관리자의 순서대로 카메라를 지정할 수 있음(0, 1 등))
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        
        # 생성되는 cv2의 frame 크기를 조정할 수 있음
        # reframe = cv2.resize(frame, None, fx=3.2, fy=3.2, interpolation=cv2.INTER_AREA)

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            # 아래와 같이 tracker와 출력 결과를 보지 않게 만들 수 있음
            # results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            results = model.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
  ```

<br>

- ultralytic YOLO & OBSBOT OSC

  <br>

  <b>TASK</b>

  - 카메라를 OBSBOT의 Tiny 2를 사용하고 기본적인 작동은 OSC를 활용
  - ultralytic YOLO의 human model detection을 활용해서 BBox를 그림
    - 프로그램 시작 시 카메라 초기 세팅 설정 (`done`)
      - 프로그램 실행 및 종료 시 로직 수정 (`done`)
    - 카메라 움직임 수정 (fixing)
    - 추론 빈도 낮추기 (`done`)
    - 중심점을 맞추기 위해서 motor를 작동시킬 때, 어느 축이 중심에서 더 먼지 판단하여 더 먼 쪽부터 motor 이동 (`done`)
    - 사람 객체의 스켈레톤의 중심점을 frame 단위로 수집. EMA 방식으로 최근 프레임에 가중치를 두어 motor를 작동 (ing)
    - BBox의 confidence 값이 0.6 이상인 데이터만 show 되도록 수정 (`done`)
    - 두 개의 ArUco marker를 통과했을 때, sub window로 출력 (fixing)
      - ArUco marker 통과 후 객체의 스켈레톤 중 손목 부분을 확대 촬영 (to do)
    - 라우터를 통한 멀티 카메라 출력 (to do)
    - etc...