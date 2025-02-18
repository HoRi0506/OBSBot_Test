# OBSBot_Test

## OSC
- OSC를 활용해 Tiny 2를 작동

<br>

- library install
```py
# osc library install
pip install python-osc
# reference: https://pypi.org/project/python-osc/
```

<br>

```py
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

- Tiny 2는 VISCA에서 지원되지 않거나 Sony 측에서 port를 차단한 것으로 보임

<br>

---

## SDK
