# Tiny2 모델은 visca를 지원하지 않거나 sony 측에서 지원하는 port가 차단된 것으로 보임

import time
from visca_over_ip import Camera
import keyboard

cam = Camera(ip='127.0.0.1', port=52381)

while True:
    if keyboard.is_pressed('esc'):
        print('프로그램을 종료합니다.')
        cam.set_power(False)
        cam.close_connection()
        break
    
    if keyboard.is_pressed('0'):
        print('프로그램을 시작합니다.')
        cam.set_power(True)
        time.sleep(2)