from pythonosc.udp_client import SimpleUDPClient
from pywinauto.application import Application
from pywinauto import Desktop
import keyboard
import time
import subprocess

obsbot_center_apppath = r"C:\Program Files\OBSBOT Center\bin\OBSBOT_Main.exe"
obsbot_center_working_dir = r"C:\Program Files\OBSBOT Center\bin"

ip = "127.0.0.1"
port = 16284

client = SimpleUDPClient(ip, port)

while True:
    if keyboard.is_pressed('esc'):
        print('프로그램을 종료합니다.')
        client.send_message("/OBSBOT/WebCam/General/WakeSleep", 0)
        # time.sleep(2)
        
        try:
            preview_win = Desktop(backend="uia").window(title_re=".*(Preview|미리보기|OBSBOT_M).*")
            preview_win.close()
            print("미리보기 창 종료 성공")
        except Exception as e:
            print("미리보기 창 종료 실패 또는 미리보기 창을 찾을 수 없음:", e)
        
        # 메인 OBSBOT 앱 종료
        pg.terminate()
        break
    
    if keyboard.is_pressed('0'):
        print('프로그램을 연결합니다.')
        pg = subprocess.Popen(obsbot_center_apppath, cwd=obsbot_center_working_dir)
        time.sleep(5)
         
        try:
            app = Application(backend='uia').connect(title_re='OBSBOT*')
            main_window = app.window(title_re='OBSBOT*')
            main_window.wait('visible', timeout=10)
            # main_window.print_control_identifiers()
            main_window.set_focus()
            try:
                preview_button = main_window.child_window(
                    auto_id="WindowBasic.wBasic.wCentralWidget.swMain.pgMain.wMainPanel.wDevTitle.wToolBar.pbVideoPreview",
                    control_type="CheckBox"
                )
                preview_button.wait('visible', timeout=10)
                time.sleep(5)
                client.send_message("/OBSBOT/WebCam/General/WakeSleep", 1)
                # preview_button.click_input()
                preview_button.click()
                print('비디오 실행')
            except Exception as e:
                print(f'button exception: {e}')
        except Exception as e:
            print(f'exception: {e}')
    
    if keyboard.is_pressed('1'):
        print('auto tracking mode: Motion를 실행합니다.')
        client.send_message("/OBSBOT/WebCam/Tiny/SetTrackingMode", 2)
        time.sleep(2)
    
    if keyboard.is_pressed('2'):
        print('auto tracking mode: Standard를 실행합니다.')
        client.send_message("/OBSBOT/WebCam/Tiny/SetTrackingMode", 1)
        time.sleep(2)
    
    if keyboard.is_pressed('3'):
        print('AI Mode: Normal Tracking Mode를 실행합니다.')
        client.send_message("/OBSBOT/WebCam/Tiny/SetAiMode", 1)
        time.sleep(2)
    
    if keyboard.is_pressed('4'):
        print('AI Mode: Upper Body Mode를 실행합니다.')
        client.send_message("/OBSBOT/WebCam/Tiny/SetAiMode", 2)
        time.sleep(2)
        
    if keyboard.is_pressed('5'):
        print('AI Mode: Close-up Mode를 실행합니다.')
        client.send_message("/OBSBOT/WebCam/Tiny/SetAiMode", 3)
        time.sleep(2)
        
    if keyboard.is_pressed('6'):
        print('AI Mode: Hand Tracking Mode를 실행합니다.')
        client.send_message("/OBSBOT/WebCam/Tiny/SetAiMode", 8)
        time.sleep(2)