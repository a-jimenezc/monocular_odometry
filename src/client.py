import requests
import json
import time
from pynput import keyboard
#{"T":138,"L":1,"R":1}
#{"T":139}
#{"T":136,"cmd":3000}

direction_keys = set()
throttle_keys = set()

speed = 0
last_update_time = 0
UPDATE_INTERVAL = 0.5

URL = "http://10.0.0.90:5000/chassis_control"

class ChassisClient:
    
    def camera_rotation(self,x_angle=0,y_angle=0):
        data = {'raw':{"T":133,"X":x_angle,"Y":y_angle,"SPD":0,"ACC":0}}
        resp = requests.post(URL,json=json.dumps(data))
        #print(resp.content)
    
    def move(self,x,y,speed):
        data = {'movement':{'speed':speed, 'X': x, 'Y': y}}
        resp = requests.post(URL,json=json.dumps(data))
        print(resp)
    
    def rotate(self,direction,t):  # just for initialization
        if direction == 'left':
            data = {'raw':{"T":1, "L":-0.1, "R":0.1}}
            for _ in range(0,t):
                requests.post(URL,json=json.dumps(data))
                time.sleep(0.5)
                
        elif direction == "right":
            data = {'raw':{"T":1, "L":0.1, "R":-0.1}}
            for _ in range(0,t):
                requests.post(URL,json=json.dumps(data))
                time.sleep(0.5)
            
        data = {'raw':{"T":1, "L":0, "R":0}}
        requests.post(URL,json=json.dumps(data))
        
    def throttle(self):
        global speed
        if 'e' in throttle_keys:
            s = speed+0.05
            speed = min(s, 1)
        elif 'q' in throttle_keys:
            s = speed-0.05
            speed = max(s, 0)
        else:
            ...
        print(f"Current Speed:{speed}")

    
    def movement(self):
        global last_update_time
        global speed
        
        data = {'speed': speed, 'X': 0, 'Y': 0}
        x = 0
        y = 0
        current_time = time.time()
        if current_time - last_update_time < UPDATE_INTERVAL:
            return
        
        if 'w' in direction_keys and 'a' in direction_keys:
            x = 1 *0.8 * speed
            y = 1 * speed 
        elif 'w' in direction_keys and 'd' in direction_keys:
            x = -1 * 0.8 * speed
            y = 1 * speed
        elif 'w' in direction_keys:
            x = 0
            y = 1
        elif 's' in direction_keys and 'a' in direction_keys:    
            x = -0.25
            y = -1 * speed
        elif 's' in direction_keys and 'd' in direction_keys:
            x = 0.25
            y = -1 * speed
        elif 's' in direction_keys:
            x = 0
            y = -1
        elif 'a' in direction_keys:
            x = 1 * speed
            y = 0
        elif 'd' in direction_keys:
            x = -1 * speed
            y = 0
        else:
            x = 0 
            y = 0
        
        data = {'movement':{'speed':speed, 'X': x, 'Y': y}}
        resp = requests.post(URL,json=json.dumps(data))
        print(resp.content)

    def on_press(self,key):
        try:
            print(key)
            if key.char in ['w', 'a', 's', 'd','']:
                direction_keys.add(key.char)
                self.movement()
            elif key.char in ('q','e'):
                throttle_keys.add(key.char)
                self.throttle()
        except AttributeError:
            print(f"pressed: {key}")

    def on_release(self,key):
        try:
            if key.char in direction_keys:
                direction_keys.remove(key.char)
                self.movement()
            elif key.char in ('q','e'):
                throttle_keys.remove(key.char)
        except AttributeError:
            print(f"released: {key}")
        
        if key == keyboard.Key.esc:  
            print("exit")
            return False


    def control_process_fake(self):

        speed = 0

        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()


if __name__ == "__main__":

    #data1 = {'movement': {'speed': 0.2, 'X': 0, 'Y': 1}}
    # for _ in range(0,20):
    #     data = {'movement': {'speed': 0.2,'X': 0,'Y':1}}
    #     resp = requests.post(url, json=json.dumps(data))
    #     time.sleep(0.5)
    #     print(resp.content)
    #     data = {'movement': {'speed': 0.1,'X': 0,'Y':-1}}
    #     resp = requests.post(url, json=json.dumps(data))
    #     time.sleep(0.5)
    #     print(resp.content)
    c = ChassisClient()
    c.control_process_fake()
    #c.rotate()
    #c.move(0,1,0.4)