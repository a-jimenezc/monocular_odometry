"""
    implementing robot control interface
    also run hardware check here.
"""

import json
import serial
import queue
import threading
import glob
import time
import logging

"""
        Config loading
"""


class ReadLine:
    def __init__(self, s):
        """
        From waveshare's example code.
"""
        self.buf = bytearray()
        self.s = s
        self.sensor_data = []
        self.sensor_list = []
        try:
            self.sensor_data_ser = serial.Serial(
                glob.glob('/dev/ttyUSB*')[0], 115200)
            print("/dev/ttyUSB* connected succeed")
        except:
            self.sensor_data_ser = None
            self.sensor_data_max_len = 51

        try:
            self.lidar_ser = serial.Serial(
                glob.glob('/dev/ttyACM*')[0], 230400, timeout=1)
            print("/dev/ttyACM* connected succeed")
        except:
            self.lidar_ser = None
        self.ANGLE_PER_FRAME = 12
        self.HEADER = 0x54
        self.lidar_angles = []
        self.lidar_distances = []
        self.lidar_angles_show = []
        self.lidar_distances_show = []
        self.last_start_angle = 0

    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(512, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)

    def clear_buffer(self):
        self.s.reset_input_buffer()

    def read_sensor_data(self):
        if self.sensor_data_ser == None:
            return

        try:
            buffer_clear = False
            while self.sensor_data_ser.in_waiting > 0:
                buffer_clear = True
                sensor_readline = self.sensor_data_ser.readline()
                if len(sensor_readline) <= self.sensor_data_max_len:
                    self.sensor_list.append(
                        sensor_readline.decode('utf-8')[:-2])
                else:
                    self.sensor_list.append(sensor_readline.decode(
                        'utf-8')[:self.sensor_data_max_len])
                    self.sensor_list.append(sensor_readline.decode(
                        'utf-8')[self.sensor_data_max_len:-2])
            if buffer_clear:
                self.sensor_data = self.sensor_list.copy()
                self.sensor_list.clear()
                self.sensor_data_ser.reset_input_buffer()
        except Exception as e:
            print(f"[base_ctrl.read_sensor_data] error: {e}")


class ChassisControl:

    def __init__(self):

        # init serial information transmittion (uart)
        self.BAUD_RATE = 115200
        self.DATA_MAX_SIZE = 512
        self.WHEEL_DIAMETER = 75 # milimeter or 80 milimeter from specs
        self.WHEEL_WIDTH = 42.5 #

        self.ser = serial.Serial(
            port="/dev/ttyAMA0",
            baudrate=self.BAUD_RATE,
            timeout=1)
        self.rl = ReadLine(self.ser)

        # self check
        assert self.ser.is_open

        # potential memory leak and deadlock here
        self._msg_queue = queue.Queue(maxsize=1024)
        self.msg_thread = threading.Thread(target=self.msg_proc,daemon=True) # daemon = True
        self.msg_thread.start()
        print("Chassis Control initialed")

    def on_data_received(self):
        self.ser.reset_input_buffer()
        data_read = json.loads(self.rl.readline().decode('utf-8'))
        return data_read

    def msg_proc(self):
        print("msg queue started")
        while True:
            data = self._msg_queue.get()  # blocked here
            print(f"[{time.time()}] From msg queue:",data)
            self.ser.write((json.dumps(data) + '\n').encode("utf-8"))

    def movement(self,speed,X,Y):
        """
            X,Y,speed should all be 0-1
        """
        left_speed = speed * Y - speed * X
        right_speed = speed * Y + speed * X
        print(left_speed,right_speed)
        self._msg_queue.put({"T":1, "L":left_speed, "R":right_speed})
        #time.sleep(0.5)
        return (left_speed,right_speed)
        #time.sleep(0.1) # await for thread execute before exit.


    def handle_msg(self,data):

        resp = {}
        data = json.loads(data)
        if data.get('movement'):
            movement = data.get('movement')
            speed = movement.get('speed')
            x = movement.get('X')
            y = movement.get('Y')
            self.movement(speed,x,y)
            return self.generate_feedback()
        elif data.get('raw'):
            instruction = data.get('raw')
            self._msg_queue.put(instruction)
            ...


    def generate_feedback(self):
        try:
            while True:
                try:
                    data_recv_buffer = json.loads(self.rl.readline().decode('utf-8'))
                    if 'T' in data_recv_buffer:
                        if data_recv_buffer['T'] == 1001:
                            print(data_recv_buffer)
                            break
                except:
                    ...
            return json.dumps(data_recv_buffer)
            #return  {"time":time.time(),"data":data_recv_buffer}
        except Exception as e:
            logging.error(f"[Chassis][Feedback]:{e}")

    def generate_feedback_stream(self):
        try:
            while True:
                try:
                    data = None
                    data_recv_buffer = json.loads(self.rl.readline().decode('utf-8'))
                    if 'T' in data_recv_buffer:
                        data = {"time":time.time(),data:data_recv_buffer}
                        yield(
                            json.dumps(data)
                        )
                except KeyboardInterrupt as e:
                    logging.info(f"[Chassis][Feedback]: Exit from KeyboardInterrupt")
        except Exception as e:
            logging.error(f"[Chassis][Feedback]:{e}")

        logging.info(f"[Chassis][Feedback]: Exit")


if __name__ == "__main__":

    cc = ChassisControl()
    time.sleep(1)
    #cc.movement(0.2, -0.5, 0.5)
    #cc.movement(0.0, -0.5, 0.5)
    exit()