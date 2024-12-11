# A simple flask app for camera feed
# Running on the raspberry, only for feed.
from flask import Flask, render_template, Response, request, jsonify
from picamera2 import Picamera2
from logging import Logger
import time
import cv2
import threading
from camera import Camera
from remote.chassis_control import ChassisControl

app = Flask(__name__)

cam = Camera(grayscale=False,rectification=True)
chassis = ChassisControl()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/start', methods=['GET'])
def start_camera():
    cam.start_camera()
    return jsonify({"status": "Camera started"}), 200

@app.route('/stop', methods=['GET'])
def stop_camera():
    cam.stop_camera()
    return jsonify({"status": "Camera stopped"}), 200

@app.route('/video_feed')
def video_feed():
    return Response(cam.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/frame', methods=['GET'])
def get_frame():
    frame = cam.get_latest_frame()
    if frame:
        return Response(frame, mimetype='image/jpeg')
    return jsonify({"error": "No frame available"}), 500


@app.route('/chassis_feedback')
def chassis_feedback():
    return Response(chassis.generate_feedback(),content_type='application/json')

# deprecated probabliy, the feedback from normal request is enough.
@app.route('/chassis_feedback_stream')
def chassis_feedback_stream():
    return Response(chassis.generate_feedback_stream(),content_type='text/event-stream')

@app.route('/chassis_control',methods=['POST'])
def chassis_control():
    data = request.get_json()
    return Response(chassis.handle_msg(data),content_type="application/json")



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)