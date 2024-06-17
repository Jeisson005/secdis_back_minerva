"""App"""
__author__ = "Edwar Granados and Jehison Rodríguez"
__credits__ = ["Edwar Granados", "Jehison Rodríguez", "Deepak Birla"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Edwar Granados"
__email__ = "hegranadosl@correo.udistrital.edu.co"
__status__ = "Development"

import base64
import csv
import os
import time
import datetime
import json

import cv2
import numpy as np
from flask import Flask, render_template, session
from flask_socketio import SocketIO, emit, disconnect

from config.settings import Config
from owl.detectors.detector_factory import DetectorFactory
from owl.captures.utils.capture_utils import (get_capture_data,
                                              get_video_capture
                                              )
from owl.detectors.utils.detector_utils import prepare_detector, run_detection
from owl.utils.plot import bird_eye_view, social_distancing_view
from owl.utils.utils import (
    get_count,
    get_distances,
    get_transformed_points,
    save_detection_object,
    validate_message_fields, validate_response
)
from owl.views.birds_eye_view import BirdsEyeView

async_mode = None
env_name = os.getenv('FLASK_ENV')
app = Flask(__name__)
app.config.from_object(Config)
socket_ = SocketIO(app, async_mode=async_mode, cors_allowed_origins="*")

video_capture = None
URL_VULCAN = app.config.get('URL_VULCAN_API')


def __disconnect_all(error=None):
    print("\n\nDesconectado ...")
    __disconnect_and_logger(error)
    if video_capture:
        video_capture.release()


def __disconnect_and_logger(error=None):
    """Disconnect and logger"""
    print(error)
    disconnect()


@app.route('/')
def index():
    try:
        return render_template('index.html', async_mode=socket_.async_mode)
    except Exception as ex:
        print(ex)
        return None


@socket_.on('social_distance', namespace='/real_time')
def social_distance(message):
    try:
        model, class_names, scale_w, scale_h, error = prepare_detector(
            message['data'])
        if error:
            emit('my_response',
                 {
                     'error': error
                 },
                 broadcast=True)
            __disconnect_and_logger(error)
        else:
            result, error = run_detection()
            if error:
                pass
            else:
                detection_metrics = result[0]
                boxes_filtered = result[1]
                counted_labels = result[2]
    except Exception as ex:
        # disconnect
        pass


@socket_.on('my_event', namespace='/test')
def test_message(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response', {
        'data': message['data'],
        'count': session['receive_count']
    })


@socket_.on('full_detection', namespace='/test')
def get_full_detection(message):
    global vc
    vc = None
    detector_number = 1

    if detector_number == 1:
        name = "YOLO"
    else:
        name = "TensorFlow"
    detector = DetectorFactory.get_detector(name)
    print("###################################")
    print(f'#           {message}           #')
    print("###################################")
    try:
        print("Entre al try")
        capture_id = message['data']['capture_id']
        print("paso 1")
        print(f'capture_id: {capture_id}')
        if capture_id:
            capture = get_capture_object(capture_id)
            if 'err' in capture:
                disconnect_general(capture['err'])

            # Detection
            path = capture['path']

            vc = cv2.VideoCapture(path)

            # Get video height and width
            height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))

            # Set scale for birds eye view
            # Bird's eye view will only show ROI
            scale_w, scale_h = BirdsEyeView.get_scale(width, height)

            if detector_number == 1:
                cfg_file_path = './models/yolo/yolov4.cfg'
                wgt_file_path = './models/yolo/yolov4.weights'
                cl_path = './models/yolo/coco.names'
            else:
                cfg_file_path = './models/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
                wgt_file_path = './models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
                cl_path = './models/coco.names'

            model = detector.load_detector_model(cfg_file_path, wgt_file_path)

            class_names = detector.load_classes(cl_path)

            classes = None
            scores = None
            boxes = None

            frame_back_flag = False
            prev_frame_time = 0
            new_frame_time = 0

            while cv2.waitKey(1) < 1:
                ret, frame = vc.read()
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                print("FPS: {}".format(int(fps)))

                if frame.shape[0] > 640:
                    frame = cv2.resize(frame, (640, 480))

                session['id'] = session.get('id', 0) + 1
                count_person = 0
                data = {}
                (H, W) = frame.shape[:2]
                if ret:
                    points = capture['points']
                    if points:
                        if not frame_back_flag:
                            frame_back = np.copy(frame)
                            frame_back_flag = True

                        roi_points = [(P['x'], P['y']) for P in points['rectangle']]

                        pts = []
                        center = (points['points']['center']['x'], points['points']['center']['y'])
                        pts.append(center)
                        hor = (points['points']['horizontal']['x'], points['points']['horizontal']['y'])
                        pts.append(hor)
                        ver = (points['points']['vertical']['x'], points['points']['vertical']['y'])
                        pts.append(ver)

                        src = np.float32(np.array(roi_points))
                        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
                        prespective_transform = cv2.getPerspectiveTransform(src, dst)
                        warped_img = cv2.warpPerspective(frame_back, prespective_transform, (W, H))
                        warped_img = cv2.resize(warped_img, (int(W * scale_w), int(H * scale_h)))

                        # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
                        pts = np.float32(np.array([pts]))
                        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

                        distance_w = np.sqrt(
                            (warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
                        distance_h = np.sqrt(
                            (warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
                        pnts = np.array(roi_points, np.int32)
                        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)

                        classes, scores, boxes = detector.detect(frame, 0.5, 0.5, model)

                        boxes1 = []
                        if len(classes) and len(scores):
                            for (class_id, score, box) in zip(classes.flatten(), scores.flatten(), boxes):
                                if class_names[class_id] == 'person':
                                    boxes1.append(box)
                                    count_person += 1

                        # Here we will be using bottom center point of bounding box for all boxes and will transform all those
                        # bottom center points to bird eye view
                        person_points = get_transformed_points(boxes1, prespective_transform)

                        # Here we will calculate distance between transformed points(humans)
                        distances_mat, bxs_mat = get_distances(boxes1, person_points, distance_w, distance_h)
                        risk_count = get_count(distances_mat)

                        frame1 = np.copy(frame)

                        # Draw bird eye view and frame with bouding boxes around humans according to risk factor
                        bird_image = bird_eye_view(frame, warped_img, distances_mat, person_points, scale_w,
                                                   scale_h, risk_count)
                        bird_image = cv2.flip(bird_image, 0)
                        img = social_distancing_view(frame1, bxs_mat, boxes1, risk_count)

                        bird_image = cv2.imencode('.jpg', bird_image)[1].tobytes()
                        bird_image = base64.b64encode(bird_image).decode('utf-8')

                        img = cv2.imencode('.jpg', img)[1].tobytes()
                        img = base64.b64encode(img).decode('utf-8')

                        data = {
                            'safe_count': risk_count[2],
                            'low_risk_count': risk_count[1],
                            'high_risk_count': risk_count[0],
                            'other_metrics': {}
                        }
                        path = str(path)
                        video_name = path.split("/")[-1]
                        with open("./{detector}_{video_name}.csv".format(detector=name, video_name=video_name),
                                  mode='a') as file:
                            writer = csv.writer(file, delimiter=',')
                            writer.writerow([count_person, risk_count[2], risk_count[1], risk_count[0]])
                        print("count: {}".format(count_person))
                        print("data: {}".format(data))
                        print("donde es")
                        print(type(classes))
                        print(type(scores))
                        print(type(boxes))
                        print(classes)
                        print(scores)
                        print(boxes)

                        # cv2.imshow('Social distance', img)
                        # cv2.imshow("Bird's eye view", bird_image)
                        classes_aux = classes.flatten().tolist()
                        scores_aux = scores.flatten().tolist()
                        boxes_aux = boxes.tolist()

                        detection_metrics = {
                            "classes": classes_aux,
                            "scores": scores_aux,
                            "boxes": boxes_aux
                        }
                        detection_metrics = json.dumps(detection_metrics)
                        detection = {
                            "file_name": f"{capture['name']}_{str(int(time.time()))}",
                            "file_path": "None",
                            "processing_date": datetime.datetime.now(),
                            "safe_count": risk_count[2],
                            "low_risk_count": risk_count[1],
                            "high_risk_count": risk_count[0],
                            "detection_metrics": detection_metrics,
                            "other_metrics": {},
                            "detection_model": 1,
                            "capture": int(capture['id'])
                        }
                        print("vamos a guardar los datos")
                        response = save_detection_object(detection)
                        print(response)
                        print("emitir")
                        emit('my_response', {
                            'id': session['id'],
                            'data': message['data'],
                            'image': "data:image/jpeg;base64,{}".format(img),
                            'image_bird': "data:image/jpeg;base64,{}".format(bird_image),
                            'count': count_person,
                            'other': data,
                        }, broadcast=True)
                        time.sleep(0)
                    else:
                        classes, scores, boxes = detector.detect(frame, 0.5, 0.5, model)
                        if len(classes) and len(scores):
                            for (class_id, score, box) in zip(classes.flatten(), scores.flatten(), boxes):
                                if class_names[class_id] == 'person':
                                    label = "{}".format(class_names[class_id])
                                    count_person += 1
                                    frame = detector.draw_rectangle(frame, box, (0, 0, 0), 2)
                                    cv2.putText(
                                        frame,
                                        label,
                                        (box[0], box[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 0, 0),
                                        2
                                    )
                                    data = {
                                        'safe_count': count_person,
                                        'low_risk_count': count_person,
                                        'high_risk_count': count_person,
                                        'other_metrics': {}
                                    }
                        frame = cv2.imencode('.jpg', frame)[1].tobytes()
                        frame = base64.b64encode(frame).decode('utf-8')
                        emit('my_response', {
                            'id': session['id'],
                            'data': message['data'],
                            'image': "data:image/jpeg;base64,{}".format(frame),
                            'image_bird': "data:image/jpeg;base64,{}".format(frame),
                            'count': count_person,
                            'other': data,
                        }, broadcast=True)
                        time.sleep(0)
        else:
            disconnect_general('capture_id is None')
    except Exception as ex:
        print(f"Volamos full {ex}")
        disconnect_general(ex)


@socket_.on('my_broadcast_event', namespace='/test')
def test_broadcast_message(message):
    global vc
    vc = None
    print("->Mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
    print(message)
    print("<-Mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
    try:
        path = '0'

        if path == '0':
            vc = cv2.VideoCapture(0)
        else:
            vc = cv2.VideoCapture(path)

        cfg_file_path = './models/yolov4.cfg'
        wgt_file_path = './models/yolov4.weights'
        cl_path = './models/coco.names'

        net = cv2.dnn.readNetFromDarknet(cfg_file_path, wgt_file_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(416, 416), scale=1 / 255)

        class_names = open(cl_path).read().strip().split('\n')

        classes = None
        scores = None
        boxes = None

        while cv2.waitKey(1) < 1:
            ret, frame = vc.read()
            session['id'] = session.get('id', 0) + 1
            count_person = 0
            data = {}
            (H, W) = frame.shape[:2]
            if ret:
                classes, scores, boxes = model.detect(
                    frame=frame,
                    confThreshold=0.5,
                    nmsThreshold=0.5
                )
                if len(classes) and len(scores):
                    for (class_id, score, box) in zip(classes.flatten(), scores.flatten(), boxes):
                        if class_names[class_id] == 'person':
                            label = "{}".format(class_names[class_id])
                            count_person += 1
                            cv2.rectangle(frame, box, color=(0, 255, 255), thickness=2)
                            cv2.putText(
                                frame,
                                label,
                                (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 0),
                                2
                            )
                            data = {
                                'safe_count': count_person,
                                'low_risk_count': count_person,
                                'high_risk_count': count_person,
                                'other_metrics': {}
                            }
                frame = cv2.imencode('.jpg', frame)[1].tobytes()
                frame = base64.b64encode(frame).decode('utf-8')
                emit('my_response', {
                    'id': session['id'],
                    'data': message['data'],
                    'image': "data:image/jpeg;base64,{}".format(frame),
                    'image_bird': "data:image/jpeg;base64,{}".format(frame),
                    'count': count_person,
                    'other': data,
                }, broadcast=True)
                time.sleep(0)
    except Exception as ex:
        print(f"Volamos {ex}")
        disconnect_general(ex)


@socket_.on('disconnect_request', namespace='/test')
def disconnect_request(message):
    __disconnect_all()


if __name__ == '__main__':
    socket_.run(app, debug=True)
