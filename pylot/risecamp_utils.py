import inspect, glob, os, logging
from collections import namedtuple

import cv2
import erdos
import numpy as  np
import tensorflow as tf

import pylot.utils
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D, \
    OBSTACLE_LABELS, load_coco_bbox_colors, load_coco_labels
from pylot.perception.messages import ObstaclesMessage
from pylot.perception.camera_frame import CameraFrame
from pylot.perception.messages import FrameMessage

ModelParams = namedtuple("ModelParams", "detection_graph, tf_session, detection_boxes, "\
                                        "detection_scores, detection_classes, num_detections, "\
                                        "image_tensor, coco_labels")
SEQUENCE_LENGTH = 55
#SEQUENCE_LENGTH = 1

class CameraReplayOperator(erdos.Operator):
    def __init__(self, camera_stream):
        self.camera_stream = camera_stream
        self.data_path = "/home/erdos/workspace/pylot/data/"

    @staticmethod
    def connect():
        camera_stream = erdos.WriteStream()
        return [camera_stream]

    def run(self):
        # Read data from the folder and present it in increasing order.
        images = glob.glob(self.data_path + '*.png')
        images = sorted(images, key=lambda a: int(os.path.basename(a)[13:os.path.basename(a).find('.')]))

        # Send the data on the output stream.
        for i, image_filename in enumerate(images):
            image = cv2.imread(image_filename)
            timestamp = erdos.Timestamp(coordinates=[i])
            msg = FrameMessage(timestamp, CameraFrame(image, 'BGR'))
            self.camera_stream.send(msg)
            self.camera_stream.send(erdos.WatermarkMessage(timestamp))

def get_camera_stream():
    op_config = erdos.OperatorConfig(name='camera_operator')
    return erdos.connect(CameraReplayOperator, op_config, [])[0]

def connect_visualizer(camera_stream, detected_objects_stream):
    return erdos.ExtractStream(camera_stream), erdos.ExtractStream(detected_objects_stream)

def visualize_detected_objects(camera_stream, obstacles_stream):
    from PIL import Image
    from ipywidgets import Output, Layout
    from IPython.display import display
    out = Output()
    _coco_labels = load_coco_labels("dependencies/models/pylot.names")

    _bbox_colors = load_coco_bbox_colors(_coco_labels)
    with out:
        sequence = 0
        while True:
            image = camera_stream.read()
            #print("Got the camera image for {} {}".format(sequence, type(image)))
            detected_obstacles = obstacles_stream.read()
            #print("Got the detected obstacles for {} {}".format(sequence, type(detected_obstacles)))
            if type(image) is not erdos.WatermarkMessage and type(detected_obstacles) is not erdos.WatermarkMessage:
                sequence += 1
                assert image.timestamp == detected_obstacles.timestamp, "The timestamps were not the same"
                image.frame.annotate_with_bounding_boxes(image.timestamp, detected_obstacles.obstacles, None, _bbox_colors)
                display(Image.fromarray(image.frame.as_rgb_numpy_array()))
                out.clear_output(wait=True)
            if sequence == SEQUENCE_LENGTH:
                break


# Utils for making PSets easier.
def load_model(model_path):
    _detection_graph = tf.Graph()
    # Load the model from the model file.
    pylot.utils.set_tf_loglevel(logging.ERROR)
    with _detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    _gpu_options = tf.GPUOptions(
        allow_growth=True,
        visible_device_list=str(0),
        per_process_gpu_memory_fraction=0.8)
    # Create a TensorFlow session.
    _tf_session = tf.Session(
        graph=_detection_graph,
        config=tf.ConfigProto(gpu_options=_gpu_options))
    # Get the tensors we're interested in.
    _image_tensor = _detection_graph.get_tensor_by_name(
        'image_tensor:0')
    _detection_boxes = _detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    _detection_scores = _detection_graph.get_tensor_by_name(
        'detection_scores:0')
    _detection_classes = _detection_graph.get_tensor_by_name(
        'detection_classes:0')
    _num_detections = _detection_graph.get_tensor_by_name(
        'num_detections:0')
    _coco_labels = load_coco_labels("dependencies/models/pylot.names")
    _bbox_colors = load_coco_bbox_colors(_coco_labels)

    # Save the model.
    model = ModelParams(_detection_graph, _tf_session, _detection_boxes,
            _detection_scores, _detection_classes, _num_detections,
            _image_tensor, _coco_labels)
    # Serve some junk image to load up the model.
    infer_from_model(model, np.zeros((108, 192, 3)), frame=False)
    return model

def infer_from_model(model, image_message, frame=True):
    if frame:
        image_np = image_message.frame.frame
    else:
        image_np = image_message
    image_np_expanded = np.expand_dims(image_np, axis=0)
    (boxes, scores, classes, num_detections) = model.tf_session.run(
        [
            model.detection_boxes, model.detection_scores,
            model.detection_classes, model.num_detections
        ],
        feed_dict={model.image_tensor: image_np_expanded})

    num_detections = int(num_detections[0])
    res_classes = [int(cls) for cls in classes[0][:num_detections]]
    res_boxes = boxes[0][:num_detections]
    res_scores = scores[0][:num_detections]
    obstacles = []
    for i in range(0, num_detections):
        if res_classes[i] in model.coco_labels:
            if (res_scores[i] >= 0.50):
                if (model.coco_labels[res_classes[i]] in OBSTACLE_LABELS):
                    obstacles.append(
                        Obstacle(BoundingBox2D(
                            int(res_boxes[i][1] * 1920),
                            int(res_boxes[i][3] * 1920),
                            int(res_boxes[i][0] * 1080),
                            int(res_boxes[i][2] * 1080)),
                                 res_scores[i],
                                 model.coco_labels[res_classes[i]],
                                 id=0))

    if frame:
        obstacles_message = ObstaclesMessage(image_message.timestamp, obstacles, 0)
    else:
        obstacles_message = ObstaclesMessage(erdos.Timestamp(coordinates=[0]), obstacles, 0)

    return obstacles_message


### Validation code for PSets.

def validate_problem_1(op):
    add_callback_line = list(filter(lambda s: s.find('add_callback') != -1 and not s.startswith('#'),
        map(lambda a: a.strip(), inspect.getsourcelines(op.__init__)[0])))

    if len(add_callback_line) == 0:
        failure_reason = "add_callback was not called!"
        return True, failure_reason

    if len(add_callback_line) != 1:
        failure_reason = "add_callback was called {} times! Make sure there is only one add_callback_invocation".format(len(add_callback_line))
        return True, failure_reason

    add_callback_line = add_callback_line[0].strip() 
    arguments = add_callback_line[add_callback_line.find("(")+1:add_callback_line.find(")")].split(',', maxsplit=1)

    if len(arguments) == 0:
        failure_reason = "add_callback was not invoked with the callback and the output stream."
        return True, failure_reason

    if arguments[0] != "self.on_message":
        failure_reason = "self.on_message was not registered with the camera_stream"
        return True, failure_reason

    if len(arguments) == 1:
        failure_reason = "Only a single argument was passed to the add_callback method. Did you forget the output stream?" 
        return True, failure_reason

    second_argument = arguments[1].strip()

    if not second_argument.startswith('[') or not second_argument.endswith(']'):
        failure_reason = "The second argument to add_callback was not a list! " \
        "Make sure you're invoking add_callback with the list of output stream(s)"
        return True, failure_reason

    list_args = list(map(lambda a: a.strip(), second_argument[1:-1].split(',')))
    if len(list_args) != 1:
        failure_reason = "More than one output stream was passed to add_callback. " \
        "Make sure to only pass the detected_objects_stream."
        return True, failure_reason

    if list_args[0] != "detected_objects_stream":
        failure_reason = "The detected_objects_stream was not passed as the second argument to add_callback"
        return True, failure_reason


    # Check errors in on_message.
    detect_objects_line = list(filter(lambda s: s.find('detect_objects') != -1 and not s.startswith('#'),
        map(lambda a: a.strip(), inspect.getsourcelines(op.on_message)[0])))

    if len(detect_objects_line) == 0:
        failure_reason = "detect_objects was not called!"
        return True, failure_reason

    if len(detect_objects_line) != 1:
        failure_reason = "detect_objects was called {} times! Make sure that there is only invocation of detect_objects".format(len(detect_objects_line))
        return True, failure_reason

    detect_objects_line = detect_objects_line[0].strip()
    saved_variable = detect_objects_line.split('=')[0].strip()
    arguments = detect_objects_line[detect_objects_line.find("(")+1:detect_objects_line.find(")")].split(',', maxsplit=1)

    if len(arguments) != 1:
        failure_reason = "detect_objects was invoked with {} argument(s)! Make sure that it is only invoked with message.".format(len(arguments))
        return True, failure_reason
        
    if arguments[0] != "message":
        failure_reason = "detect_objects was not invoked with the message."
        return True, failure_reason

    # Check errors in send.
    send_line = list(filter(lambda s: s.find('send') != -1 and not s.startswith('#'),
        map(lambda a: a.strip(), inspect.getsourcelines(op.on_message)[0])))

    if len(send_line) == 0:
        failure_reason = "send was not called!"
        return True, failure_reason

    if len(send_line) != 1:
        failure_reason = "send was called {} times! Make sure that there is only one invocation of send".format(len(send_line))
        return True, failure_reason

    send_line = send_line[0].strip()
    arguments = send_line[send_line.find("(")+1:send_line.find(")")].split(",", maxsplit=1)

    if len(arguments) == 1 and arguments[0] == '':
        failure_reason = "send was invoked with no arguments! Make sure that it is only invoked with detected_objects."
        return True, failure_reason

    if len(arguments) > 1:
        failure_reason = "send was invoked with {} argument(s)! Make sure that it is only invoked with detected_objects.".format(len(arguments))
        return True, failure_reason

    if arguments[0] != saved_variable:
        failure_reason = "send was not invoked with {}".format(saved_variable)
        return True, failure_reason

    return False, ""
