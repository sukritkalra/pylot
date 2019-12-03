from collections import deque
import erdust
import time

from pylot.perception.detection.utils import visualize_image
from pylot.utils import time_epoch_ms


class ObjectTrackerOperator(erdust.Operator):
    def __init__(self,
                 obstacles_stream,
                 camera_stream,
                 obstacle_tracking_stream,
                 name,
                 tracker_type,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):

        obstacles_stream.add_callback(self.on_obstacles_msg)
        camera_stream.add_callback(self.on_frame_msg)
        self._name = name
        self._flags = flags
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdust.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._tracker_type = tracker_type
        try:
            if tracker_type == 'cv2':
                from pylot.perception.tracking.cv2_tracker import\
                    MultiObjectCV2Tracker
                self._tracker = MultiObjectCV2Tracker(self._flags)
            elif tracker_type == 'da_siam_rpn':
                from pylot.perception.tracking.da_siam_rpn_tracker import\
                    MultiObjectDaSiamRPNTracker
                self._tracker = MultiObjectDaSiamRPNTracker(self._flags)
            elif tracker_type == 'deep_sort':
                from pylot.perception.tracking.deep_sort_tracker import\
                    MultiObjectDeepSORTTracker
                self._tracker = MultiObjectDeepSORTTracker(self._flags,
                                                           self._logger)
            elif tracker_type == 'sort':
                from pylot.perception.tracking.sort_tracker import\
                    MultiObjectSORTTracker
                self._tracker = MultiObjectSORTTracker(self._flags)
            else:
                self._logger.fatal(
                    'Unexpected tracker type {}'.format(tracker_type))
        except ImportError:
            self._logger.fatal('Error importing {}'.format(tracker_type))
        # True when the tracker is ready to update bboxes.
        self._ready_to_update = False
        self._ready_to_update_timestamp = None
        self._to_process = deque()

    @staticmethod
    def connect(obstacles_stream, camera_stream):
        obstacle_tracking_stream = erdust.WriteStream()
        return [obstacle_tracking_stream]

    def on_frame_msg(self, msg):
        """ Invoked when a FrameMessage is received on the camera stream."""
        start_time = time.time()
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        frame = msg.frame
        # Store frames so that they can be re-processed once we receive the
        # next update from the detector.
        self._to_process.append((msg.timestamp, frame))
        # Track if we have a tracker ready to accept new frames.
        if self._ready_to_update:
            self.__track_bboxes_on_frame(frame, msg.timestamp, False)
        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(time_epoch_ms(),
                                                     self._name, msg.timestamp,
                                                     runtime))

    def on_obstacles_msg(self, msg):
        """ Invoked when detected objects are received on the stream."""
        self._ready_to_update = False
        self._logger.info("Received {} bboxes for {}".format(
            len(msg.detected_objects), msg.timestamp))
        # Remove frames that are older than the detector update.
        while len(self._to_process
                  ) > 0 and self._to_process[0][0] < msg.timestamp:
            self._logger.info("Removing stale {} {}".format(
                self._to_process[0][0], msg.timestamp))
            self._to_process.popleft()

        # Track all pedestrians.
        bboxes, ids, confidence_scores = self.__get_pedestrians(
            msg.detected_objects)
        if len(bboxes) > 0:
            if len(self._to_process) > 0:
                # Found the frame corresponding to the bounding boxes.
                (timestamp, frame) = self._to_process.popleft()
                assert timestamp == msg.timestamp
                # Re-initialize trackers.
                self.__initialize_trackers(
                    frame, bboxes, msg.timestamp, confidence_scores)
                self._logger.info('Trackers have {} frames to catch-up'.format(
                    len(self._to_process)))
                for (timestamp, frame) in self._to_process:
                    if self._ready_to_update:
                        self.__track_bboxes_on_frame(frame, timestamp, True)
            else:
                self._logger.info(
                    'Received bboxes update {}, but no frame to process'.
                    format(msg.timestamp))

    def __get_highest_confidence_pedestrian(self, detected_objs):
        max_confidence = 0
        max_corners = None
        for detected_obj in detected_objs:
            if (detected_obj.label == 'person' and
                detected_obj.confidence > max_confidence):
                max_corners = detected_obj.corners
                max_confidence = detected_obj.confidence
        if max_corners:
            return [max_corners]
        else:
            return []

    def __get_pedestrians(self, detector_objs):
        bboxes = []
        ids = []
        confidence_scores = []
        for detected_obj in detector_objs:
            if detected_obj.label == 'person':
                bboxes.append(detected_obj.corners)
                ids.append(detected_obj.obj_id)
                confidence_scores.append(detected_obj.confidence)
        return bboxes, ids, confidence_scores

    def __initialize_trackers(
            self, frame, bboxes, timestamp, confidence_scores):
        self._ready_to_update = True
        self._ready_to_update_timestamp = timestamp
        self._logger.info('Restarting trackers at frame {}'.format(timestamp))
        self._tracker.reinitialize(frame, bboxes, confidence_scores)

    def __track_bboxes_on_frame(self, frame, timestamp, catch_up):
        self._logger.info('Processing frame {}'.format(timestamp))
        # Sequentually update state for each bounding box.
        ok, tracked_objects = self._tracker.track(frame)
        if not ok:
            self._logger.error(
                'Tracker failed at timestamp {} last ready_to_update at {}'.
                format(timestamp, self._ready_to_update_timestamp))
            # The tracker must be reinitialized.
            self._ready_to_update = False
        else:
            if self._flags.visualize_tracker_output and not catch_up:
                for tracked_object in tracked_objects:
                    # tracked objects have no label, draw black bbox for them ([0, 0, 0])
                    tracked_object.visualize_on_img(frame, {"": [0, 0, 0]})
                visualize_image(self._name, frame)

