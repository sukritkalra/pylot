from collections import deque
import erdust

from pylot.perception.detection.utils import get_precision_recall_at_iou
from pylot.utils import time_epoch_ms


class DetectionEvalGroundOperator(erdust.Operator):
    def __init__(self,
                 obstacles_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        obstacles_stream.add_callback(self.on_ground_obstacles)
        self._name = name
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdust.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._flags = flags
        self._ground_bboxes = deque()
        self._iou_thresholds = [0.1 * i for i in range(1, 10)]

    @staticmethod
    def connect(obstacles_stream):
        return []

    def on_ground_obstacles(self, msg):
        # Ignore the first several seconds of the simulation because the car is
        # not moving at the beginning.
        game_time = msg.timestamp.coordinates[0]
        bboxes = []
        # Select the pedestrian bounding boxes.
        for det_obj in msg.detected_objects:
            if det_obj.label == 'pedestrian':
                bboxes.append(det_obj.corners)

        # Remove the buffered bboxes that are too old.
        while (len(self._ground_bboxes) > 0 and
               game_time - self._ground_bboxes[0][0] >
               self._flags.decay_max_latency):
            self._ground_bboxes.popleft()

        for (old_game_time, old_bboxes) in self._ground_bboxes:
            # Ideally, we would like to take multiple precision values at
            # different recalls and average them, but we can't vary model
            # confidence, so we just return the actual precision.
            if (len(bboxes) > 0 or len(old_bboxes) > 0):
                latency = game_time - old_game_time
                precisions = []
                for iou in self._iou_thresholds:
                    (precision, _) = get_precision_recall_at_iou(
                        bboxes, old_bboxes, iou)
                    precisions.append(precision)
                self._logger.info("Precision {}".format(precisions))
                avg_precision = float(sum(precisions)) / len(precisions)
                self._logger.info(
                    "The latency is {} and the average precision is {}".format(
                        latency, avg_precision))
                self._csv_logger.info('{},{},{},{}'.format(
                    time_epoch_ms(), self._name, latency, avg_precision))

        # Buffer the new bounding boxes.
        self._ground_bboxes.append((game_time, bboxes))
