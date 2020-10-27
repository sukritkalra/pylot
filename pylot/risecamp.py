import signal

from absl import app, flags

import erdos

import pylot.flags
import pylot.component_creator  # noqa: I100
import pylot.operator_creator
import pylot.utils
from pylot.simulation.utils import get_world, set_asynchronous_mode
from pylot.planning.planning_operator import PlanningOperator

FLAGS = flags.FLAGS

flags.DEFINE_list('goal_location', '234, 59, 39', 'Ego-vehicle goal location')

# The location of the center camera relative to the ego-vehicle.
CENTER_CAMERA_LOCATION = pylot.utils.Location(1.3, 0.0, 1.8)

class Simulator():
    def __init__(self, sim_pid):
        self.sim_pid = sim_pid

    def reset(self):
        kill_simulator()


class Scenario():
    def __init__(self, scenario_pid):
        self.scenario_pid = scenario_pid

    def reset(self):
        self.scenario_pid.kill()

def start_simulator():
    import time, subprocess
    print("Starting the simulator...")
    simulator = subprocess.Popen(["/home/erdos/workspace/pylot/scripts/run_simulator.sh"])
    time.sleep(5)
    print("Finished running the simulator...")
    return Simulator(simulator)


def kill_simulator():
    import subprocess
    subprocess.call("pkill CarlaUE4", shell=True)

def setup_scenario():
    import time, subprocess, os
    print("Setting up the scenario... ")
    scenario_runner = subprocess.Popen(["python3", "scenario_runner.py", "--scenario", "ERDOSPedestrianBehindCar", "--reloadWorld", "--output", "--timeout", "6000"], cwd="/home/erdos/workspace/scenario_runner")
    time.sleep(5)
    print("Finished setting up the scenario...")
    return Scenario(scenario_runner)

class SolvedPlanner(PlanningOperator):
    def on_prediction_update(self, message):
        return message.predictions

def start_pylot(planner=SolvedPlanner, vehicle_speed=12.0, time_discretization=0.35, road_width=0.35):
    from random import randint
    filename = 'video-{}.mp4'.format(randint(1, 1000))
    FLAGS([
        "risecamp.py", 
        "--flagfile=configs/scenarios/risecamp.conf", 
        "--target_speed={}".format(float(vehicle_speed)), 
        "--carla_traffic_manager_port={}".format(8000+randint(0, 100)),
        "--d_road_w={}".format(float(road_width)),
        "--dt={}".format(float(time_discretization)),
        ])
    main(None, planner, filename)
    return filename

def add_evaluation_operators(vehicle_id_stream, pose_stream, imu_stream,
                             pose_stream_for_control,
                             waypoints_stream_for_control):
    if FLAGS.evaluation:
        # Add the collision sensor.
        collision_stream = pylot.operator_creator.add_collision_sensor(
            vehicle_id_stream)

        # Add the lane invasion sensor.
        lane_invasion_stream = pylot.operator_creator.add_lane_invasion_sensor(
            vehicle_id_stream)

        # Add the traffic light invasion sensor.
        traffic_light_invasion_stream = \
            pylot.operator_creator.add_traffic_light_invasion_sensor(
                vehicle_id_stream, pose_stream)

        # Add the evaluation logger.
        pylot.operator_creator.add_eval_metric_logging(
            collision_stream, lane_invasion_stream,
            traffic_light_invasion_stream, imu_stream, pose_stream)

        # Add control evaluation logging operator.
        pylot.operator_creator.add_control_evaluation(
            pose_stream_for_control, waypoints_stream_for_control)

def add_spectator_camera(transform,
                   vehicle_id_stream,
                   release_sensor_stream,
                   name='center_rgb_camera',
                   fov=90):
    from pylot.drivers.sensor_setup import RGBCameraSetup
    rgb_camera_setup = RGBCameraSetup(name, 480, 480, transform,
                                      fov)
    camera_stream, notify_reading_stream = pylot.operator_creator._add_camera_driver(
        vehicle_id_stream, release_sensor_stream, rgb_camera_setup)
    return (camera_stream, notify_reading_stream, rgb_camera_setup)

def driver(planner):
    transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                      pylot.utils.Rotation(pitch=-15))
    streams_to_send_top_on = []
    control_loop_stream = erdos.LoopStream()
    time_to_decision_loop_stream = erdos.LoopStream()
    if FLAGS.simulator_mode == 'pseudo-asynchronous':
        release_sensor_stream = erdos.LoopStream()
        pipeline_finish_notify_stream = erdos.LoopStream()
    else:
        release_sensor_stream = erdos.IngestStream()
        pipeline_finish_notify_stream = erdos.IngestStream()
    notify_streams = []

    # Create operator that bridges between pipeline and the simulator.
    (
        pose_stream,
        pose_stream_for_control,
        ground_traffic_lights_stream,
        ground_obstacles_stream,
        ground_speed_limit_signs_stream,
        ground_stop_signs_stream,
        vehicle_id_stream,
        open_drive_stream,
        global_trajectory_stream,
    ) = pylot.operator_creator.add_simulator_bridge(
        control_loop_stream,
        release_sensor_stream,
        pipeline_finish_notify_stream,
    )

    # Add sensors.
    (center_camera_stream, notify_rgb_stream,
     center_camera_setup) = pylot.operator_creator.add_rgb_camera(
         transform, vehicle_id_stream, release_sensor_stream)
    (spectator_camera_stream, notify_spectator_camera_stream,
        spectator_camera_setup) = add_spectator_camera(
            pylot.utils.Transform(pylot.utils.Location(-5.0, 0.0, 5.0),
                pylot.utils.Rotation(pitch=-15)), vehicle_id_stream, release_sensor_stream)

    notify_streams.append(notify_rgb_stream)
    notify_streams.append(notify_spectator_camera_stream)
    if pylot.flags.must_add_depth_camera_sensor():
        (depth_camera_stream, notify_depth_stream,
         depth_camera_setup) = pylot.operator_creator.add_depth_camera(
             transform, vehicle_id_stream, release_sensor_stream)
    else:
        depth_camera_stream = None
    if pylot.flags.must_add_segmented_camera_sensor():
        (ground_segmented_stream, notify_segmented_stream,
         _) = pylot.operator_creator.add_segmented_camera(
             transform, vehicle_id_stream, release_sensor_stream)
    else:
        ground_segmented_stream = None

    if pylot.flags.must_add_lidar_sensor():
        # Place LiDAR sensor in the same location as the center camera.
        (point_cloud_stream, notify_lidar_stream,
         lidar_setup) = pylot.operator_creator.add_lidar(
             transform, vehicle_id_stream, release_sensor_stream)
    else:
        point_cloud_stream = None
        lidar_setup = None

    if FLAGS.obstacle_location_finder_sensor == 'lidar':
        depth_stream = point_cloud_stream
        # Camera sensors are slower than the lidar sensor.
        notify_streams.append(notify_lidar_stream)
    elif FLAGS.obstacle_location_finder_sensor == 'depth_camera':
        depth_stream = depth_camera_stream
        notify_streams.append(notify_depth_stream)
    else:
        raise ValueError(
            'Unknown --obstacle_location_finder_sensor value {}'.format(
                FLAGS.obstacle_location_finder_sensor))

    imu_stream = None
    if pylot.flags.must_add_imu_sensor():
        (imu_stream, _) = pylot.operator_creator.add_imu(
            pylot.utils.Transform(location=pylot.utils.Location(),
                                  rotation=pylot.utils.Rotation()),
            vehicle_id_stream)

    gnss_stream = None
    if pylot.flags.must_add_gnss_sensor():
        (gnss_stream, _) = pylot.operator_creator.add_gnss(
            pylot.utils.Transform(location=pylot.utils.Location(),
                                  rotation=pylot.utils.Rotation()),
            vehicle_id_stream)

    if FLAGS.localization:
        pose_stream = pylot.operator_creator.add_localization(
            imu_stream, gnss_stream, pose_stream)

    obstacles_stream, perfect_obstacles_stream = \
        pylot.component_creator.add_obstacle_detection(
            center_camera_stream, center_camera_setup, pose_stream,
            depth_stream, depth_camera_stream, ground_segmented_stream,
            ground_obstacles_stream, ground_speed_limit_signs_stream,
            ground_stop_signs_stream, time_to_decision_loop_stream)
    tl_transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                         pylot.utils.Rotation())
    traffic_lights_stream, tl_camera_stream = \
        pylot.component_creator.add_traffic_light_detection(
            tl_transform, vehicle_id_stream, release_sensor_stream,
            pose_stream, depth_stream, ground_traffic_lights_stream)

    lane_detection_stream = pylot.component_creator.add_lane_detection(
        center_camera_stream, pose_stream, open_drive_stream)
    if lane_detection_stream is None:
        lane_detection_stream = erdos.IngestStream()
        streams_to_send_top_on.append(lane_detection_stream)

    obstacles_tracking_stream = pylot.component_creator.add_obstacle_tracking(
        center_camera_stream, center_camera_setup, obstacles_stream,
        depth_stream, vehicle_id_stream, pose_stream, ground_obstacles_stream,
        time_to_decision_loop_stream)

    segmented_stream = pylot.component_creator.add_segmentation(
        center_camera_stream, ground_segmented_stream)

    depth_stream = pylot.component_creator.add_depth(transform,
                                                     vehicle_id_stream,
                                                     center_camera_setup,
                                                     depth_camera_stream)

    if FLAGS.fusion:
        pylot.operator_creator.add_fusion(pose_stream, obstacles_stream,
                                          depth_stream,
                                          ground_obstacles_stream)

    prediction_stream, prediction_camera_stream, notify_prediction_stream = \
        pylot.component_creator.add_prediction(
            obstacles_tracking_stream, vehicle_id_stream, transform,
            release_sensor_stream, pose_stream, point_cloud_stream,
            lidar_setup)
    if prediction_stream is None:
        prediction_stream = obstacles_stream
    if notify_prediction_stream:
        notify_streams.append(notify_prediction_stream)

    goal_location = pylot.utils.Location(float(FLAGS.goal_location[0]),
                                         float(FLAGS.goal_location[1]),
                                         float(FLAGS.goal_location[2]))
    waypoints_stream = pylot.component_creator.add_planning(planner,
        goal_location, pose_stream, prediction_stream, traffic_lights_stream,
        lane_detection_stream, open_drive_stream, global_trajectory_stream,
        time_to_decision_loop_stream)

    if FLAGS.simulator_mode == "pseudo-asynchronous":
        # Add a synchronizer in the pseudo-asynchronous mode.
        (
            waypoints_stream_for_control,
            pose_stream_for_control,
            sensor_ready_stream,
            _pipeline_finish_notify_stream,
        ) = pylot.operator_creator.add_planning_pose_synchronizer(
            waypoints_stream, pose_stream_for_control, pose_stream,
            *notify_streams)
        release_sensor_stream.set(sensor_ready_stream)
        pipeline_finish_notify_stream.set(_pipeline_finish_notify_stream)
    else:
        waypoints_stream_for_control = waypoints_stream
        pose_stream_for_control = pose_stream

    control_stream = pylot.component_creator.add_control(
        pose_stream_for_control, waypoints_stream_for_control,
        vehicle_id_stream, perfect_obstacles_stream)
    control_loop_stream.set(control_stream)

    add_evaluation_operators(vehicle_id_stream, pose_stream, imu_stream,
                             pose_stream_for_control,
                             waypoints_stream_for_control)

    time_to_decision_stream = pylot.operator_creator.add_time_to_decision(
        pose_stream, obstacles_stream)
    time_to_decision_loop_stream.set(time_to_decision_stream)

    control_display_stream = None
    if pylot.flags.must_visualize():
        control_display_stream, ingest_streams = \
            pylot.operator_creator.add_visualizer(
                pose_stream, center_camera_stream, tl_camera_stream,
                prediction_camera_stream, depth_camera_stream,
                point_cloud_stream, segmented_stream, imu_stream,
                obstacles_stream, traffic_lights_stream,
                obstacles_tracking_stream, lane_detection_stream,
                prediction_stream, waypoints_stream, control_stream)
        streams_to_send_top_on += ingest_streams

    camera_visualize_stream = erdos.ExtractStream(spectator_camera_stream)
    pose_synchronize_camera_stream = erdos.ExtractStream(pose_stream)
    top_down_visualize_stream = erdos.ExtractStream(prediction_camera_stream)
    prediction_stream_ex = erdos.ExtractStream(prediction_stream)
    traffic_lights_stream_ex = erdos.ExtractStream(traffic_lights_stream)
    waypoints_stream_ex = erdos.ExtractStream(waypoints_stream)

    node_handle = erdos.run_async()

    for stream in streams_to_send_top_on:
        stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
    # If we did not use the pseudo-asynchronous mode, ask the sensors to
    # release their readings whenever.
    if FLAGS.simulator_mode != "pseudo-asynchronous":
        release_sensor_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
        pipeline_finish_notify_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    return (
            node_handle, 
            control_display_stream, 
            camera_visualize_stream, 
            pose_synchronize_camera_stream, 
            top_down_visualize_stream,
            prediction_stream_ex,
            traffic_lights_stream_ex,
            waypoints_stream_ex,
            )


def shutdown_pylot(node_handle, client, world):
    node_handle.shutdown()
    if FLAGS.simulation_recording_file is not None:
        client.stop_recorder()
    set_asynchronous_mode(world)
    if pylot.flags.must_visualize():
        import pygame
        pygame.quit()


def shutdown(sig, frame):
    raise KeyboardInterrupt


def visualize_camera_stream(camera_stream, pose_stream, top_down_visualize_stream, 
        prediction_stream, traffic_lights_stream, waypoints_stream, world, node_handle, client, filename):
    from PIL import Image
    from ipywidgets import Output, Layout
    from IPython.display import display, Video
    import time, erdos, random, cv2
    import numpy as np
    from pylot.planning.world import World
    out = Output()
    logger = erdos.utils.setup_logging("visualizer", None)
    planning_world = World(FLAGS, logger)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(filename, fourcc, 25, (960, 480))
    with out:
        while True:
            pose = pose_stream.read()
            if type(pose) is not erdos.WatermarkMessage and pose.data.transform.location.l2_distance(pylot.utils.Location(0, 0, 0)) < 0.01:
                shutdown_pylot(node_handle, client, world)
                video.release()
                break
                
            image = camera_stream.read()
            top_down_image = top_down_visualize_stream.read()
            prediction_msg = prediction_stream.read()
            traffic_lights_msg = traffic_lights_stream.read()
            waypoint_msg = waypoints_stream.read()

            if type(image) is not erdos.WatermarkMessage and type(top_down_image) is not erdos.WatermarkMessage \
                    and type(pose) is not erdos.WatermarkMessage and type(prediction_msg) is not erdos.WatermarkMessage \
                    and type(traffic_lights_msg) is not erdos.WatermarkMessage:
                # Get the top down image.
                top_down_image.frame.transform_to_cityscapes()
                planning_world.update(image.timestamp, pose.data, prediction_msg.predictions, traffic_lights_msg.obstacles, None, None)
                planning_world.update_waypoints(None, waypoint_msg.waypoints)
                planning_world.draw_on_frame(top_down_image.frame)

                # Get the speed on the image.
                #image.frame.draw_text(Vector2D(10, 10), "Speed: {:.1}m/s".format(pose.data.forward_speed))

                image_array = np.concatenate((image.frame.as_rgb_numpy_array(), top_down_image.frame.as_cityscapes_palette()), axis=1)
                display(Image.fromarray(image_array))
                video.write(cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
                out.clear_output(wait=True)
            else:
                if image.timestamp.is_top:
                    shutdown_pylot(node_handle, client, world)

            #ego_vehicle_location = ego_vehicle.get_location()
            #if ego_vehicle_location == carla.Location(0.00, 0.00, 0.00):
            #    shutdown_pylot(node_handle, client, world)
            #    break
            #time.sleep(1)

    out.clear_output()

def main(args, planner, filename):
    # Connect an instance to the simulator to make sure that we can turn the
    # synchronous mode off after the script finishes running.
    client, world = get_world(FLAGS.simulator_host, FLAGS.simulator_port,
                              FLAGS.simulator_timeout)
    try:
        if FLAGS.simulation_recording_file is not None:
            client.start_recorder(FLAGS.simulation_recording_file)
        (node_handle, 
                control_display_stream, 
                camera_visualize_stream, 
                pose_synchronize_stream, 
                top_down_visualize_stream, 
                prediction_stream,
                traffic_lights_stream,
                waypoints_stream) = driver(planner)
        signal.signal(signal.SIGINT, shutdown)
        if pylot.flags.must_visualize():
            pylot.utils.run_visualizer_control_loop(control_display_stream)
        visualize_camera_stream(camera_visualize_stream, 
                pose_synchronize_stream, 
                top_down_visualize_stream, 
                prediction_stream, 
                traffic_lights_stream,
                waypoints_stream,
                world, node_handle, client, filename)
        node_handle.wait()
    except KeyboardInterrupt:
        shutdown_pylot(node_handle, client, world)
    except Exception:
        shutdown_pylot(node_handle, client, world)
        raise


if __name__ == '__main__':
    app.run(main)
