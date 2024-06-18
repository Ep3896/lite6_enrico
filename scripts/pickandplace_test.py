#!/usr/bin/env python

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import tf2_ros as tf
import time

def open_gripper(posture):
    posture.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    point = moveit_msgs.msg.JointTrajectoryPoint()
    point.positions = [0.04, 0.04]
    point.time_from_start = rospy.Duration(0.5)
    posture.points = [point]

def closed_gripper(posture):
    posture.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    point = moveit_msgs.msg.JointTrajectoryPoint()
    point.positions = [0.00, 0.00]
    point.time_from_start = rospy.Duration(0.5)
    posture.points = [point]

def pick(move_group):
    grasps = []
    grasp = moveit_msgs.msg.Grasp()

    # Setting grasp pose
    grasp.grasp_pose.header.frame_id = "panda_link0"
    orientation = tf.quaternion_from_euler(-3.14 / 2, -3.14 / 4, -3.14 / 2)
    grasp.grasp_pose.pose.orientation = geometry_msgs.msg.Quaternion(*orientation)
    grasp.grasp_pose.pose.position.x = 0.415
    grasp.grasp_pose.pose.position.y = 0
    grasp.grasp_pose.pose.position.z = 0.5

    # Setting pre-grasp approach
    grasp.pre_grasp_approach.direction.header.frame_id = "panda_link0"
    grasp.pre_grasp_approach.direction.vector.x = 1.0
    grasp.pre_grasp_approach.min_distance = 0.095
    grasp.pre_grasp_approach.desired_distance = 0.115

    # Setting post-grasp retreat
    grasp.post_grasp_retreat.direction.header.frame_id = "panda_link0"
    grasp.post_grasp_retreat.direction.vector.z = 1.0
    grasp.post_grasp_retreat.min_distance = 0.1
    grasp.post_grasp_retreat.desired_distance = 0.25

    # Setting posture of eef before grasp
    open_gripper(grasp.pre_grasp_posture)

    # Setting posture of eef during grasp
    closed_gripper(grasp.grasp_posture)

    grasps.append(grasp)

    move_group.set_support_surface_name("table1")
    move_group.pick("object", grasps)

def place(move_group):
    place_location = moveit_msgs.msg.PlaceLocation()

    # Setting place location pose
    place_location.place_pose.header.frame_id = "panda_link0"
    orientation = tf.quaternion_from_euler(0, 0, 3.14 / 2)
    place_location.place_pose.pose.orientation = geometry_msgs.msg.Quaternion(*orientation)
    place_location.place_pose.pose.position.x = 0
    place_location.place_pose.pose.position.y = 0.5
    place_location.place_pose.pose.position.z = 0.5

    # Setting pre-place approach
    place_location.pre_place_approach.direction.header.frame_id = "panda_link0"
    place_location.pre_place_approach.direction.vector.z = -1.0
    place_location.pre_place_approach.min_distance = 0.095
    place_location.pre_place_approach.desired_distance = 0.115

    # Setting post-place retreat
    place_location.post_place_retreat.direction.header.frame_id = "panda_link0"
    place_location.post_place_retreat.direction.vector.y = -1.0
    place_location.post_place_retreat.min_distance = 0.1
    place_location.post_place_retreat.desired_distance = 0.25

    # Setting posture of eef after placing object
    open_gripper(place_location.post_place_posture)

    move_group.set_support_surface_name("table2")
    move_group.place("object", [place_location])

def add_collision_objects(planning_scene_interface):
    collision_objects = []

    # Add the first table
    table1 = moveit_msgs.msg.CollisionObject()
    table1.id = "table1"
    table1.header.frame_id = "panda_link0"
    table1.primitives = [moveit_msgs.msg.SolidPrimitive()]
    table1.primitives[0].type = table1.primitives[0].BOX
    table1.primitives[0].dimensions = [0.2, 0.4, 0.4]
    table1.primitive_poses = [geometry_msgs.msg.Pose()]
    table1.primitive_poses[0].position.x = 0.5
    table1.primitive_poses[0].position.y = 0
    table1.primitive_poses[0].position.z = 0.2
    table1.operation = table1.ADD

    # Add the second table
    table2 = moveit_msgs.msg.CollisionObject()
    table2.id = "table2"
    table2.header.frame_id = "panda_link0"
    table2.primitives = [moveit_msgs.msg.SolidPrimitive()]
    table2.primitives[0].type = table2.primitives[0].BOX
    table2.primitives[0].dimensions = [0.4, 0.2, 0.4]
    table2.primitive_poses = [geometry_msgs.msg.Pose()]
    table2.primitive_poses[0].position.x = 0
    table2.primitive_poses[0].position.y = 0.5
    table2.primitive_poses[0].position.z = 0.2
    table2.operation = table2.ADD

    # Add the object to be manipulated
    object_to_manipulate = moveit_msgs.msg.CollisionObject()
    object_to_manipulate.id = "object"
    object_to_manipulate.header.frame_id = "panda_link0"
    object_to_manipulate.primitives = [moveit_msgs.msg.SolidPrimitive()]
    object_to_manipulate.primitives[0].type = object_to_manipulate.primitives[0].BOX
    object_to_manipulate.primitives[0].dimensions = [0.02, 0.02, 0.2]
    object_to_manipulate.primitive_poses = [geometry_msgs.msg.Pose()]
    object_to_manipulate.primitive_poses[0].position.x = 0.5
    object_to_manipulate.primitive_poses[0].position.y = 0
    object_to_manipulate.primitive_poses[0].position.z = 0.5
    object_to_manipulate.operation = object_to_manipulate.ADD

    collision_objects.append(table1)
    collision_objects.append(table2)
    collision_objects.append(object_to_manipulate)

    planning_scene_interface.apply_collision_objects(collision_objects)

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('panda_arm_pick_place', anonymous=True)
    rospy.loginfo("Starting the pick and place node...")

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    move_group.set_planning_time(45.0)

    add_collision_objects(scene)

    rospy.sleep(1)

    pick(move_group)

    rospy.sleep(1)

    place(move_group)

    rospy.spin()

if __name__ == '__main__':
    main()
