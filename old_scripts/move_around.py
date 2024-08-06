import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, Point, Quaternion
from lite6_enrico_interfaces.action import GoToPose
from rclpy.action import ActionClient

class MoveAroundNode(Node):
    def __init__(self):
        super().__init__('move_around_node')
        self.subscription = self.create_subscription(Bool, '/start_signal', self.start_callback, 10)
        self._action_client = ActionClient(self, GoToPose, 'go_to_pose')
        self.moving = False
        self.goals = Pose(position=Point(x=0.1, y=0.0, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))

    def start_callback(self, msg):
        if msg.data and not self.moving:
            self.moving = True
            self.move_to_next_goal()

    def move_to_next_goal(self):
        if self.goal_index >= len(self.goals):
            self.goal_index = 0

        goal_pose = self.goals[self.goal_index]
        self.send_goal(goal_pose)

    def send_goal(self, pose):
        goal_msg = GoToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info(f'Sending goal: {pose}')
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            self.moving = False
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result}')
        if result.success:
            self.get_logger().info('Goal succeeded!')
        else:
            self.get_logger().info('Goal failed!')
        
        self.goal_index += 1
        self.move_to_next_goal()

def main(args=None):
    rclpy.init(args=args)
    node = MoveAroundNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
