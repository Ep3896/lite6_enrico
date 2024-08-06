#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt

class GoalPlotter(Node):
    def __init__(self):
        super().__init__('goal_plotter')
        self.subscription = self.create_subscription(Float32MultiArray, 'goal_coordinates', self.plot_callback, 10)
        self.x_data, self.y_data = [], []
        self.setup_plot()

    def setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'ro')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.axhline(0, color='black',linewidth=0.5)
        self.ax.axvline(0, color='black',linewidth=0.5)
        self.ax.grid(color='gray', linestyle='-', linewidth=0.5)
        plt.show()

    def plot_callback(self, msg):
        goal_x, goal_y, goal_z = msg.data
        goal_x = goal_x + 0.1
        goal_y = goal_y
        self.x_data.append(goal_x)
        self.y_data.append(goal_y)
        self.line.set_data(self.x_data, self.y_data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    node = GoalPlotter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
