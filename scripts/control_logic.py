import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from enum import Enum
import graphviz

class States(Enum):
    DETECTING_CARDS = 1
    APPROACHING_CARDS = 2
    MOVE_AROUND_LR = 3
    ALIGNMENT_PHASE = 4
    CATCHING_PHASE = 5
    CONTROL_IF_CARD_TAKEN = 6
    DETECTING_POS = 7
    READY_STATE = 8

class FSMNode(Node):
    def __init__(self):
        super().__init__('fsm_node')
        self.state = States.DETECTING_CARDS
        self.create_subscription(Bool, 'start_signal', self.start_callback, 10)
        self.create_subscription(Bool, 'card_detected', self.card_detected_callback, 10)
        self.create_subscription(Bool, 'card_approached', self.card_approached_callback, 10)
        self.create_subscription(Bool, 'camera_detected', self.camera_detected_callback, 10)
        self.create_subscription(Bool, 'card_caught', self.card_caught_callback, 10)
        self.create_subscription(Bool, 'card_taken', self.card_taken_callback, 10)
        self.publisher_ = self.create_publisher(String, 'fsm_state', 10)
        self.control_pub = self.create_publisher(Bool, 'control_commands', 10)
        self.timer_ = self.create_timer(1.0, self.timer_callback)
        self.create_state_diagram()

    def start_callback(self, msg):
        if msg.data and self.state == States.DETECTING_CARDS:
            self.transition_to(States.APPROACHING_CARDS)

    def card_detected_callback(self, msg):
        if msg.data and self.state == States.DETECTING_CARDS:
            self.transition_to(States.APPROACHING_CARDS)

    def card_approached_callback(self, msg):
        if msg.data and self.state == States.APPROACHING_CARDS:
            self.transition_to(States.MOVE_AROUND_LR)

    def camera_detected_callback(self, msg):
        if msg.data and self.state in [States.MOVE_AROUND_LR, States.ALIGNMENT_PHASE]:
            self.transition_to(States.ALIGNMENT_PHASE)
        elif not msg.data and self.state == States.ALIGNMENT_PHASE:
            self.transition_to(States.MOVE_AROUND_LR)

    def card_caught_callback(self, msg):
        if msg.data and self.state == States.ALIGNMENT_PHASE:
            self.transition_to(States.CATCHING_PHASE)

    def card_taken_callback(self, msg):
        if msg.data and self.state == States.CATCHING_PHASE:
            self.transition_to(States.CONTROL_IF_CARD_TAKEN)
        else:
            self.transition_to(States.DETECTING_CARDS)

    def transition_to(self, new_state):
        self.state = new_state
        self.get_logger().info(f'Transitioned to {self.state}')
        self.publish_state()

    def publish_state(self):
        msg = String()
        msg.data = str(self.state)
        self.publisher_.publish(msg)
        
        # Send control commands based on the state
        control_msg = Bool()
        if self.state == States.DETECTING_CARDS:
            control_msg.data = True  # For example, start detection
        elif self.state == States.APPROACHING_CARDS:
            control_msg.data = True  # For example, start approaching
        elif self.state == States.MOVE_AROUND_LR:
            control_msg.data = True  # For example, start moving left/right
        elif self.state == States.ALIGNMENT_PHASE:
            control_msg.data = True  # For example, start aligning
        elif self.state == States.CATCHING_PHASE:
            control_msg.data = True  # For example, start catching
        elif self.state == States.CONTROL_IF_CARD_TAKEN:
            control_msg.data = True  # For example, check if card is taken
        elif self.state == States.DETECTING_POS:
            control_msg.data = True  # For example, start detecting POS
        elif self.state == States.READY_STATE:
            control_msg.data = True  # For example, ready state
        
        self.control_pub.publish(control_msg)

    def timer_callback(self):
        # This can be used to monitor and handle timeout conditions
        pass

    def create_state_diagram(self):
        dot = graphviz.Digraph(comment='Finite State Machine')
        dot.node('DETECTING_CARDS', 'Detecting Cards')
        dot.node('APPROACHING_CARDS', 'Approaching Cards')
        dot.node('MOVE_AROUND_LR', 'Move around (Left/right Movement)')
        dot.node('ALIGNMENT_PHASE', 'Alignment Phase')
        dot.node('CATCHING_PHASE', 'Catching Phase')
        dot.node('CONTROL_IF_CARD_TAKEN', 'Control if the Card was taken')
        dot.node('DETECTING_POS', 'Detecting POS')
        dot.node('READY_STATE', 'Go to ready state')

        dot.edge('DETECTING_CARDS', 'APPROACHING_CARDS', label='Found one?')
        dot.edge('APPROACHING_CARDS', 'MOVE_AROUND_LR', label='Yes')
        dot.edge('MOVE_AROUND_LR', 'ALIGNMENT_PHASE', label='Is the camera still Detected?')
        dot.edge('ALIGNMENT_PHASE', 'CATCHING_PHASE', label='Yes')
        dot.edge('ALIGNMENT_PHASE', 'MOVE_AROUND_LR', label='No')
        dot.edge('CATCHING_PHASE', 'CONTROL_IF_CARD_TAKEN')
        dot.edge('CONTROL_IF_CARD_TAKEN', 'DETECTING_POS', label='Yes')
        dot.edge('CONTROL_IF_CARD_TAKEN', 'DETECTING_CARDS', label='No')
        dot.edge('DETECTING_POS', 'READY_STATE')
        dot.edge('READY_STATE', 'DETECTING_CARDS')

        dot.render('fsm_diagram', format='png')
        self.get_logger().info("FSM diagram created as fsm_diagram.png")

def main(args=None):
    rclpy.init(args=args)
    node = FSMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
