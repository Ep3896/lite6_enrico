import socket
import struct
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI
arm = XArmAPI('192.168.1.190', debug=True)

def bytes_to_fp32(bytes_data, is_big_endian=False):
    """
    bytes to float
    :param bytes_data: bytes
    :param is_big_endian: is big endian or not，default is False.
    :return: fp32
    """
    return struct.unpack('>f' if is_big_endian else '<f', bytes_data)[0]

def bytes_to_fp32_list(bytes_data, n=0, is_big_endian=False):
    """
    bytes to float list
    :param bytes_data: bytes
    :param n: quantity of parameters need to be converted，default is 0，all bytes converted.
    :param is_big_endian: is big endian or not，default is False.
    :return: float list
    """
    ret = []
    count = n if n > 0 else len(bytes_data) // 4
    for i in range(count):
        ret.append(bytes_to_fp32(bytes_data[i * 4: i * 4 + 4], is_big_endian))
    return ret

def bytes_to_u32(data):
    data_u32 = data[0] << 24 | data[1] << 16 | data[2] << 8 | data[3]
    return data_u32

robot_ip = '192.168.1.190' # IP of controller
robot_port = 30003 # Port of controller

# create socket to connect controller
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setblocking(True)
sock.settimeout(1)
sock.connect((robot_ip, robot_port))

arm.set_report_tau_or_i(tau_or_i=0)  #The acquired data is the data from J1 to J7

while True:
    data = sock.recv(4)
    length = bytes_to_u32(data)
    data += sock.recv(length - 4)
    joint_data = bytes_to_fp32_list(data[59:87])
    print(joint_data)
