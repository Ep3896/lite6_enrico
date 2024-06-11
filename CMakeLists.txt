cmake_minimum_required(VERSION 3.5)
project(lite6_enrico)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python modules
ament_python_install_package(scripts)

# Install Python scripts
install(DIRECTORY scripts
  DESTINATION lib/${PROJECT_NAME})


# Install config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/)

ament_package()
