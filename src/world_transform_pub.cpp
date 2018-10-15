#include "camera_calibration_virtana/world_transform.h"


int main(int argc, char** argv) {

  ros::init(argc, argv, "world_tf_node");

  ros::NodeHandle nh;


  while(ros::Time::now() == ros::Time()){
    ros::Duration(0.1).sleep();
  }

  camera_calibration_virtana::WorldTransform world_tf;

  dynamic_reconfigure::Server<camera_calibration_virtana::calibrationConfig> server;
  dynamic_reconfigure::Server<camera_calibration_virtana::calibrationConfig>::CallbackType f;

  f = boost::bind(&camera_calibration_virtana::WorldTransform::JointCallback, &world_tf, _1, _2);
  server.setCallback(f);

  ROS_INFO("Spinning node");
  ros::spin();
  return 0;
}