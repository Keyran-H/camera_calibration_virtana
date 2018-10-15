#include "camera_calibration_virtana/world_transform.h"

namespace camera_calibration_virtana{
  WorldTransform::WorldTransform():private_nh_("~")
    {
      initialisePublishers();
      initialiseSubscribers();
    };

  WorldTransform::~WorldTransform()
    {

    };

  void WorldTransform::initialiseSubscribers()
    {
      link_state_sub_ = ros::NodeHandle("gazebo").subscribe("link_states", 1000,&WorldTransform::UpdateTfTree, this);
    };

  void WorldTransform::initialisePublishers()
    {
      link_state_pub_ = ros::NodeHandle("gazebo").advertise<gazebo_msgs::LinkState>("set_link_state", 10);
    };

  void WorldTransform::JointCallback(const camera_calibration_virtana::calibrationConfig &config, uint32_t level)
    {
      gazebo_msgs::LinkState calibration_state;

      tf::Quaternion qt;
      tf::Vector3 vt;

      qt.setRPY(config.roll, config.pitch, config.yaw);
      vt = tf::Vector3(config.x_trans, config.y_trans, config.z_trans);

      tf::Pose pt(qt, vt);

      tf::poseTFToMsg(pt, calibration_state.pose);

      //TODO: make these strings params or also part of the dynamic reconfigure

      calibration_state.link_name = "calibration_target_link";

      calibration_state.reference_frame = "world";

      link_state_pub_.publish(calibration_state);

    };

  void WorldTransform::UpdateTfTree(const gazebo_msgs::LinkStatesConstPtr& joints_msg)
    {
      std::ptrdiff_t link_pos;
      std::ptrdiff_t cam_pos;

      if((std::find(std::begin(joints_msg->name), std::end(joints_msg->name), "checkerboard::calibration_target_link")
          != std::end(joints_msg->name)) &&
          (std::find(std::begin(joints_msg->name), std::end(joints_msg->name), "camera::camera_link")
          != std::end(joints_msg->name)))
      {

        link_pos = std::find(std::begin(joints_msg->name), std::end(joints_msg->name), "checkerboard::calibration_target_link")
                              - std::begin(joints_msg->name);
        cam_pos = std::find(std::begin(joints_msg->name), std::end(joints_msg->name), "camera::camera_link")
                              - std::begin(joints_msg->name);
      }
      else
      {
        ROS_FATAL("NO LINK FOUND THAT MATCHES REQUESTED LINK!");
        ros::shutdown();
      }

      tf::Pose pt;

      tf::poseMsgToTF(joints_msg->pose[link_pos], pt);

      transform_broadcaster_.sendTransform(tf::StampedTransform(pt, ros::Time::now(), "world" , "calibration_target_link"));

      tf::poseMsgToTF(joints_msg->pose[cam_pos], pt);

      transform_broadcaster_.sendTransform(tf::StampedTransform(pt, ros::Time::now(), "world" , "camera_link"));

    };
}