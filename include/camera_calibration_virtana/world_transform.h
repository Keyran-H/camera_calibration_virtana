#ifndef CAMERA_CALIBRATION_VIRTANA_WORLD_TRANSFORM_H
#define CAMERA_CALIBRATION_VIRTANA_WORLD_TRANSFORM_H

#include <ros/ros.h>
#include <boost/bind.hpp>
#include <string>
#include <memory>
#include <map>
#include <ros/console.h>
#include <math.h>
#include <vector>

//For transform support
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>

#include <gazebo_msgs/LinkState.h>
#include <gazebo_msgs/LinkStates.h>

#include <dynamic_reconfigure/server.h>
#include <camera_calibration_virtana/calibrationConfig.h>

namespace camera_calibration_virtana{

  class WorldTransform{
    public:
        WorldTransform();
        ~WorldTransform();

        void initialiseSubscribers();

        void initialisePublishers();

        void JointCallback(const camera_calibration_virtana::calibrationConfig &config, uint32_t level);

        void UpdateTfTree(const gazebo_msgs::LinkStatesConstPtr& joints_msg);

     private:
        ros::NodeHandle private_nh_;
        ros::Publisher link_state_pub_;
        ros::Subscriber link_state_sub_;

        ros::Timer tf_timer_;

        tf::TransformBroadcaster transform_broadcaster_;

  };
};

#endif