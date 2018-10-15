#ifndef CAMERA_CALIBRATION_IMAGE_PROC_H
#define CAMERA_CALIBRATION_IMAGE_PROC_H

#include <ros/ros.h>
#include <boost/bind.hpp>
#include <string>
#include <memory>
#include <map>
#include <ros/console.h>
#include <math.h>
#include <vector>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>

//For transform support
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>



namespace camera_calibration_virtana{
  class ImageProcessor{

    public:

      ImageProcessor();

      ~ImageProcessor();

      void SetCamInfo(const sensor_msgs::CameraInfoConstPtr& info);

      void ImageCB(const sensor_msgs::ImageConstPtr& msg);

      void InitialiseSubscribers();

    private:
      ros::NodeHandle private_nh_;
      ros::Subscriber cam_info_sub_;
      image_transport::ImageTransport it_;
      image_transport::Subscriber image_sub_;
      //image_transport::Publisher image_pub_;
      std::string OPENCV_WINDOW;

      sensor_msgs::CameraInfo camera_info_;

      tf::TransformListener listener_;
      tf::StampedTransform cam_to_calibration_;
      tf::StampedTransform world_to_cam_;
      tf::TransformBroadcaster transform_broadcaster_;

      std::unique_ptr<cv::FileStorage> fs_;



      float cube_dimensions_[3]; // x, y ,z format

  };
};

#endif