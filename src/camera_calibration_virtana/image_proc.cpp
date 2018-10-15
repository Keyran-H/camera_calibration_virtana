#include "camera_calibration_virtana/image_proc.h"

namespace camera_calibration_virtana{

  ImageProcessor::ImageProcessor():private_nh_("~"), it_(ros::NodeHandle())
  {
    cube_dimensions_[0] = 1.0;
    cube_dimensions_[1] = 1.0;
    cube_dimensions_[2] = 1.0;
    OPENCV_WINDOW = "Image Window";

    fs_.reset(new cv::FileStorage ("src/camera_calibration_virtana/calibration_points4.xml", cv::FileStorage::WRITE));

    InitialiseSubscribers();
    *fs_ << "points" << "[";
  };

  ImageProcessor::~ImageProcessor()
  {
    *fs_ << "]";
    fs_->release();
  };

  void ImageProcessor::SetCamInfo(const sensor_msgs::CameraInfoConstPtr& info)
  {
    camera_info_ = *info;
  };

  void ImageProcessor::InitialiseSubscribers()
  {
    //TODO: Maybe a param here.
    cam_info_sub_ = ros::NodeHandle().subscribe("/camera1/camera_info", 1, &ImageProcessor::SetCamInfo, this);
    image_sub_ = it_.subscribe("/camera1/image_raw", 1, &ImageProcessor::ImageCB, this);

  };

  void ImageProcessor::ImageCB(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    std::vector<cv::Point2f> corners;

    if((ros::Time::now().toSec() - msg->header.stamp.toSec()) > 1.0)
    {
      ROS_WARN("Time mismatch. Message time: %f, ROS time: %f Discarding current msg",
                msg->header.stamp.toSec(), ros::Time::now().toSec());
      return;
    }

    listener_.lookupTransform("image_link", "calibration_target_link", msg->header.stamp, cam_to_calibration_);


    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }



    bool patternfound = cv::findChessboardCorners(cv_ptr->image, cvSize(7,6), corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FAST_CHECK);

    if(patternfound)
    {

      ROS_INFO("Found board");
      ROS_INFO("Number of corners found: , %lu", corners.size());
      /*for(std::vector<cv::Point2f>::iterator corners_iter = corners.begin(); corners_iter != corners.end(); corners_iter++)
      {

         ROS_INFO("Points on Image <x,y>: <%f,%f> \n", corners_iter->x, corners_iter->y);


      }*/
      *fs_ << "{:";

//      *fs_ <<"time" << msg->header.stamp.toSec();

      *fs_ <<"corners" << "[";

      for(int i = 0; i < corners.size(); i++)
      {
        //Generate the XYZ of the matched corner based on the transform from cam_t_calib and the positions of the
        //points on the face.

        float x,y,z;
        x = cube_dimensions_[0]/2.0;
        y = -cube_dimensions_[1]*(0.5 - 0.2 - (i%7)*(0.1));
        z = cube_dimensions_[2]*(0.5 - 0.2 - (i/7)*(0.1));

        //TODO: For each match write the corresponding points (x,y) to transform(x,y,z) and repeat until the end of the calibration. NOTE THE TRANSFORM MUST BE IN THE IMAGE/CAMERA frame

        tf::Transform image_T_cam;


        tf::Vector3 trans = cam_to_calibration_*tf::Vector3(x,y,z);


        ROS_INFO("[%f,%f,%f,1]'", trans.getX(), trans.getY(), trans.getZ());

        //        float pixel_x = ((camera_info_.P[0]/10.0)*trans.getX() + (camera_info_.P[2]/10.0)*trans.getZ() + camera_info_.P[3])/trans.getZ();
        //        float pixel_y = ((camera_info_.P[5]/10.0)*trans.getY() + (camera_info_.P[6]/10.0)*trans.getZ() + camera_info_.P[7])/trans.getZ();

        //        ROS_INFO("Expected point (x:%f,y:%f), Actual point (x:%f,y:%f)", pixel_x, pixel_y, corners[i].x, corners[i].y);

        *fs_ << "{:";
        *fs_ << "pixel"<< "{:" << "x" << (float)corners[i].x << "y" << (float)corners[i].y << "}";
        *fs_ << "transform" <<"{:" << "X" << (float)trans.getX() << "Y" << (float)trans.getY() << "Z"<< (float)trans.getZ()<< "}";
        *fs_ << "}";

      }

      *fs_ << "]";

      *fs_ << "}";


              cv::drawChessboardCorners(cv_ptr->image, cvSize(7,6), cv::Mat(corners), patternfound);

    }

          cv::imshow(OPENCV_WINDOW, cv_ptr->image);
          cv::waitKey(3);
  };

}