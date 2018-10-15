// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A minimal, self-contained bundle adjuster using Ceres, that reads
// files from University of Washington' Bundle Adjustment in the Large dataset:
// http://grail.cs.washington.edu/projects/bal
//
// This does not use the best configuration for solving; see the more involved
// bundle_adjuster.cc file for details.
#include <cmath>
#include <cstdio>
#include <iostream>
#include <opencv2/core/core.hpp>

#include <ros/ros.h>
#include "ceres/ceres.h"

struct ReProjectionError {
  ReProjectionError(double observed_x, double observed_y, const double *point)
      : observed_x_(observed_x), observed_y_(observed_y)
      {
        point_[0] = point[0];
        point_[1] = point[1];
        point_[2] = point[2];
      }
  template <typename T>
  bool operator()(const T* const camera, T* residuals)
  const
  {
    //TODO: Insert the proper function using the camera parameters of size 8

    //camera[0,1,2] = fx, cx, Tx, camera[3,4,5] = fy, cy, Ty camera[6,7] = k1, k2

    T xp = (camera[0]*(T)point_[0] + camera[1]*(T)point_[2] + camera[2])/(T)point_[2];
    T yp = (camera[3]*(T)point_[1] + camera[4]*(T)point_[2] + camera[5])/(T)point_[2];

    T projected_x = xp + xp*(camera[6]*(xp*xp + yp*yp) + camera[7]*(xp*xp + yp*yp)*(xp*xp + yp*yp));
    T projected_y = yp + yp*(camera[6]*(xp*xp + yp*yp) + camera[7]*(xp*xp + yp*yp)*(xp*xp + yp*yp));


    residuals[0] = (T)observed_x_ - projected_x;

    residuals[1] = (T)observed_y_ - projected_y;

    return true;
  }
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y,
                                     const double *point) {
    return (new ceres::AutoDiffCostFunction<ReProjectionError, 2, 8>(
                new ReProjectionError(observed_x, observed_y, point)));
  }
  double observed_x_;
  double observed_y_;
  double point_[3];
};
int main(int argc, char** argv) {
  //Load data
  cv::FileStorage fs2("src/camera_calibration_virtana/calibration_points4.xml", cv::FileStorage::READ);

  cv::FileNode points = fs2["points"];

  std::vector<std::vector<float>> xyz;

  std::vector<std::vector<float>> pixels;

  ROS_INFO("Points size: %lu", points.size());

  for(cv::FileNodeIterator fi = points.begin(); fi != points.end(); fi++)
  {
    for(int i = 0; i < ((*fi)["corners"]).size(); i++)
    {
      std::vector<float> temp1;
      temp1.push_back((*fi)["corners"][i]["transform"]["X"]);
      temp1.push_back((*fi)["corners"][i]["transform"]["Y"]);
      temp1.push_back((*fi)["corners"][i]["transform"]["Z"]);
      xyz.push_back(temp1);

      std::vector<float> temp2;
      temp2.push_back((*fi)["corners"][i]["pixel"]["x"]);
      temp2.push_back((*fi)["corners"][i]["pixel"]["y"]);

      pixels.push_back(temp2);
    }
  }

//  ROS_INFO("Points size: %lu , %lu", xyz.size(), pixels.size());

  //Start to build the problem

//Initial fx: 0.000000 --> Final fx: 762.769956
//Initial cx: 0.000000 --> Final cx: 639.407253
//Initial Tx: 0.000000 --> Final Tx: 0.007769
//Initial fy: 0.000000 --> Final fy: 763.024999
//Initial cy: 0.000000 --> Final cy: 639.379343
//Initial Ty: 0.000000 --> Final Ty: 0.034106
//Initial k1: 0.000000 --> Final k1: 0.000000
//Initial k2: 0.000000 --> Final k2: -0.000000


//  double camera[] = {762.769956, 639.407253, 0.007769, 763.024999, 639.379343, 0.034106, 0.00, 0.00};
  double camera[] = {900, 500, 0.00, 900, 500, 0.0, 0.00, 0.00};


  double fx_1 = camera[0], cx_1 = camera[1], tx_1 = camera[2], fy_1 = camera[3], cy_1 = camera[4], ty_1 = camera[5], k1 = camera[6], k2 = camera[7];

  ceres::Problem problem;

  for (int i = 0; i < pixels.size(); ++i) {

    double point[3] = {(xyz[i])[0],(xyz[i])[1],(xyz[i])[2]};

    point[0] = (xyz[i])[0];
    point[1] = (xyz[i])[1];
    point[2] = (xyz[i])[2];

    double current_x = (pixels[i])[0];
    double current_y = (pixels[i])[1];

    ceres::CostFunction* cost_function = ReProjectionError::Create(current_x, current_y, point);

    problem.AddResidualBlock(cost_function,NULL /* squared loss */,camera);

  }
  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 400;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  printf("Initial fx: %f --> Final fx: %f\n", fx_1, camera[0]);
  printf("Initial cx: %f --> Final cx: %f\n", cx_1, camera[1]);
  printf("Initial Tx: %f --> Final Tx: %f\n", tx_1, camera[2]);
  printf("Initial fy: %f --> Final fy: %f\n", fy_1, camera[3]);
  printf("Initial cy: %f --> Final cy: %f\n", cy_1, camera[4]);
  printf("Initial Ty: %f --> Final Ty: %f\n", ty_1, camera[5]);
  printf("Initial k1: %f --> Final k1: %f\n", k1, camera[6]);
  printf("Initial k2: %f --> Final k2: %f\n", k2, camera[7]);

  double acc_x = 0.0, acc_y = 0.0;

  for (int i = 0; i < xyz.size(); ++i)
  {
    double point[3] = {(xyz[i])[0],(xyz[i])[1],(xyz[i])[2]};

    double xp = (camera[0]*point[0] + camera[1]*point[2] + camera[2])/point[2];
    double yp = (camera[3]*point[1] + camera[4]*point[2] + camera[5])/point[2];

    double projected_x = xp + xp*(camera[6]*(xp*xp + yp*yp) + camera[7]*(xp*xp + yp*yp)*(xp*xp + yp*yp));
    double projected_y = yp + yp*(camera[6]*(xp*xp + yp*yp) + camera[7]*(xp*xp + yp*yp)*(xp*xp + yp*yp));

    if((projected_x - (pixels[i])[0] > 8.0) || (projected_y - (pixels[i])[1] > 8.0))
    {
      ROS_WARN("Error: x: %f, y: %f. Transform: < x: %f, y: %f ,z: %f>", projected_x - (pixels[i])[0], projected_y - (pixels[i])[1]
             , point[0], point[1], point[2]);
    }
    acc_x+= std::fabs(projected_x - (pixels[i])[0]);
    acc_y+= std::fabs(projected_y - (pixels[i])[1]);
  }

  ROS_INFO("Average pixel difference: x: %f, y: %f", acc_x/(double)xyz.size(), acc_y/(double)xyz.size());

  return 0;
}
