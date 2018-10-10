#include <cmath>
#include <cstdio>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

// Definitions for the dimensions of the calibration board
static const int Number_of_internal_corners_x = 7;
static const int Number_of_internal_corners_y = 6;
static const double Block_dimension = 0.02; // This is the side of a block in m.

// Definitions for the initial estimates for the values desired to be optimised
static const double initial_t1 = 0.01;
static const double initial_t2 = 0.01;
static const double initial_t3 = 0.59;

static const double initial_angle_x = -3.14;
static const double initial_angle_y = 0.0;
static const double initial_angle_z = 0.0;

static const double initial_fx = 1050;
static const double initial_fy = 1050;
static const double initial_cx = 320;
static const double initial_cy = 240;

// Data Structure definition
struct ReProjectionResidual
{ 
	// Put the actual parameters that will not be optimised right here (this would be the intrinsics,the uv pixel points and board_T_cam).
	ReProjectionResidual(const double *pixel_points, const double *board_T_point)
	{
		// Initialise the pixel points
		observed_pixel_points_[0] = pixel_points[0]; // u
		observed_pixel_points_[1] = pixel_points[1]; // v

		// Initialise the board_T_point
		board_T_point_[0] = board_T_point[0]; // X
		board_T_point_[1] = board_T_point[1]; // Y
		board_T_point_[2] = board_T_point[2]; // Z
	}

	// For this template, put all of the parameters to optimise and the residual here.
	template <typename T>
	bool operator()(const T* const camera_extrinsics, const T* const camera_intrinsics, T* residuals)
	const
	{
		// compute projective coordinates: p = RX + t.
        // camera_extrinsics[0, 1, 2]: axis-angle
        // camera_extrinsics[3, 4, 5]: translation
		const T R_angle_axis[3] = {T(camera_extrinsics[0]), T(camera_extrinsics[1]), T(camera_extrinsics[2])};
		const T point[3] = {T(board_T_point_[0]), T(board_T_point_[1]), T(board_T_point_[2])};
		T p[3];

		// AngleAxisRotatePoint used to rotate the board_T_point about the axis of rotation which is set 
		// as the R component of the camera extrinsic matric after the rotation matrix to angle axis conversion
		ceres::AngleAxisRotatePoint(R_angle_axis, point, p);

		// AngleAxisRotatePoint gives the "RX" therefore it must be translated by "t" (from camera extrinsics) to give "p".
		p[0] += camera_extrinsics[3]; // X component of camera to calibration point
		p[1] += camera_extrinsics[4]; // Y component of camera to calibration point
		p[2] += camera_extrinsics[5]; // Z component of camera to calibration point

		// The projected pixel coordinates would now be computed. (for now i am not including distortion)
		T up = p[0] / p[2];
		T vp = p[1] / p[2];

		// The projected uv pixel values are calculated here based on the current camera intrinsics and extrinsics
		T projected_u = up * camera_intrinsics[0] + camera_intrinsics[2];
		T projected_v = vp * camera_intrinsics[1] + camera_intrinsics[3];

		// The residuals are calculated here
		residuals[0] = projected_u - T(observed_pixel_points_[0]);
		residuals[1] = projected_v - T(observed_pixel_points_[1]);

		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction *Create(const double *pixel_points, const double *board_T_point)
	{
		return (new ceres::AutoDiffCostFunction<ReProjectionResidual, 2, 6, 4>(
			new ReProjectionResidual(pixel_points, board_T_point)));
	}

	// Declare struct variables
	private:
		double observed_pixel_points_[3];
		double board_T_point_[4];
};

int main()
{
    // Note: This matrix may have to be a multidimensional one to facilitate the images.ADJ_OFFSET_SINGLESHOT
    //cv::Mat image = cv::imread("../Calibration_images/2018-10-09-102628.jpg", 1);
	cv::Mat image = cv::imread("../Calibration_Board_tests/2018-10-10-125321.jpg", 1);

    //cv::Size patternsize(7,6);
    cv::Size patternsize = cv::Size(7, 6); // (width, height) or (columns, rows) or (Y, X)
    std::vector<cv::Point2f> corners; //This will be filled by the detected corners

    bool chessboard_found = cv::findChessboardCorners(image, patternsize, corners);

    if(chessboard_found)
    {
        std::cout << "The Chessboard was found within the image. \r\n";
    }
    else
    {
        std::cout << "The Chessboard was not found within the image. \r\n";
    }

/*
    cv::drawChessboardCorners(image, patternsize, corners, chessboard_found);

    cv::namedWindow("Window");
    while(true)
    {
        cv::imshow("Window", image);
        cv::waitKey(33);
    }
*/


    // The "corners" variable is a vector which holds the pixel coordinates of the calibration points.
    // For my sake, this is essentially the matrix A. You would need the matrix D next in order to optimise the 
    // camera intrinsics and extrinsics.

    // The calibration board is generated here and is regarded as matrix D
	std::vector<std::vector<double>> matrix_D;

	for (int y = 0; y != Number_of_internal_corners_y; y++)
	{
		for (int x = 0; x != Number_of_internal_corners_x; x++)
		{
			std::vector<double> point;

			point.push_back(x * Block_dimension);
			point.push_back(y * Block_dimension);
			point.push_back(0 * Block_dimension);

			matrix_D.push_back(point);
		}
	}

/*
    // Display the contents of the matrix D in the format xyz.

    int count = 0;
	for (int y = 0; y != matrix_D.size(); y++)
	{
        std::cout << "Point number: " << count++ << "\r\n";
        std::cout << "XYZ: ";
		for (int x = 0; x != matrix_D[y].size(); x++)
		{            
			std::cout << matrix_D[y][x];
		}
		std::cout << "\r\n";
	}  

    // Display the corners which is basically the matrix A.
    count = 0;
    for (std::vector<cv::Point2f>::iterator i = corners.begin(); i != corners.end(); i++)
	{
        std::cout << "Point number: " << count++ << "\r\n";
		std::cout << "pixel u: " << i->x << "\r\n";
        std::cout << "pixel v: " << i->y << "\r\n";
        
		std::cout << "\r\n";
	}
*/
    ///////////// The optimisation code begins from here /////////////

    // Set the initial values for the mutable parameters.
    double camera_intrinsics[4] = {initial_fx, initial_fy, initial_cx, initial_cy};
    double camera_extrinsics[6] = {initial_angle_x, initial_angle_y, initial_angle_z, initial_t1, initial_t2, initial_t3};

    // Begin building the problem
    ceres::Problem problem;

    int count = 0;
    for (std::vector<cv::Point2f>::iterator i = corners.begin(); i != corners.end(); i++)
	{
        double board_T_point[3] = {matrix_D[count][0], matrix_D[count][1], matrix_D[count][2]}; //This is the calibration point of the format: X,Y,Z
		double image_pixels[2] = {i->x, i->y}; //This is the image point of the format: u, v.

        // std::cout << "Calibration point number: " << count << "\r\n";
        // std::cout << "XYZ: " << matrix_D[count][0] << ", " << matrix_D[count][1] << ", " << matrix_D[count][2] << "\r\n\n";

        // std::cout << "Pixel point number: " << count << "\r\n";
		// std::cout << "pixel u: " << image_pixels[0] << "\r\n";
        // std::cout << "pixel v: " << image_pixels[1] << "\r\n\n";

		ceres::CostFunction* cost_function = ReProjectionResidual::Create(image_pixels, board_T_point);
		problem.AddResidualBlock(cost_function, NULL, camera_extrinsics, camera_intrinsics); 

        count++;
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 10000;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << "\r\n";

	// The results from the optimization is displayed here

	// Camera Extrinsics results
	std::cout << "CAMERA EXTRINSICS RESULTS: \r\n\n";
	std::cout << "Initial axis-angle x: " << initial_angle_x << "\r\n";
	std::cout << "Final   axis-angle x: " << camera_extrinsics[0]  << "\r\n\n";

	std::cout << "Initial axis-angle y: " << initial_angle_y << "\r\n";
	std::cout << "Final   axis-angle y: " << camera_extrinsics[1]  << "\r\n\n";

	std::cout << "Initial axis-angle z: " << initial_angle_z << "\r\n";
	std::cout << "Final   axis-angle z: " << camera_extrinsics[2]  << "\r\n\n";

	std::cout << "Initial t1: " << initial_t1 << "\r\n";
	std::cout << "Final   t1: " << camera_extrinsics[3]  << "\r\n\n";

	std::cout << "Initial t2: " << initial_t2 << "\r\n";
	std::cout << "Final   t2: " << camera_extrinsics[4]  << "\r\n\n";

	std::cout << "Initial t3: " << initial_t3 << "\r\n";
	std::cout << "Final   t3: " << camera_extrinsics[5]  << "\r\n\n";


	// Camera Intrinsics results
	std::cout << "CAMERA INTRINSICS RESULTS: \r\n\n";
	std::cout << "Initial fx: " << initial_fx << "\r\n";
	std::cout << "Final   fx: " << camera_intrinsics[0]  << "\r\n\n";

	std::cout << "Initial fy: " << initial_fy << "\r\n";
	std::cout << "Final   fy: " << camera_intrinsics[1]  << "\r\n\n";

	std::cout << "Initial cx: " << initial_cx << "\r\n";
	std::cout << "Final   cx: " << camera_intrinsics[2]  << "\r\n\n";

	std::cout << "Initial cy: " << initial_cy << "\r\n";
	std::cout << "Final   cy: " << camera_intrinsics[3]  << "\r\n\n";

/*
    cv::drawChessboardCorners(image, patternsize, corners, chessboard_found);

    cv::namedWindow("Window");
    while(true)
    {
        cv::imshow("Window", image);
        cv::waitKey(33);
    }
*/	

	// This step is not necessary but it is being done to determine if the optimised camera intrinsics and
	// extrinsics returns correct pixel points for the calibration ponits.

	double matrix_B[3][3] = {{camera_intrinsics[0], 0, camera_intrinsics[2]}, {0, camera_intrinsics[1], camera_intrinsics[3]}, {0, 0, 1} };

	// For matrix_C I have to convert the axis angle into the 9x9 rotation matrix.
	double angle_axis[3] = {camera_extrinsics[0], camera_extrinsics[1],  camera_extrinsics[2]};
	double rotation_matrix[9];
	ceres::AngleAxisToRotationMatrix(&angle_axis[0], &rotation_matrix[0]);
	double matrix_C[3][4] = {{rotation_matrix[0], rotation_matrix[3], rotation_matrix[6], camera_extrinsics[3]}, {rotation_matrix[1], rotation_matrix[4], rotation_matrix[7], camera_extrinsics[4]}, {rotation_matrix[2], rotation_matrix[5], rotation_matrix[8], camera_extrinsics[5]}};

	// The matrix_D is recomputed here in a format acceptable for the matrix multiplication.
	double matrix_D_[4][Number_of_internal_corners_y * Number_of_internal_corners_x];

	count = 0;
	for (int y = 0; y != Number_of_internal_corners_y; y++)
	{
		for (int x = 0; x != Number_of_internal_corners_x; x++)
		{
			matrix_D_[0][count] = x * Block_dimension;
			matrix_D_[1][count] = y * Block_dimension;
			matrix_D_[2][count] = 0 * Block_dimension;
			matrix_D_[3][count] = 1;

			count++;
		}	
	}

	// The multidimensional arrays are converted into OpenCV matrices so that the multiplication can be done.
	cv::Mat matrix_B_ = cv::Mat(3, 3, CV_64FC1, matrix_B);
	cv::Mat matrix_C_ = cv::Mat(3, 4, CV_64FC1, matrix_C);
	cv::Mat matrix_D__ = cv::Mat(4, Number_of_internal_corners_y * Number_of_internal_corners_x, CV_64FC1, matrix_D_);

	cv::Mat matrix_A = matrix_B_ * matrix_C_ * matrix_D__;
	//std::cout << "matrix_A: \r\n" << matrix_A;
	//std::cout << "\r\n";

	// Display the real and calculated matrix A here. 
	double true_matrix_A[2][Number_of_internal_corners_y * Number_of_internal_corners_x];

    count = 0;
    for (std::vector<cv::Point2f>::iterator i = corners.begin(); i != corners.end(); i++)
	{
		true_matrix_A[0][count] = i->x;
		true_matrix_A[1][count] = i->y;

		count++;
	}

	double calculated_matrix_A[4][Number_of_internal_corners_y * Number_of_internal_corners_x];

	for (int y = 0; y != 3; y++)
	{
		for (int x = 0; x != Number_of_internal_corners_y * Number_of_internal_corners_x; x++)
		{
			calculated_matrix_A[y][x] = matrix_A.at<double>(y,x);
		}	
	}

	// Display the 2 matrix_A
	for (int i = 0; i != Number_of_internal_corners_y * Number_of_internal_corners_x; i++)
	{
		std::cout << "Point number: " << i << "\r\n";
		std::cout << "Real: \r\n";
		std::cout << "Pixel u: " << true_matrix_A[0][i] << "\r\n";
		std::cout << "Pixel v: " << true_matrix_A[1][i] << "\r\n";

		std::cout << "Calculated: \r\n";
		std::cout << "Pixel u: " << (calculated_matrix_A[0][i] / calculated_matrix_A[2][i]) << "\r\n";
		std::cout << "Pixel v: " << (calculated_matrix_A[1][i] / calculated_matrix_A[2][i]) << "\r\n\n";
	}	
 
    return 0;
}