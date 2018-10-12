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

/*The code MUST be edited to reflect the initial estimates based on the number of images for the camera extrinsics*/

// Definitions for the dimensions of the calibration board
static const int Number_of_internal_corners_x = 7;
static const int Number_of_internal_corners_y = 6;
static const double Block_dimension = 0.02; // This is the side of a block in m.

// Definitions for the initial estimates for the values desired to be optimised
static const double initial_t1 = 0.01;
static const double initial_t2 = 0.01;
static const double initial_t3 = 0.59;

static const double initial_angle_x = 0.0;
static const double initial_angle_y = 0.0;
static const double initial_angle_z = 0.0;

static const double initial_fx = 950;
static const double initial_fy = 950;
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
	unsigned int count = 0;
	unsigned int num_of_images = 0;
	unsigned int total_chessboards_detected = 0;
	cv::VideoCapture cap("../Calibration_images_numbered/image%02d.jpg");

	cv::Size patternsize = cv::Size(7, 6); // (width, height) or (columns, rows) or (X, Y)
	std::vector<cv::Mat> images;
	std::vector<std::vector<cv::Point2f>> Chessboard_corners;

	// This loop goes into the folder defined previously and reads the images one by one.
	// The files MUST be of the same type and have the same name.
	// The files MUST have the same name and have a consecutive numbering scheme
	// For example: image01.jpg, image02.jpg, image03.jpg ...
	while(cap.isOpened())
	{
		// The image is read into the OpenCV matrix vector here
		cv::Mat img;
		cap.read(img);
		if(img.empty())
		{
			std::cout << "End of Sequence \r\n";
			break;
		}
		
		// The recently read image is processed here
		std::vector<cv::Point2f> corners; //This will be filled by the detected corners
		bool chessboard_found = cv::findChessboardCorners(img, patternsize, corners);

		if(chessboard_found)
		{
			std::cout << "The Chessboard was found within the image: " << num_of_images << "\r\n";
			total_chessboards_detected++;
			images.push_back(img); // Only store the images where the Chessboards were found.
			Chessboard_corners.push_back(corners); // Store the calibration points from the image but only for the chessboards that were found.

			// cv::namedWindow("Display window");
			// cv::drawChessboardCorners(img, patternsize, corners, chessboard_found);
			// cv::imshow("Dsplay Window", img);
			// cv::waitKey(0);
		}
		else
		{
			std::cout << "The Chessboard was not found within the image: " << num_of_images << "\r\n";
		}
		num_of_images++;
	}

	std::cout << "Finished reading from the folder! \r\n";
	std::cout << "The total number of images found: " << num_of_images << "\r\n";
	std::cout << "The number of images with Chess boards within the image: " << total_chessboards_detected << "\r\n\n";

	// The calibration board is generated here and is regarded as matrix D
	std::vector<cv::Point3f> matrix_D;
 
	for (int y = 0; y != Number_of_internal_corners_y; y++)
	{
		for (int x = 0; x != Number_of_internal_corners_x; x++)
		{
			matrix_D.push_back(cv::Point3f(x * Block_dimension, y * Block_dimension, 0 * Block_dimension));
		}
	}

	// The solvePnP function is be used here to determine the initial estimates for the camera extrinsics for each image.
	double matrix_B[3][3] = {{initial_fx, 0, initial_cx}, {0, initial_fy, initial_cy}, {0, 0, 1} };
	cv::Mat matrix_B_ = cv::Mat(3, 3, CV_64FC1, matrix_B);
	cv::Mat rvec;
	cv::Mat tvec;
	double camera_extrinsics[total_chessboards_detected][6] = {{0}};

	for (int z = 0; z != total_chessboards_detected; z++)
	{
		cv::solvePnP(matrix_D, Chessboard_corners[z], matrix_B_, cv::Mat(4,1,CV_64FC1,cv::Scalar(0)), rvec, tvec, false);
		camera_extrinsics[z][0] = rvec.at<double>(0, 0); // axis-angle x
		camera_extrinsics[z][1] = rvec.at<double>(0, 1); // axis-angle y
		camera_extrinsics[z][2] = rvec.at<double>(0, 2); // axis-angle z
		camera_extrinsics[z][3] = tvec.at<double>(0, 0); // t1
		camera_extrinsics[z][4] = tvec.at<double>(0, 1); // t2
		camera_extrinsics[z][5] = tvec.at<double>(0, 2); // t3
	}


	// Display the initial camera extrinsics multidimentional array
	for (int z = 0; z != total_chessboards_detected; z++)
	{
		std::cout << "Camera extrinsic: " << z << "\r\n";
		std::cout << "Initial axis-angle x: " << camera_extrinsics[z][0] << "\r\n";
		std::cout << "Initial axis-angle y: " << camera_extrinsics[z][1] << "\r\n";
		std::cout << "Initial axis-angle z: " << camera_extrinsics[z][2] << "\r\n";
		std::cout << "Initial t1: " << camera_extrinsics[z][3] << "\r\n";
		std::cout << "Initial t2: " << camera_extrinsics[z][4] << "\r\n";
		std::cout << "Initial t3: " << camera_extrinsics[z][5] << "\r\n\n";
	} 


	///////////// The optimisation code begins from here /////////////

    // Set the initial values for the mutable parameters.
    double camera_intrinsics[4] = {initial_fx, initial_fy, initial_cx, initial_cy};

    // Begin building the problem
    ceres::Problem problem;

	for (int z = 0; z != Chessboard_corners.size(); z++)
	{
		// std::cout << "////////////////////////// Image number: " << z << " ////////////////////////// \r\n";	
		
		for (int i = 0; i != Chessboard_corners[z].size(); i++)
		{
			double board_T_point[3] = {matrix_D[i].x, matrix_D[i].y, matrix_D[i].z}; //This is the calibration point of the format: X,Y,Z
			double image_pixels[2] = {Chessboard_corners[z][i].x, Chessboard_corners[z][i].y}; //This is the image point of the format: u, v.

			// std::cout << "Calibration point number: " << i << "\r\n";
			// std::cout << "XYZ: " << board_T_point[0] << ", " << board_T_point[1] << ", " << board_T_point[2] << "\r\n\n";

			// std::cout << "Pixel point number: " << i << "\r\n";
			// std::cout << "pixel u: " << image_pixels[0] << "\r\n";
			// std::cout << "pixel v: " << image_pixels[1] << "\r\n\n";

			ceres::CostFunction* cost_function = ReProjectionResidual::Create(image_pixels, board_T_point);
			// double residuals[2];
			// cost_function(&camera_extrinsics[z][0], camera_intrinsics, residuals);
			// for (?)
			//   print residuals[i]
			problem.AddResidualBlock(cost_function, NULL, &camera_extrinsics[z][0], camera_intrinsics);
		}
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 10000;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << "\r\n";

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

	for (int z = 0; z != total_chessboards_detected; z++)
	{
		// Camera Extrinsics results
		std::cout << "CAMERA EXTRINSICS RESULTS FOR IMAGE: " << z << " \r\n";
		std::cout << "Final   axis-angle x: " << camera_extrinsics[z][0]  << "\r\n";
		std::cout << "Final   axis-angle y: " << camera_extrinsics[z][1]  << "\r\n";
		std::cout << "Final   axis-angle z: " << camera_extrinsics[z][2]  << "\r\n";
		std::cout << "Final   t1: " << camera_extrinsics[z][3]  << "\r\n";
		std::cout << "Final   t2: " << camera_extrinsics[z][4]  << "\r\n";
		std::cout << "Final   t3: " << camera_extrinsics[z][5]  << "\r\n\n";
	}

	// This step is not necessary but it is being done to determine if the optimised camera intrinsics and
	// extrinsics returns correct pixel points for the calibration ponits.

	double matrix_B__[3][3] = {{camera_intrinsics[0], 0, camera_intrinsics[2]}, {0, camera_intrinsics[1], camera_intrinsics[3]}, {0, 0, 1} };

	// For matrix_C I have to convert the axis angle into the 9x9 rotation matrix.

	double matrix_C[total_chessboards_detected][3][4];

	for (int z = 0; z != total_chessboards_detected; z++)
	{
		double angle_axis[3] = {camera_extrinsics[z][0], camera_extrinsics[z][1],  camera_extrinsics[z][2]};
		double rotation_matrix[9];
		ceres::AngleAxisToRotationMatrix(&angle_axis[0], &rotation_matrix[0]);

		matrix_C[z][0][0] = rotation_matrix[0];
		matrix_C[z][1][0] = rotation_matrix[1];
		matrix_C[z][2][0] = rotation_matrix[2];
		matrix_C[z][0][1] = rotation_matrix[3];
		matrix_C[z][1][1] = rotation_matrix[4];
		matrix_C[z][2][1] = rotation_matrix[5];
		matrix_C[z][0][2] = rotation_matrix[6];
		matrix_C[z][1][2] = rotation_matrix[7];
		matrix_C[z][2][2] = rotation_matrix[8];
		matrix_C[z][0][3] = camera_extrinsics[z][3];
		matrix_C[z][1][3] = camera_extrinsics[z][4];
		matrix_C[z][2][3] = camera_extrinsics[z][5];

	}

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
	cv::Mat matrix_A[total_chessboards_detected];
	cv::Mat matrix_B___ = cv::Mat(3, 3, CV_64FC1, matrix_B__);
	std::vector<cv::Mat> matrix_C_;
	cv::Mat matrix_D__ = cv::Mat(4, Number_of_internal_corners_y * Number_of_internal_corners_x, CV_64FC1, matrix_D_);
	

	for (int z = 0; z != total_chessboards_detected; z++)
	{
		matrix_C_.push_back(cv::Mat(3, 4, CV_64FC1, &matrix_C[z][0][0]));
		//std::cout << "matrix_C_: \r\n" << matrix_C_[z];
	}

	std::cout << "\r\n";


	for (int z = 0; z != total_chessboards_detected; z++)
	{
		matrix_A[z] = matrix_B___ * matrix_C_[z] * matrix_D__;
		// std::cout << "matrix_A: \r\n" << matrix_A[z];
		// std::cout << "\r\n";
	}

	// Display the real and calculated matrix A here. 
	double true_matrix_A[total_chessboards_detected][2][Number_of_internal_corners_y * Number_of_internal_corners_x];

	for (int z = 0; z != total_chessboards_detected; z++)
	{
		for (int i = 0; i != Number_of_internal_corners_y * Number_of_internal_corners_x; i++)
		{
			true_matrix_A[z][0][i] = Chessboard_corners[z][i].x;
			true_matrix_A[z][1][i] = Chessboard_corners[z][i].y;
		}
	}

	double calculated_matrix_A[total_chessboards_detected][3][Number_of_internal_corners_y * Number_of_internal_corners_x];

	for (int z = 0; z != total_chessboards_detected; z++)
	{
		for (int y = 0; y != 3; y++)
		{
			for (int x = 0; x != Number_of_internal_corners_y * Number_of_internal_corners_x; x++)
			{
				calculated_matrix_A[z][y][x] = matrix_A[z].at<double>(y,x);
			}	
		}
	}

	// Display the 2 matrix_A
	for (int z = 0; z != total_chessboards_detected; z++)
	{
		for (int i = 0; i != Number_of_internal_corners_y * Number_of_internal_corners_x; i++)
		{
			std::cout << "Point number: " << i << "\r\n";
			std::cout << "Real: \r\n";
			std::cout << "Pixel u: " << true_matrix_A[z][0][i] << "\r\n";
			std::cout << "Pixel v: " << true_matrix_A[z][1][i] << "\r\n";

			std::cout << "Calculated: \r\n";
			std::cout << "Pixel u: " << (calculated_matrix_A[z][0][i] / calculated_matrix_A[z][2][i]) << "\r\n";
			std::cout << "Pixel v: " << (calculated_matrix_A[z][1][i] / calculated_matrix_A[z][2][i]) << "\r\n\n";
		}	
	}
	

    return 0;
}