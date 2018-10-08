#include <cmath>
#include <cstdio>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <vector>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

/*For this board, the reference frame for the calibration board would be alligned with its edge.
This means that the x-axis would be alligned with the width axis and the y-axis would be alligned length axis of the board.
This would therefore imply that the z component of each point is 0 since all points are strictly on the x-y plane.

Key:
matrix [A] - 3x1 matrix - this is the image frame
matrix [B] - 3x3 matrix - this is the camera properties matrix
matrix [C] - 3x4 matrix - this is the camera to calibration board transformation
matris [D] - 4x1 matrix - this is the calibration board reference to point on calibration point

The coordinates of the projection point in pixels is therefore given by the equation: [A] = [B][C][D] 
*/

// Definitions for the dimensions of the calibration board
static const int Number_of_blocks_y = 5;
static const int Number_of_blocks_x = 4;
static const double Block_dimension = 1; // This is the side of a block in m.

// Definitions for the camera intrinsics.
static const double fx = 4;
static const double fy = 5;
static const double cx = 2;
static const double cy = 2;

// Definitions for the initial estimates for the values desired to be optimised
static const double initial_t1 = 5;
static const double initial_t2 = 7;
static const double initial_t3 = 2;

static const double initial_angle_x = 1.5;
static const double initial_angle_y = 0;
static const double initial_angle_z = 0;

static const double initial_fx = 4;
static const double initial_fy = 5;
static const double initial_cx = 2;
static const double initial_cy = 2;

// Define the number of images (number of camera extrinsics)
static const int images = 10;

// Data Structure definition
struct ReProjectionResidual
{ 

	// Put the actual parameters that will not be optimised right here (this would be the intrinsics,the uv pixel points and board_T_cam).
	ReProjectionResidual(const double *pixel_points, const double *board_T_point)
	{
		// Initialise the pixel points
		observed_pixel_points_[0] = pixel_points[0] / pixel_points[2]; // u
		observed_pixel_points_[1] = pixel_points[1] / pixel_points[2]; // v
		observed_pixel_points_[2] = pixel_points[2]; // s

		// Initialise the board_T_point
		board_T_point_[0] = board_T_point[0]; // X
		board_T_point_[1] = board_T_point[1]; // Y
		board_T_point_[2] = board_T_point[2]; // Z
		board_T_point_[3] = board_T_point[3]; // 1
	}

	// For this template, put all of the parameters to optimise and the residual here.
	template <typename T>
	bool operator()(const T* angle_x, const T* angle_y, const T* angle_z, const T* t1, const T* t2, const T* t3, const T* const camera_intrinsics, T* residuals)
	const
	{
		// compute projective coordinates: p = RX + t.
        // camera_extrinsics[0, 1, 2]: axis-angle
        // camera_extrinsics[3, 4, 5]: translation
		const T R_angle_axis[3] = {*angle_x, *angle_y, *angle_z};
		const T point[3] = {T(board_T_point_[0]), T(board_T_point_[1]), T(board_T_point_[2])};
		T p[3];

		// AngleAxisRotatePoint used to rotate the board_T_point about the axis of rotation which is set 
		// as the R component of the camera extrinsic matric after the rotation matrix to angle axis conversion
		ceres::AngleAxisRotatePoint(R_angle_axis, point, p);

		// AngleAxisRotatePoint gives the "RX" therefore it must be translated by "t" (from camera extrinsics) to give "p".
		p[0] += *t1; // X component of camera to calibration point
		p[1] += *t2; // Y component of camera to calibration point
		p[2] += *t3; // Z component of camera to calibration point

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
		return (new ceres::AutoDiffCostFunction<ReProjectionResidual, 2, 1, 1, 1, 1, 1, 1, 4>(
			new ReProjectionResidual(pixel_points, board_T_point)));
	}

	// Declare struct variables
	private:
		double observed_pixel_points_[3];
		double board_T_point_[4];

};

int main()
{
	// The matrix [B] is populated here. (3x3 matrix)
	double matrix_B[3][3] = {{0}};

	matrix_B[0][0] = fx;
	matrix_B[1][1] = fy;
	matrix_B[0][2] = cx;
	matrix_B[1][2] = cy;
	matrix_B[2][2] = 1;

	// The matrix [C] is populated here. (3x4 matrix)
	double matrix_C[images][3][4];

	// Multiple extrinsics are generated here but only when there is a rotation about the x-axis.
	// More work could possible be done to explore rotations about the other 2 axes as well.
	for (int z = 0; z != images; z++)
	{
		matrix_C[z][0][0] = 1;
		matrix_C[z][0][1] = 0;
		matrix_C[z][0][2] = 0;
		matrix_C[z][0][3] = (rand() % 5 + 1);
		matrix_C[z][1][0] = 0;
		matrix_C[z][1][1] = 0;
		matrix_C[z][1][2] = -1;
		matrix_C[z][1][3] = (rand() % 5 + 1);
		matrix_C[z][2][0] = 0;
		matrix_C[z][2][1] = 1;
		matrix_C[z][2][2] = 0;
		matrix_C[z][2][3] = (rand() % 5 + 1);
	}

	// The matrix [D] is computed here. It is a (4x1 matrix) but a list of these matrices.
	// The code generates a list of matrices for the calibration points. Note that the z components for all of the points would be 0.
	double matrix_D[4][Number_of_blocks_y * Number_of_blocks_x];

	int count = 0;
	for (int y = 0; y != Number_of_blocks_y; y++)
	{
		for (int x = 0; x != Number_of_blocks_x; x++)
		{
			matrix_D[0][count] = x * Block_dimension;
			matrix_D[1][count] = y * Block_dimension;
			matrix_D[2][count] = 0 * Block_dimension;
			matrix_D[3][count] = 1;

			count++;
		}	
	}

	// Compute the matrix multiplication of [B][C] = [E]. (3x4 matrix)
	double matrix_E[images][3][4] = {{{0}}};

	for (int z = 0; z != images; z++)
	{
		for (int y = 0; y != 3; y++)
		{
			for (int x = 0; x != 4; x++)
			{
				for (int index = 0; index != 3; index++)
				{
					matrix_E[z][y][x] += matrix_B[y][index] * matrix_C[z][index][x];
				}
			}
		}
	}

	// Compute the matrix multiplication of [E][D] = [A]. It is a (3x1 matrix) but a list of these matrices.
	double matrix_A[images][3][Number_of_blocks_y * Number_of_blocks_x] = {{{0}}};

	for (int z = 0; z != images; z++) //Loop to increment the different extrinsics
	{
		for (int x = 0; x != Number_of_blocks_y * Number_of_blocks_x; x++) // Loop to index different calibration points
		{
			for (int y = 0; y != 3; y++) // Loop to index rows in matrix [E]
			{
				for (int index = 0; index != 4; index++)
				{
					matrix_A[z][y][x] += matrix_E[z][y][index] * matrix_D[index][x];
				}	
				
			}
		}
	}
	
	///////////// The optimisation code begins from here /////////////

	// For this code, the pixel points (matrix [A]) are known for the calibration points (matrix [D]), given the camera intrinsics (matrix [B]) and the extrinsics (matrix [C]).
	// The next step would be to ignore the known contents of the matrix [B] and matrix [C] and try to optimise it given an initial estimate to give a new matrix [A] that approximate the true pixel values. 
	// The initial estimates of matrix [C] and matrix [B] are defined here. Remember that the rotation part of matrix [C] must be represented as 3 values (the angle axis). The position is defined as normal

	// This code is used to initialise the camera intrinsics and the camera extrinsics.
	double camera_intrinsics[4] = {initial_fx, initial_fy, initial_cx, initial_cy};
	double camera_extrinsics[images][6];

	// All of the initial camera extrinsics are defined here. Note that if there were multiple cameras then I believe that the camera intrinsics would have to be
	// defined the same way here and just like the "problem.ResudualBlock()" function the camera intrinsics should be defined in a similar manner to the extrinsics.
	for (int z = 0; z != images; z++)
	{
		camera_extrinsics[z][0] = 1.5; // Initial angle-x
		camera_extrinsics[z][1] = 0; // Initial angle-y
		camera_extrinsics[z][2] = 0; // Initial angle-z
		camera_extrinsics[z][3] = 2.5; // Initial t1
		camera_extrinsics[z][4] = 2.5; // Initial t2
		camera_extrinsics[z][5] = 2.5; // Initial t3
	}

	// Begin building the problem.
	ceres::Problem problem;
	
	for (int z = 0; z != images; z++)
	{	
		for (int observations = 0; observations != Number_of_blocks_y * Number_of_blocks_x; observations++)
		{
			double board_T_point[4] = {matrix_D[0][observations], matrix_D[1][observations], matrix_D[2][observations], matrix_D[3][observations]}; // This is the calibration point of the format: X,Y,Z,1
			double image_pixels[3] = {matrix_A[z][0][observations], matrix_A[z][1][observations], matrix_A[z][2][observations]}; // This is the image point of the format: s*u,s*v,s.

			ceres::CostFunction* cost_function = ReProjectionResidual::Create(image_pixels, board_T_point);
			problem.AddResidualBlock(cost_function, NULL, &camera_extrinsics[z][0], &camera_extrinsics[z][1], &camera_extrinsics[z][2], &camera_extrinsics[z][3], &camera_extrinsics[z][4], &camera_extrinsics[z][5], camera_intrinsics); 
		}
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 1000;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << "\r\n";

	// The results from the optimization is displayed here

	// Camera Intrinsics results
	std::cout << "CAMERA INTRINSICS RESULTS: \r\n\n";
	std::cout << "Final   fx: " << camera_intrinsics[0]  << "\r\n";
	std::cout << "True    fx: " << fx  << "\r\n\n";

	std::cout << "Final   fy: " << camera_intrinsics[1]  << "\r\n";
	std::cout << "True    fy: " << fy  << "\r\n\n";

	std::cout << "Final   cx: " << camera_intrinsics[2]  << "\r\n";
	std::cout << "True    cx: " << cx  << "\r\n\n";

	std::cout << "Final   cy: " << camera_intrinsics[3]  << "\r\n";
	std::cout << "True    cy: " << cy  << "\r\n\n";

	return 0;
}