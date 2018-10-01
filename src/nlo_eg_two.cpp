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
static const float Number_of_blocks_y = 5;
static const float Number_of_blocks_x = 4;
static const float Block_dimension = 2; //This is the side of a block in cm.

// Definitions for the camera intrinsics.
static const float fx = 4;
static const float fy = 5;
static const float cx = 2;
static const float cy = 2;
static const float Tx = 4;
static const float Ty = 5;
static const float k1 = 2;
static const float k2 = 2;

// Definitions for the transformation from the camera to edge of calibration board
static const float r11 = 1, r12 = 0, r13 = 0, t1 = 1;
static const float r21 = 0, r22 = 1, r23 = 0, t2 = 1;
static const float r31 = 0, r32 = 0, r33 = 1, t3 = 1;

// Definitions for the initial estimates for the values desired to be optimised
static const float initial_t1 = 1;
static const float initial_t2 = 2;
static const float initial_t3 = 0.5;

static const float initial_angle_x = 1;
static const float initial_angle_y = 2;
static const float initial_angle_z = 0.5;

// Data Structure definition
struct ReProjectionResidual
{ 
	// Put the actual parameters that will not be optimised right here (this would be the intrinsics,the uv pixel points and board_T_cam).
	ReProjectionResidual(float *pixel_points, float *camera_intrinsics, float *board_T_point)
	{
		// Initialise the pixel points
		observed_pixel_points_[0] = pixel_points[0]; // s * u
		observed_pixel_points_[1] = pixel_points[1]; // s * v
		observed_pixel_points_[2] = pixel_points[2]; // s

		// Initialise the camera_intrinsics
		camera_intrinsics_[0] = camera_intrinsics[0]; // fx
		camera_intrinsics_[1] = camera_intrinsics[1]; // fy
		camera_intrinsics_[2] = camera_intrinsics[2]; // cx
		camera_intrinsics_[3] = camera_intrinsics[3]; // cy
		camera_intrinsics_[4] = camera_intrinsics[4]; // Tx
		camera_intrinsics_[5] = camera_intrinsics[5]; // Ty
		camera_intrinsics_[6] = camera_intrinsics[6]; // k1
		camera_intrinsics_[7] = camera_intrinsics[7]; // k2

		// Initialise the board_T_point
		board_T_point_[0] = board_T_point[0]; // X
		board_T_point_[1] = board_T_point[1]; // Y
		board_T_point_[2] = board_T_point[2]; // Z
		board_T_point_[3] = board_T_point[3]; // 1
	}

	// For this template, put all of the parameters to optimise and the residual here.
	template <typename T>
	bool operator()(const T* const camera_extrinsics, const T* const residuals)
	const
	{
		// compute projective coordinates: p = RX + t.
        // extrinsics[0, 1, 2]: axis-angle
        // extrinsics[3, 4, 5]: translation
		T p[3];
		ceres::AngleAxisRotatePoint(camera_extrinsics, board_T_point_, p);
		p[0] += camera_extrinsics[3]; // X
        p[1] += camera_extrinsics[4]; // Y
        p[2] += camera_extrinsics[5]; // Z

		// The projected pixel coordinates would now be computed. (for now i am not including distortion)
		T up = p[0] / p[2];
		T vp = p[1] / p[2];

		T projected_u = up * camera_intrinsics_[0] + camera_intrinsics_[2];
		T projected_v = vp * camera_intrinsics_[1] + camera_intrinsics_[3];

		// The residuals are calculated here
		residuals[0] = projected_u - T(observed_pixel_points_[0]);
		residuals[1] = projected_u - T(observed_pixel_points_[1]);

		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction *Create(const float *pixel_points, const float *camera_intrinsics, const float *board_T_point)
	{
		return (new ceres::AutoDiffCostFunction<ReProjectionResidual, 2, 6>(
			new ReProjectionResidual(*pixel_points, *camera_intrinsics, *board_T_point)));
	}

	// Declare struct variables
	private:
		float observed_pixel_points_[3];
		float camera_intrinsics_[8];
		float board_T_point_[4];
};

int main()
{
	// The matrix [B] is populated here. (3x3 matrix)
	float matrix_B[3][3] = {{0}};

	matrix_B[0][0] = fx;
	matrix_B[1][1] = fy;
	matrix_B[0][2] = cx;
	matrix_B[1][2] = cy;
	matrix_B[2][2] = 1;

	// The matrix [C] is populated here. (3x4 matrix)
	float matrix_C[3][4];

	matrix_C[0][0] = r11;
	matrix_C[0][1] = r12;
	matrix_C[0][2] = r13;
	matrix_C[0][3] = t1;
	matrix_C[1][0] = r21;
	matrix_C[1][1] = r22;
	matrix_C[1][2] = r23;
	matrix_C[1][3] = t2;
	matrix_C[2][0] = r31;
	matrix_C[2][1] = r32;
	matrix_C[2][2] = r33;
	matrix_C[2][3] = t3;

	// The matrix [D] is computed here. It is a (4x1 matrix) but a list of these matrices.
	// The code generates a list of matrices for the calibration points. Note that the z components for all of the points would be 0.
	std::vector<std::vector<float>> matrix_D;
	for (int y = 0; y != Number_of_blocks_y; y++)
	{
		for (int x = 0; x != Number_of_blocks_x; x++)
		{
			std::vector<float> point;

			point.push_back(x);
			point.push_back(y);
			point.push_back(0);
			point.push_back(1);

			matrix_D.push_back(point);
		}
	}

	// Compute the matrix multiplication of [B][C] = [E]. (3x4 matrix)
	float matrix_E[3][4] = {{0}};
	for (int y = 0; y != 3; y++)
	{
		for (int x = 0; x != 4; x++)
		{
			for (int z = 0; z != 3; z++)
			{
				matrix_E[y][x] += matrix_B[y][z] * matrix_C[z][x];
			}
		}
	}

	// Compute the matrix multiplication of [E][D] = [A]. It is a (3x1 matrix) but a list of these matrices.
	std::vector<std::vector<float>> matrix_A;
	for (int num_of_points = 0; num_of_points != matrix_D.size(); num_of_points++)
	{
		std::vector<float> point;

		//This nested loop used to do the matrix multiplication
		for (int y = 0; y != 3; y++)
		{
			float matrix_product = 0;
			for (int z = 0; z != 4; z++)
			{
				matrix_product += matrix_E[y][z] * matrix_D[num_of_points][z];
			}

			point.push_back(matrix_product);
		}

		matrix_A.push_back(point);
		point.clear();
	}
	
	///////////// The optimisation code begins from here /////////////

	// For this code, the pixel points (matrix [A]) are known for the calibration points (matrix [D]), given the camera intrinsics (matrix [B]) and the cam_T_board (matrix [C]).
	// The next step would be to ignore the known contents of the matrix [C] and try to optimise it given an initial estimate to give a new matrix [A] that approximate the true pixel values. 
	// The initial estimates of matrix [C] are defined here. Remember that the rotation part of matrix [C] must be represented as 3 values (the angle axis). The position is defined as normal

	float t_1 = initial_t1;
	float t_2 = initial_t2;
	float t_3 = initial_t3;

	float angle_x = initial_angle_x;
	float angle_y = initial_angle_y;
	float angle_z = initial_angle_z;

	//Begin building the problem.
	ceres::Problem problem;

	//Format for the camera_intrinsics array [0..7]: fx, fy, cx, cy, Tx, Ty, k1, k2.
	float camera_intrinsics[8] = {fx, fy, cx, cy, Tx, Ty, k1, k2};

	for(int i = 0; i != matrix_D.size(); ++i)
	{
		float board_T_point[4] = {matrix_D[i][0], matrix_D[i][1], matrix_D[i][2], matrix_D[i][3]}; //This is the calibration point of the format: X,Y,Z,1
		float image_pixels[3] = {matrix_A[i][0], matrix_A[i][1], matrix_A[i][2]}; //This is the image point of the format: s*u,s*v,s.

		//ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction <>
	}


	return 0;
}