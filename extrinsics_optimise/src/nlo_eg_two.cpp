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
static const double Number_of_blocks_y = 5;
static const double Number_of_blocks_x = 4;
static const double Block_dimension = 2; //This is the side of a block in cm.

// Definitions for the camera intrinsics.
static const double fx = 4;
static const double fy = 5;
static const double cx = 2;
static const double cy = 2;
static const double Tx = 4;
static const double Ty = 5;
static const double k1 = 2;
static const double k2 = 2;

// Definitions for the transformation from the camera to edge of calibration board
static const double r11 = 1, r12 = 0, r13 = 0, t1 = 5;
static const double r21 = 0, r22 = 0, r23 = -1, t2 = 7;
static const double r31 = 0, r32 = 1, r33 = 0, t3 = 2;

// Definitions for the initial estimates for the values desired to be optimised
static const double initial_t1 = 50.5;
static const double initial_t2 = 100.2;
static const double initial_t3 = 30.9;

static const double initial_angle_x = 90;
static const double initial_angle_y = 2.2;
static const double initial_angle_z = 3.4;

// Data Structure definition
struct ReProjectionResidual
{ 
	// Put the actual parameters that will not be optimised right here (this would be the intrinsics,the uv pixel points and board_T_cam).
	ReProjectionResidual(const double *pixel_points, const double *camera_intrinsics, const double *board_T_point)
	{
		// Initialise the pixel points
		observed_pixel_points_[0] = pixel_points[0] / pixel_points[2]; // u
		observed_pixel_points_[1] = pixel_points[1] / pixel_points[2]; // v
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
	bool operator()(const T* const camera_extrinsics, T* residuals)
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
		p[0] += camera_extrinsics[3]; // X component of camera to calibration point+
        p[1] += camera_extrinsics[4]; // Y component of camera to calibration point
        p[2] += camera_extrinsics[5]; // Z component of camera to calibration point

		// The projected pixel coordinates would now be computed. (for now i am not including distortion)
		T up = p[0] / p[2];
		T vp = p[1] / p[2];

		T projected_u = up * T(camera_intrinsics_[0]) + T(camera_intrinsics_[2]);
		T projected_v = vp * T(camera_intrinsics_[1]) + T(camera_intrinsics_[3]);

		// The residuals are calculated here
		residuals[0] = projected_u - T(observed_pixel_points_[0]);
		residuals[1] = projected_v - T(observed_pixel_points_[1]);

		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction *Create(const double *pixel_points, const double *camera_intrinsics, const double *board_T_point)
	{
		return (new ceres::AutoDiffCostFunction<ReProjectionResidual, 2, 6>(
			new ReProjectionResidual(pixel_points, camera_intrinsics, board_T_point)));
	}

	// Declare struct variables
	private:
		double observed_pixel_points_[3];
		double camera_intrinsics_[8];
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
	double matrix_C[3][4];

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
	std::vector<std::vector<double>> matrix_D;
	for (int y = 0; y != Number_of_blocks_y; y++)
	{
		for (int x = 0; x != Number_of_blocks_x; x++)
		{
			std::vector<double> point;

			point.push_back(x);
			point.push_back(y);
			point.push_back(0);
			point.push_back(1);

			matrix_D.push_back(point);
		}
	}

	// Compute the matrix multiplication of [B][C] = [E]. (3x4 matrix)
	double matrix_E[3][4] = {{0}};
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
	std::vector<std::vector<double>> matrix_A;
	for (int num_of_points = 0; num_of_points != matrix_D.size(); num_of_points++)
	{
		std::vector<double> point;

		//This nested loop used to do the matrix multiplication
		for (int y = 0; y != 3; y++)
		{
			double matrix_product = 0;
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

	double camera_extrinsics[6] = {initial_angle_x, initial_angle_y, initial_angle_z, initial_t1, initial_t2, initial_t3};

	double camera_intrinsics[8] = {fx, fy, cx, cy, Tx, Ty, k1, k2};

	//Begin building the problem.
	ceres::Problem problem;
	
	for(int i = 0; i != matrix_D.size(); ++i)
	{
		double board_T_point[4] = {matrix_D[i][0], matrix_D[i][1], matrix_D[i][2], matrix_D[i][3]}; //This is the calibration point of the format: X,Y,Z,1
		double image_pixels[3] = {matrix_A[i][0], matrix_A[i][1], matrix_A[i][2]}; //This is the image point of the format: s*u,s*v,s.

		ceres::CostFunction* cost_function = ReProjectionResidual::Create(image_pixels, camera_intrinsics, board_T_point);
		problem.AddResidualBlock(cost_function, NULL, camera_extrinsics); 
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 1000;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << "\r\n";
	//std::cout << summary.FullReport() << "\n\n\r";

	// The true value for the R matrix in angle-axis convention is calculated here
	double rotation_matrix[9] = {r11, r21, r31, r12, r22, r32, r13, r23, r33};

	double true_angle_axis[3];

	//Transform the rotation matrix to angle axis
	ceres::RotationMatrixToAngleAxis(&rotation_matrix[0], &true_angle_axis[0]);

	// The results from the optimization is displayed here
	std::cout << "Initial axis-angle x: " << initial_angle_x << "\r\n";
	std::cout << "Final   axis-angle x: " << camera_extrinsics[0]  << "\r\n";
	std::cout << "True    axis-angle x: " << true_angle_axis[0] << "\r\n\n";

	std::cout << "Initial axis-angle y: " << initial_angle_y << "\r\n";
	std::cout << "Final   axis-angle y: " << camera_extrinsics[1]  << "\r\n";
	std::cout << "True    axis-angle y: " << true_angle_axis[1] << "\r\n\n";

	std::cout << "Initial axis-angle z: " << initial_angle_z << "\r\n";
	std::cout << "Final   axis-angle z: " << camera_extrinsics[2]  << "\r\n";
	std::cout << "True    axis-angle z: " << true_angle_axis[2] << "\r\n\n";

	std::cout << "Initial t1: " << initial_t1 << "\r\n";
	std::cout << "Final   t1: " << camera_extrinsics[3]  << "\r\n";
	std::cout << "True    t1: " << t1  << "\r\n\n";

	std::cout << "Initial t2: " << initial_t2 << "\r\n";
	std::cout << "Final   t2: " << camera_extrinsics[4]  << "\r\n";
	std::cout << "True    t2: " << t2  << "\r\n\n";

	std::cout << "Initial t3: " << initial_t3 << "\r\n";
	std::cout << "Final   t3: " << camera_extrinsics[5]  << "\r\n";
	std::cout << "True    t3: " << t3  << "\r\n\n";


	return 0;
}