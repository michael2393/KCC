#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <float.h>
#include "settings.h"


#ifdef __INTELLISENSE__
#include "intellisense_cuda_intrinsics.h"
#endif




__device__ void gpuErrchk(cudaError_t code) {
	if (code != cudaSuccess) {
		printf("cudaMalloc error: %s\n", cudaGetErrorString(code));
	}
}



double* randomCircles() {
	double *Circles;
	Circles = (double *)malloc(3 * 5 * sizeof(double));
	for (int k = 0; k < MAX_CIRCLES; k++) {
		srand(static_cast <unsigned> (time(0)*k));
		Circles[k * 3 + 0] = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / 2)) - 1.0;
		Circles[k * 3 + 1] = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / 2)) - 1.0;
		Circles[k * 3 + 2] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
		if (Circles[k * 3 + 2] < 0.1) Circles[k * 3 + 2] = 0.1;
	}
	return Circles;
}



__global__ void computeErrors(double *Circles, double *Points, int PointsOffset, int *Assignment, int currentAssignmentOffset, int numCircles, float *Errors) {
	int id = threadIdx.x, pointAssignment = Assignment[currentAssignmentOffset + id];
	double x, y;
	float error;
	x = Points[PointsOffset + id * 2];
	y = Points[PointsOffset + id * 2 + 1];
	error = pow(pow(x - Circles[pointAssignment * 3 + 0], 2) + pow(y - Circles[pointAssignment * 3 + 1], 2) - pow(Circles[pointAssignment * 3 + 2], 2), 2);
	atomicAdd(&Errors[numCircles - MIN_CIRCLES], error);
}



__global__ void fitCircles(double *Circles, double *Points, int PointsOffset, int *Assignment, int currentAssignmentOffset, int numPoints) {
	int id = threadIdx.x;
	int *listOfPoints, i, pointsInCircle = 0;
	double sums[8], tmp1, tmp2, tmp3, x, y, det, R[3][3], A, B, C;
	bool hasPoints = false;
	cudaMalloc((void **)&listOfPoints, numPoints * sizeof(int));
	for (i = 0; i < 8; i++) sums[i] = 0;
	for (i = 0; i < numPoints; i++) {
		if (Assignment[currentAssignmentOffset + i] == id) { listOfPoints[pointsInCircle] = i;  pointsInCircle++; }
	}
	if (pointsInCircle >= 5) hasPoints = true;
	if (hasPoints && Circles[id * 3 + 2] >= 0.1 && Circles[id * 3 + 2] <= 1.0) {
		for (i = 0; i < pointsInCircle; i++) {
			x = Points[PointsOffset + listOfPoints[i] * 2];
			y = Points[PointsOffset + listOfPoints[i] * 2 + 1];
			tmp1 = x * x; tmp2 = y * y; tmp3 = x * y;
			sums[0] += x;		sums[1] += y;		sums[2] += tmp1;		sums[3] += tmp2;
			sums[4] += tmp3;	sums[5] += ((tmp1 + tmp2)*x);	sums[6] += ((tmp1 + tmp2)*y);	sums[7] += (tmp1 + tmp2);
		}
		det = sums[2] * (sums[3] * pointsInCircle - sums[1] * sums[1])
			- sums[4] * (sums[4] * pointsInCircle - sums[1] * sums[0])
			+ sums[0] * (sums[4] * sums[1] - sums[3] * sums[0]);

		R[0][0] = (sums[3] * pointsInCircle - sums[1] * sums[1]) / det;
		R[0][1] = -(sums[4] * pointsInCircle - sums[1] * sums[0]) / det;
		R[0][2] = (sums[4] * sums[1] - sums[3] * sums[0]) / det;
		R[1][0] = R[0][1];
		R[1][1] = (sums[2] * pointsInCircle - sums[0] * sums[0]) / det;
		R[1][2] = -(sums[2] * sums[1] - sums[4] * sums[0]) / det;
		R[2][0] = R[0][2];
		R[2][1] = R[1][2];
		R[2][2] = (sums[2] * sums[3] - sums[4] * sums[4]) / det;

		A = R[0][0] * sums[5] + R[0][1] * sums[6] + R[0][2] * sums[7];
		B = R[1][0] * sums[5] + R[1][1] * sums[6] + R[1][2] * sums[7];
		C = R[2][0] * sums[5] + R[2][1] * sums[6] + R[2][2] * sums[7];

		Circles[id * 3 + 0] = A / 2.0;
		Circles[id * 3 + 1] = B / 2.0;
		Circles[id * 3 + 2] = (sqrt(4 * C + A * A + B * B)) / 2.0;
	}
	else {
		Circles[id * 3 + 0] = 0.0;
		Circles[id * 3 + 1] = 0.0;
		Circles[id * 3 + 2] = INVALID_CIRCLE_RADIOUS;
	}
	cudaFree(listOfPoints);
}



__global__ void distThread(double *Circles, double *Points, int PointsOffset, int numCircles, int *Assignment, int currentAssignmentOffset, int *changes) {
	int id = threadIdx.x, i, newAssignment = -1, prevAssignment = Assignment[currentAssignmentOffset + id];
	double x, y, minDist = DBL_MAX, tmp;
	x = Points[PointsOffset + id * 2];
	y = Points[PointsOffset + id * 2 + 1];
	for (i = 0; i < numCircles; i++) {
		tmp = pow(pow(x - Circles[i * 3 + 0], 2) + pow(y - Circles[i * 3 + 1], 2) - pow(Circles[i * 3 + 2], 2), 2);
		if (tmp < minDist) { minDist = tmp; newAssignment = i; }
	}
	if (prevAssignment != newAssignment) atomicAdd(changes, 1);

	Assignment[currentAssignmentOffset + id] = newAssignment;
	__syncthreads();
}



__global__ void mainThread(double *points, int *eventPoints_num, int *pointsPointers, double *randCircles, double *Results, int *numResult, int batchIndex) {
	int id = threadIdx.x, block = blockIdx.x, currentAssignmentOffset, maxIter, *changes, *Assignment;
	double *Circles;
	float *Errors, minError = FLT_MAX;

	gpuErrchk(cudaMalloc((void **)&Assignment, (MAX_CIRCLES - MIN_CIRCLES + 1) * eventPoints_num[block] * sizeof(int)));
	gpuErrchk(cudaMalloc((void **)&changes, sizeof(int)));
	gpuErrchk(cudaMalloc((void **)&Errors, (MAX_CIRCLES - MIN_CIRCLES + 1) * sizeof(float)));


	for (int c = MIN_CIRCLES; c <= MAX_CIRCLES; c++) {
		cudaMalloc((void **)&Circles, c * 3 * sizeof(double));
		for (int j = 0; j < 3 * c; j++) { Circles[j] = randCircles[j]; }
		*changes = 1; // Updated in 'distThread' (atomic add). It counts the number of changes in the assignments of points to circles
		currentAssignmentOffset = (c - MIN_CIRCLES) * eventPoints_num[block];
		maxIter = MAX_ITERATIONS;

		while (maxIter > 0 && *changes > 0) {
			maxIter--;
			*changes = 0;

			distThread <<<1, eventPoints_num[block]>>> (Circles, points, pointsPointers[batchIndex * EVENTS_BATCH + block], c, Assignment, currentAssignmentOffset, changes);
			cudaDeviceSynchronize();

			fitCircles <<<1, c>>> (Circles, points, pointsPointers[batchIndex * EVENTS_BATCH + block], Assignment, currentAssignmentOffset, eventPoints_num[block]);
			cudaDeviceSynchronize();
		}

		computeErrors <<<1, eventPoints_num[block]>>> (Circles, points, pointsPointers[batchIndex * EVENTS_BATCH + block], Assignment, currentAssignmentOffset, c, Errors);

		if (cudaDeviceSynchronize() != cudaSuccess) {
			printf("computeErrors failed!\n");
		}


		Errors[c - MIN_CIRCLES] += 0.01*c*c;	// Penalty
		if (Errors[c - MIN_CIRCLES] < minError) {
			numResult[batchIndex * EVENTS_BATCH + block] = 0;
			minError = Errors[c - MIN_CIRCLES];
			for (int i = 0; i < c; i++) {
				if (Circles[i * 3 + 2] != INVALID_CIRCLE_RADIOUS) {
					Results[(batchIndex * EVENTS_BATCH + block) * MAX_CIRCLES * 3 + numResult[batchIndex * EVENTS_BATCH + block] * 3 + 0] = Circles[i * 3 + 0];
					Results[(batchIndex * EVENTS_BATCH + block) * MAX_CIRCLES * 3 + numResult[batchIndex * EVENTS_BATCH + block] * 3 + 1] = Circles[i * 3 + 1];
					Results[(batchIndex * EVENTS_BATCH + block) * MAX_CIRCLES * 3 + numResult[batchIndex * EVENTS_BATCH + block] * 3 + 2] = Circles[i * 3 + 2];
					numResult[batchIndex*EVENTS_BATCH + block]++;
				}
			}
		}
		cudaFree(Circles);
	}
	cudaFree(&Errors);
	cudaFree(&Assignment);
	cudaFree(&changes);
}


