#pragma

double* randomCircles();
__global__ void computeErrors(double *Circles, double *Points, int PointsOffset, int *Assignment, int currentAssignmentOffset, int numCircles, float *Errors);
__global__ void fitCircles(double *Circles, double *Points, int PointsOffset, int *Assignment, int currentAssignmentOffset, int numPoints);
__global__ void distThread(double *Circles, double *Points, int PointsOffset, int numCircles, int *Assignment, int currentAssignmentOffset, int *changes);
__global__ void mainThread(double *Points, int *sumPoints, int *pointsPointers, double *randCircles, double *Result, int *numResult, int ep);
