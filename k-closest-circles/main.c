
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys\timeb.h>

#include "kcc.cuh"
#include "settings.h"

#ifdef __INTELLISENSE__
#include "intellisense_cuda_intrinsics.h"
#endif




// Struct for input data
struct inputDataStruct {
	double *eventPoints;		// Flattened list of all points
	int *eventPoints_num;		// Number of points in each event
	int *eventPoint_pointers;	// Indices of starting positions of each event in eventsPoints
	int numEvents;				// Total number of events
	int numPoints;				// Total number of points
};

typedef struct inputDataStruct inputData;


// Global Variables
double *Circles;

// GPU Variables
int *cnumResult;
double *cCircles, *cResult;

// Functions
void parallelEvents();
void readData2(char *);
int sum(int array[], int array_sz, int start, int end);
void printResults(inputData *data);





#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	}
}



void processBatches(inputData *data) {
	int pointsTmp = 0;
	int remainingEvents = data->numEvents;
	int batchIndex = 0, batchSize = 0;
	struct timeb start, end;

	// GPU vars
	double *points_gpu;
	int *eventPoints_num_gpu;
	int *pointsPointers_gpu;

	gpuErrchk(cudaMalloc((void **)&pointsPointers_gpu, data->numEvents * sizeof(int)));
	gpuErrchk(cudaMemcpy(pointsPointers_gpu, data->eventPoint_pointers, data->numEvents * sizeof(int), cudaMemcpyHostToDevice));		//Copy Point pointers

	gpuErrchk(cudaMalloc((void **)&points_gpu, data->numPoints * 2 * sizeof(double)));
	gpuErrchk(cudaMemcpy(points_gpu, data->eventPoints, data->numPoints * 2 * sizeof(double), cudaMemcpyHostToDevice));		//Copy Points

	gpuErrchk(cudaMalloc((void **)&eventPoints_num_gpu, data->numEvents * sizeof(int)));
	gpuErrchk(cudaMemcpy(eventPoints_num_gpu, data->eventPoints_num, data->numEvents * sizeof(int), cudaMemcpyHostToDevice));		//Copy Points

	printf("\n\nExecution started...");
	ftime(&start);

	while (remainingEvents > 0) {
		batchSize = remainingEvents >= EVENTS_BATCH ? EVENTS_BATCH : remainingEvents;

		mainThread <<<batchSize, 1>>> (points_gpu, eventPoints_num_gpu, pointsPointers_gpu, cCircles, cResult, cnumResult, batchIndex);
		gpuErrchk(cudaDeviceSynchronize());

		remainingEvents -= EVENTS_BATCH;
		batchIndex += 1;
	}

	ftime(&end);

	cudaFree(points_gpu);
	cudaFree(eventPoints_num_gpu);
	cudaFree(pointsPointers_gpu);

	printf("\nExecution finished in %u milliseconds\n\n", (int)(1000.0 * (end.time - start.time) + (end.millitm - start.millitm)));
	printf("Press a key to print the results and save them to file (Results.txt)...");
	getchar();
	printResults(data);
}



void printResults(inputData *data) {
	double *Result;
	int *numResult;

	FILE *myFile = fopen("Results.txt", "w");
	FILE *myFile_raw = fopen("Results_raw.txt", "w");
	Result = (double *)malloc(data->numEvents * 3 * MAX_CIRCLES * sizeof(double));
	numResult = (int *)malloc(data->numEvents * sizeof(int));
	gpuErrchk(cudaMemcpy(Result, cResult, data->numEvents * 3 * MAX_CIRCLES * sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(numResult, cnumResult, data->numEvents * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < data->numEvents; i++) {
		printf("Event: %d -> number of circles = %d\n", i + 1, numResult[i]);
		fprintf(myFile, "Event: %d -> number of circles = %d\n", i + 1, numResult[i]);
		fprintf(myFile_raw, "%d,%d\n", i + 1, numResult[i]);
		for (int j = 0; j < numResult[i]; j++) {
			printf("Circle %d => (%f, %f) - %f\n", (j + 1), Result[i * 3 * MAX_CIRCLES + j * 3 + 0], Result[i * 3 * MAX_CIRCLES + j * 3 + 1], Result[i * 3 * MAX_CIRCLES + j * 3 + 2]);
			fprintf(myFile, "Circle %d => (%f, %f) - %f\n", (j + 1), Result[i * 3 * MAX_CIRCLES + j * 3 + 0], Result[i * 3 * MAX_CIRCLES + j * 3 + 1], Result[i * 3 * MAX_CIRCLES + j * 3 + 2]);
			fprintf(myFile_raw, "%f, %f, %f\n", Result[i * 3 * MAX_CIRCLES + j * 3 + 0], Result[i * 3 * MAX_CIRCLES + j * 3 + 1], Result[i * 3 * MAX_CIRCLES + j * 3 + 2]);
		}
		printf("\n\n");
		fprintf(myFile, "\n\n");
	}

	cudaDeviceReset();
}



inputData* readData(char * file) {
	inputData *data = (inputData *) malloc(sizeof(inputData));
	if (data == NULL)
		exit(2);

	FILE *fp = fopen(file, "r");
	if (fp == NULL) {
		printf("File %s not found.\n", file);
		system("PAUSE");
		exit(0);
	}
	fscanf(fp, "%d\n", &data->numEvents);		//Get Number of Events
	data->numPoints = 0;

	data->eventPoints_num = (int *) malloc(data->numEvents * sizeof(int));
	data->eventPoint_pointers = (int *)malloc((data->numEvents + 1) * sizeof(int));
	data->eventPoints = (double *) malloc(sizeof(double));
	data->eventPoint_pointers[0] = 0;

	gpuErrchk(cudaMalloc((void **)&cResult, 3 * MAX_CIRCLES * data->numEvents * sizeof(double)));		// Stores the results (x, y, radius) for each event
	gpuErrchk(cudaMalloc((void **)&cnumResult, data->numEvents * sizeof(int)));						// Stores the number of circles for each event

	for (int i = 0; i < data->numEvents; i++) {
		fscanf(fp, "%d\n", &data->eventPoints_num[i]);
		data->eventPoints = (double *) realloc(data->eventPoints, (data->numPoints + data->eventPoints_num[i]) * 2 * sizeof(double));
		for (int point = 0; point < data->eventPoints_num[i]; point++) {
			fscanf(fp, "%lf %lf\n", &data->eventPoints[data->numPoints * 2 + point * 2], &data->eventPoints[data->numPoints * 2 + point * 2 + 1]);	// Get Points - flattened
		}
		data->numPoints += data->eventPoints_num[i];
		data->eventPoint_pointers[i + 1] = data->eventPoint_pointers[i] + 2 * data->eventPoints_num[i];
	}

	return data;
}




int main(int argc, char *argv[]) {
	if (argc == 1)
		printf("\nNo input data file given.\n\nDefault file name 'batch00.dat'");
	
	Circles = randomCircles();

	// Copy the initial random cycles to GPU
	gpuErrchk(cudaSetDevice(0));
	gpuErrchk(cudaMalloc((void **)&cCircles, MAX_CIRCLES * 3 * sizeof(double)));
	gpuErrchk(cudaMemcpy(cCircles, Circles, MAX_CIRCLES * 3 * sizeof(double), cudaMemcpyHostToDevice));
	
	// Read input data
	inputData *data = readData(argc > 1? argv[1] : "Input/batch00.dat");
	processBatches(data);

	fflush(stdout);
	system("PAUSE");

	return 0;
}


