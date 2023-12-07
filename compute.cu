#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda.h>

__global__ void accelcreate(vector3** d_accels, vector3* d_values);
__global__ void pairwise( vector3** d_accels, vector3* d_hPos, double* d_mass);
__global__ void sumrows(vector3** d_accels, vector3* d_hVel, vector3* d_hPos);



//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3* d_values;
	cudaMalloc(&d_values, sizeof(values));
	cudaMemcpy(d_values, values, sizeof(values), cudaMemcpyHostToDevice);

	// vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	//for (i=0;i<NUMENTITIES;i++)
	//	accels[i]=&values[i*NUMENTITIES];
	
	vector3** d_accels;
	cudaMalloc(&d_accels, sizeof(vector3*)*NUMENTITIES);
	// cudaMemcpy(d_accels, accels, sizeof(accels), cudaMemcpyHostToDevice);

	int accelgriddimension;
	if (NUMENTITIES % 256 != 0){
		accelgriddimension = (NUMENTITIES / 256) + 1;
	} else {
		accelgriddimension = NUMENTITIES / 256;
	}
	dim3 dimAccelGrid(accelgriddimension, 1);
	dim3 dimAccelBlock(256, 1);
	
	accelcreate<<<dimAccelGrid,dimAccelBlock>>>(d_accels, d_values);


	int griddimension;
	if (NUMENTITIES % 16 != 0){
		griddimension = (NUMENTITIES / 16) + 1;
	} else {
		griddimension = NUMENTITIES / 16;
	}
	dim3 dimGrid(griddimension, griddimension);
	dim3 dimBlock(16, 16, 3);

	// i = threadindex.x + blockindex.x * blockdim.x
	// j = threadindex.y + blockindex.y * blockdim.y
	// k = threadindex.z

	double* d_mass;
	cudaMalloc(&d_hPos, sizeof(hPos));
	cudaMalloc(&d_hVel, sizeof(hVel));
	cudaMalloc(&d_mass, sizeof(mass));
	cudaMemcpy(d_hPos, hPos, sizeof(hPos), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hVel, hVel, sizeof(hVel), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(mass), cudaMemcpyHostToDevice);

	pairwise<<<dimGrid,dimBlock>>>(d_accels, d_hPos, d_mass);

	sumrows<<<dimGrid,dimBlock>>>(d_accels, d_hVel, d_hPos);

	cudaMemcpy(hVel, d_hVel, sizeof(d_hVel), cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos, d_hPos, sizeof(d_hPos), cudaMemcpyDeviceToHost);

	cudaFree(d_hPos);
	cudaFree(d_hVel);
	cudaFree(d_mass);
	cudaFree(d_values);
	cudaFree(d_accels);
	free(values);
}

__global__ void accelcreate(vector3** d_accels, vector3* d_values){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	d_accels[i]=&d_values[i*NUMENTITIES];
}

__global__ void pairwise( vector3** d_accels, vector3* d_hPos, double* d_mass){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z;
	if (i==j) {
		FILL_VECTOR(d_accels[i][j],0,0,0);
	}
		else{
			vector3 distance;
			distance[k]=d_hPos[i][k]-d_hPos[j][k];
			if (k == 0){
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
				FILL_VECTOR(d_accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
}

__global__ void sumrows(vector3** d_accels, vector3* d_hVel, vector3* d_hPos){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z;
	vector3 accel_sum={0,0,0};
	accel_sum[k]+=d_accels[i][j][k];

	if (j == 0){
		d_hVel[i][k]+=accel_sum[k]*INTERVAL;
		d_hPos[i][k]+=d_hVel[i][k]*INTERVAL;
	}
}
