#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "compute.h"
#include <cuda.h>

//__global__ void accelcreate(vector3** d_accels, vector3* d_values);
__global__ void pairwise( vector3* d_accels, vector3* d_hPos, double* d_mass);
__global__ void sumrows(vector3* d_accels, vector3* d_hVel, vector3* d_hPos);



//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	/*
	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3* d_values;
	cudaMalloc(&d_values, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	cudaMemcpy(d_values, values, sizeof(vector3)*NUMENTITIES*NUMENTITIES, cudaMemcpyHostToDevice);
	*/
/*
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (int i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];
	
	*/
	vector3* d_accels;
	cudaMalloc(&d_accels, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	//cudaMemcpy(d_accels, accels, sizeof(vector3)*NUMENTITIES*NUMENTITIES, cudaMemcpyHostToDevice);

	/*
	int accelgriddimension = (NUMENTITIES / 256) + 1;
	dim3 dimAccelGrid(accelgriddimension, 1);
	dim3 dimAccelBlock(256, 1);
	*/
	
	// accelcreate<<<dimAccelGrid,dimAccelBlock>>>(d_accels, d_values);

	double *d_mass;
	vector3 *d_hPos, *d_hVel;
	cudaMalloc(&d_mass, sizeof(double) * NUMENTITIES);
	cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMalloc(&d_hPos, sizeof(vector3) * NUMENTITIES);
	cudaMalloc(&d_hVel, sizeof(vector3) * NUMENTITIES);
	cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

	int griddimension = (NUMENTITIES / 16) + 1;
	dim3 dimGrid(griddimension, griddimension);
	dim3 dimBlock(16, 16);

	// i = threadindex.x + blockindex.x * blockdim.x
	// j = threadindex.y + blockindex.y * blockdim.y
	// k = threadindex.z

	pairwise<<<dimGrid,dimBlock>>>(d_accels, d_hPos, d_mass);

	sumrows<<<dimGrid,dimBlock>>>(d_accels, d_hVel, d_hPos);

	cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

	cudaFree(d_accels);
	cudaFree(d_mass);
	cudaFree(d_hPos);
	cudaFree(d_hVel);
}


/*
__global__ void accelcreate(vector3** d_accels, vector3* d_values){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < NUMENTITIES){
		d_accels[i]=&d_values[i*NUMENTITIES];
	}
}
*/



__global__ void pairwise( vector3* d_accels, vector3* d_hPos, double* d_mass){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int index = i * NUMENTITIES + j;
	if (i < NUMENTITIES && j < NUMENTITIES){
		if (i==j) {
			FILL_VECTOR(d_accels[index],0,0,0);
		}
			else{
				vector3 distance;
				for (int k=0;k<3;k++) distance[k]=d_hPos[i][k]-d_hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
				FILL_VECTOR(d_accels[index],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
	}
}

__global__ void sumrows(vector3* d_accels, vector3* d_hVel, vector3* d_hPos){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < NUMENTITIES){
		vector3 accel_sum={0,0,0};
		for (int j=0;j<NUMENTITIES;j++){
			for (int k=0;k<3;k++)
				accel_sum[k]+=d_accels[i * NUMENTITIES + j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (int k=0;k<3;k++){
			d_hVel[i][k]+=accel_sum[k]*INTERVAL;
			d_hPos[i][k]+=d_hVel[i][k]*INTERVAL;
		}
	}
}
