#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#define MAX_THREADS_PER_BLOCK 1024

//GPU Kernel Implementation of Bitonic Sort
__global__ void bitonicSortGPU_(int* arr, int j, int k)
{
    unsigned int i, ij;

    i = threadIdx.x + blockDim.x * blockIdx.x;

    ij = i ^ j;

    if (ij > i)
    {
        if ((i & k) == 0)
        {
            if (arr[i] > arr[ij])
            {
                int temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
        else
        {
            if (arr[i] < arr[ij])
            {
                int temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
    }
}

void bitonicSortGPU(int *arr,int size){

    int* gpuArrbiton;


    //Set number of threads and blocks for kernel calls
    int threadsPerBlock = MAX_THREADS_PER_BLOCK;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;


    cudaMalloc((void**)&gpuArrbiton, size * sizeof(int));

    // Copy the input array to GPU memory
    cudaMemcpy(gpuArrbiton, arr, size * sizeof(int), cudaMemcpyHostToDevice);

    for (int k = 2; k <= size; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j = j >> 1)
        
            bitonicSortGPU_<<<blocksPerGrid, threadsPerBlock >>> (gpuArrbiton, j, k);
        
    }

    cudaMemcpy(arr, gpuArrbiton, size * sizeof(int), cudaMemcpyDeviceToHost);
    return;


}
int main(int argc, char *argv[]){

  //Standard parameters
  long int numValues = 1000000;

  std::string vec = "100000";
  std::string iter = "0";
  if (argc == 3){
    numValues = atoi(argv[1]);
    vec = argv[1];
    iter = argv[2];
    printf("NUM VALUES SET TO %ld.\n",numValues);
  }
  
  else{
        printf("NUM_VALUES NOT SETTED\n");
        exit(1);
  }

  //Create CPU based Arrays
  int* arr = new int[numValues];

int j=0;
  for (int i=numValues-1;i>=0;i--){
    arr[j] = i;
    j += 1;
  }

  clock_t t; 
  t = clock();  
  bitonicSortGPU(arr,numValues);

  double median = 0;

  if (numValues % 2 == 1){
   median = arr[(numValues-1)/2] ;
  }

  else
    median = (arr[(numValues/2)-1] + arr[numValues/2]) / 2.0;

 
  t = clock() - t; 
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 

  printf("The median is %f\n",median);


  std::string folder = "results-";
  std::string standart_1 = "Time_";
  std::string standart_2 = ".txt";
  std::string path = folder + vec  + "/"  + standart_1 + vec + "_" + iter + standart_2;
  FILE *pFile;

    pFile=fopen(path.c_str(), "a");

  if(pFile==NULL) {
        perror("Error opening file.");
    }
else {

        fprintf(pFile, "%lf\n", time_taken); fprintf(pFile, "MEDIA: %lf", median);
    }

  fclose(pFile);

  return 0;


}
