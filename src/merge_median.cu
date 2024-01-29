#include <iostream>
#include <cstdio>
#include <algorithm>
#include <getopt.h>
#include <string.h>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <assert.h>
#include <thread>
#include <vector>


// https://github.com/G-U-N/Parallel-Computing-Project/blob/master/cuda-sort-project/sort.cu

#define NUM_GPUS 3



__device__ void merge(int l,int m,int r,int data[],int tmp[])
{
    int i=l,j=m,k=l;
    while (i<m&&j<r)
    {
        if (tmp[i]<=tmp[j])
        {
            data[k++]=tmp[i++];
        }
        else
        {
            data[k++]=tmp[j++];
        }
    }
    while (i<m) data[k++]=tmp[i++];
    while (j<r) data[k++]=tmp[j++];
}



__global__ void
merge_kernel(int N, int chunk,int data[],int tmp[]) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index>N) return ;
    //当数据量过多时，这里会发生溢出。暂时使用变负数。
    int start=index*chunk;
    if (start>=N || start<0) return ;

    int left=start;
    int mid=min(start+(int)(chunk/2),N);
    int right=min(start+chunk,N);
    // printf("l=%d,m=%d,r=%d\n",left,mid,right);
    // if (start<0) assert(0);
    merge(left, mid,right,data,tmp);
}

//cuda merge sort 就是chunk取2的幂次然后每个kernel会出里start+chunk的数据



void mergeSort(int N,int *input,int *output,int device=0)
{
    cudaSetDevice(device);

    int *device_i;
    int *tmp;
    cudaMalloc((void **)&device_i, N*sizeof(int));
    cudaMalloc((void **)&tmp, N*sizeof(int));


    cudaMemcpy(device_i, input, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tmp, input, N*sizeof(int), cudaMemcpyHostToDevice);


    
    for (int chunk=2;chunk<2*N;chunk*=2)
    {
        // const int threadsPerBlock = 512;
        const int threadsPerBlock=1;
        const int blocks = ((N + threadsPerBlock*chunk - 1) / (threadsPerBlock*chunk));
        merge_kernel<<<blocks,threadsPerBlock>>>(N,chunk,device_i,tmp);
        cudaDeviceSynchronize();
        merge_kernel<<<blocks,threadsPerBlock>>>(N,chunk,tmp,device_i);
        cudaDeviceSynchronize();
    }
    

    cudaMemcpy(output, device_i, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }



    cudaFree(device_i);
    cudaFree(tmp);
}

void mergeSortMultiGPU(int N,int *input,int *output,int offset,int device=0)
{
    cudaSetDevice(device);

    int *device_i;
    int *tmp;
    cudaMalloc((void **)&device_i, N*sizeof(int));
    cudaMalloc((void **)&tmp, N*sizeof(int));


    cudaMemcpy(device_i, &input[offset], N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tmp, &input[offset], N*sizeof(int), cudaMemcpyHostToDevice);


    
    for (int chunk=2;chunk<2*N;chunk*=2)
    {
        // const int threadsPerBlock = 512;
        const int threadsPerBlock=1;
        const int blocks = ((N + threadsPerBlock*chunk - 1) / (threadsPerBlock*chunk));
        merge_kernel<<<blocks,threadsPerBlock>>>(N,chunk,device_i,tmp);
        cudaDeviceSynchronize();
        merge_kernel<<<blocks,threadsPerBlock>>>(N,chunk,tmp,device_i);
        cudaDeviceSynchronize();
    }
    

    cudaMemcpy(output, device_i, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }



    cudaFree(device_i);
    cudaFree(tmp);
}


void calculateElements(int *elementsPerGPU,int shards_num,long int vectorSize){

        //Define o tamanho de cada shard
    for (int i=0;i<shards_num;i++){

        if (i != (shards_num)-1){ elementsPerGPU[i] = vectorSize / (shards_num);}

        else{
            elementsPerGPU[i] = vectorSize;

            for (int j=0;j<i;j++)
                elementsPerGPU[i] -= elementsPerGPU[j];
            

        }
    }

    return;
}

void merging_partial_results(int *elementsPerGPU, int **partial_results,int numValues,int *results){

  int control[NUM_GPUS];

  for(int i=0;i<NUM_GPUS;i++)
    control[i] = 0;


  for(int i=0;i<numValues;i++){

    int menor = INT_MAX;
    int idx_menor = -1;

    for(int j=0;j<NUM_GPUS;j++){

      if (control[j] == elementsPerGPU[j]){continue;}


      else if (j == 0){ menor = partial_results[j][control[j]]; idx_menor = j;}

      else{ if (menor > partial_results[j][control[j]] ) {menor = partial_results[j][control[j]]; idx_menor = j;}}

    }
    control[idx_menor] += 1;
    results[i] = menor;
  }

  for(int i=0;i<NUM_GPUS;i++)
    printf("%d -> %d (%d)", i, control[i], elementsPerGPU[i]);

  

}

void printArray(int* A, int size)
{
	int i;
	for (i = 0; i < size; i++)
		printf("%d ", A[i]);
	printf("\n");
}

void printArray2(int* A, int size)
{
	int i;
	for (i = size-1; i < size-10; i++)
		printf("%d ", A[i]);
	printf("\n");
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
  int*  result = new int[numValues];


  int j=0;
  for (int i=numValues-1;i>=0;i--){
    arr[j] = i;
    j += 1;
  }

  clock_t t; 
  t = clock(); 

  if (NUM_GPUS == 1){
    mergeSort(numValues,arr,result,0);
  }
  else{

    int *elements_per_GPU;
    elements_per_GPU = (int*)malloc(NUM_GPUS*sizeof(int));
    int *partial_results[NUM_GPUS];


    calculateElements(elements_per_GPU,NUM_GPUS,numValues);

    for(int i=0;i<NUM_GPUS;i++)
      partial_results[i] = new int[elements_per_GPU[i]];

    std::vector<std::thread> threads;

    for (int i=0;i<NUM_GPUS;i++){
      threads.push_back (std::thread ([elements_per_GPU, i,arr,partial_results] () {

        

        mergeSortMultiGPU(elements_per_GPU[i],arr,partial_results[i],i*elements_per_GPU[0],i);
      })); 
  }

    for (auto &t: threads)
    t.join ();  



  merging_partial_results(elements_per_GPU, partial_results,numValues,result);


  }


  double median = 0;

  if (numValues % 2 == 1){
   median = result[(numValues-1)/2] ;
  }

  else
    median = (result[(numValues/2)-1] + result[numValues/2]) / 2.0;

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
