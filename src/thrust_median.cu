#include "cuda_runtime.h"
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <thread>

#define NUM_GPUS 3

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

   return;
}

void GPU_thurst(thrust::host_vector<int> h_vec, int numValues, int *output){


  thrust::device_vector<int> d_vec = h_vec;

  // sort data on the device (846M keys per second on GeForce GTX 480)
  thrust::sort(d_vec.begin(), d_vec.end());

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), output);


  return ;
}

void multiGPU_thurst(thrust::host_vector<int> arr, int *output, int elements_per_GPU,int offset,int elementsOFFSET){

   thrust::host_vector<int> h_vec;
   cudaSetDevice(offset);

    int begin = elementsOFFSET*offset;
    int end = begin + elements_per_GPU;

   for(int i=begin;i<end;i++)
    h_vec.push_back(arr[i]);

  thrust::device_vector<int> d_vec = h_vec;

  // sort data on the device (846M keys per second on GeForce GTX 480)
  thrust::sort(d_vec.begin(), d_vec.end());

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), output);




  return;
}


int main( int argc, char *argv[]) {

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




  thrust::host_vector<int> vector;
    for (int i=numValues-1;i >= 0;i--){
        vector.push_back(i);
    }

  clock_t t; 
  t = clock(); 
  
 int *result = new int[numValues];
 if (NUM_GPUS == 1){

    GPU_thurst(vector,numValues,result);
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
      threads.push_back (std::thread ([elements_per_GPU, i,vector,partial_results] () {

        multiGPU_thurst(vector, partial_results[i],elements_per_GPU[i],i,elements_per_GPU[0]);

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
