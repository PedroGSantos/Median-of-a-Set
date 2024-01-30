// Em desenvolvimento

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <vector>
#include <omp.h>

using namespace std;

void merge(vector<uint>& arr, int left, int mid, int right){
  int i, j, k, size;
  i = left; j = mid; k = 0; size = right - left;

  // Ordered auxiliary vector
  vector<uint> vecAux(size);

  while (i < mid && j < right) {
    if (arr[i] <= arr[j]) vecAux[k++] = arr[i++];
    else vecAux[k++] = arr[j++];
  }

  // Taking the remaining values
  while (i < mid) vecAux[k++] = arr[i++];
  while (j < right) vecAux[k++] = arr[j++];

  for (k = 0; k < size; k++)
    arr[left + k] = vecAux[k];
}

void mergeSort(vector<uint>& arr, int left, int right){
    if (right - left > 1) {
      int mid = (left + right) / 2;
      #pragma omp parallel sections
      {
        #pragma omp section
        {
          mergeSort(arr, left, mid);
        }
        #pragma omp section
        {
          mergeSort(arr, mid, right);
        }
      }
      merge(arr, left, mid, right);
    }
}

int main( int argc, char *argv[]) {

  vector<uint> vector;

//Standard parameters
  long int numValues = 1000000;


  std::string vec = "100000";
  std::string iter = "0";
  std::string tre = "1";
  if (argc == 4){
    numValues = atoi(argv[1]);
    vec = argv[1];
    tre = argv[2];
    iter = argv[3];
    printf("NUM VALUES SET TO %ld.\n",numValues);
    int num_threads = atoi(argv[2]);

	omp_set_num_threads(num_threads);
  }

  else{
        printf("NUM_VALUES NOT SETTED\n");
        exit(1);
  }



    for (int i=numValues-1;i >= 0;i--){
        vector.push_back(i);
    }

  clock_t t; 
  t = clock(); 


  mergeSort(vector, 0, vector.size());

  double median = 0;

  if (vector.size() % 2 == 1)
    median = vector[(vector.size()-1)/2];
  else
    median = (vector[(vector.size()/2)-1] + vector[vector.size()/2]) / 2.0;


    t = clock() - t; 
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 



  printf("The median is %.3lf\n",median);

  std::string folder = "results-";
    std::string standart_1 = "Time_";
    std::string standart_2 = ".txt";
    std::string path = folder + tre + "-"  + vec  + "/"  + standart_1 + vec + "_" + iter + standart_2;
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

