#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>

using namespace std;

int main( int argc, char *argv[]) {

  vector<int> vector;


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

    for (int i=numValues-1;i >= 0;i--){
        vector.push_back(i);
    }

  clock_t t; 
  t = clock(); 


  // sort data on the device (846M keys per second on GeForce GTX 480)
  sort(vector.begin(), vector.end());

  double median = 0;

  if (vec.size() % 2 == 1)
    median = vector[(vec.size()-1)/2] ;
  else
    median = (vector[(vec.size()/2)-1] + vector[vector.size()/2]) / 2.0;


  t = clock() - t; 
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 

  printf("The median is %.3lf\n",median);

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
