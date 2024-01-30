#!/bin/bash

function clean_job() {
          echo "Limpando ambiente..."
            rm -rf "${local_job}"
    }
repeat(){

         cd thurst-benchmarking
        ./run.sh
        cd .. 
        cd multi-merge-benchmarking
        ./run.sh     
        cd ..  
        cd sortcpp-benchmarking
        ./run.sh
        cd ..
        cd seq-merge-benchmarking
        ./run.sh
        cd .. 
        cd base-benchmarking
        ./run.sh
        exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit


