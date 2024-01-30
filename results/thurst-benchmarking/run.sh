#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){


        for i in 1024 8192 65536 524288 4194304 33554432 268435456 #1073741824
        do 

	mkdir results-$i        
        for iters in 0 1 2 3 4 5 6 7 8 9
        do

        ./main $i $iters

        done

         done 
	exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit
