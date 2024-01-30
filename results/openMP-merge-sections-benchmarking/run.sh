#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){


        for i in 1073741824 #1024 8192 65536 524288 4194304 33554432 268435456 1073741824
        do 

        	for threads in  4 8 16 32 64 128 
		do
			mkdir results-$threads-$i
			for iters in 0 
			do

			./main $i $threads $iters

			done
		done

         done 
	exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit
