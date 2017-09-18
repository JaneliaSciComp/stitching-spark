#!/bin/bash

export SPARK_VERSION=2
export N_DRIVER_THREADS=1
export MEMORY_PER_NODE=105
export TERMINATE=1

N_NODES=$1;     shift

ARGS="$@ ''"

flintstone/flintstone.sh $N_NODES ../target/stitching-spark-0.0.1-SNAPSHOT.jar org.janelia.util.spark.RemoveDirectoryTreeSpark $ARGS