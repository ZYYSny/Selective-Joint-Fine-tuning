#!/usr/bin/env sh

export PATH=/usr/local/cuda-7.5/bin/$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:build/lib$LD_LIBRARY_PATH

TOOLS=.build_release/tools
WORK_FOLDER=examples/cvpr2017/mit67/image_retrieval/feature_extraction

echo "Start Extracting Features of Alex Net..."
#GLOG_logtostderr=1     
$TOOLS/caffe \
    train \
    -solver \
    $WORK_FOLDER/feature_extractor_solver.prototxt \
    -weights \
    models/bvlc_alexnet/bvlc_alexnet.caffemodel \
    -gpu \
    3 2>&1 | tee $WORK_FOLDER/loss.log
echo "Done."