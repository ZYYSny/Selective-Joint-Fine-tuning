#!/usr/bin/env sh

export PATH=/usr/local/cuda-7.5/bin/$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:build/lib$LD_LIBRARY_PATH

TOOLS=.build_release/tools
WORK_FOLDER=examples/cvpr2017/mit67/joint_training_with_selected_samples/first_round

echo "Start joint training identity mapping..."
    
GLOG_logtostderr=1 $TOOLS/caffe \
    train \
    -solver \
    $WORK_FOLDER/solver.prototxt \
    -weights \
    models/resnet_imagenet/identity_mapping_resnet152_imagenet_iter_50000.caffemodel \
    -gpu \
    3 2>&1 | tee $WORK_FOLDER/loss.log
echo "Done."
