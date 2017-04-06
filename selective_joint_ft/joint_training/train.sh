#!/usr/bin/env sh

export PATH=/usr/local/cuda-7.5/bin/$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:build/lib$LD_LIBRARY_PATH

TOOLS=.build_release/tools
WORK_FOLDER=examples/cvpr2017/mit67/joint_training_with_imagenet/first_round

echo "Start joint training identity mapping..."
    
GLOG_logtostderr=1 $TOOLS/caffe \
    train \
    -solver \
    $WORK_FOLDER/solver.prototxt \
    -weights \
    examples/cvpr2017/mit67/places_fine_tuning/snapshot/identity_mapping_resnet152_imagenet_places_hybrid_places_finetuned_iter_2000.caffemodel \
    -gpu \
    0 2>&1 | tee $WORK_FOLDER/loss.log
echo "Done."
