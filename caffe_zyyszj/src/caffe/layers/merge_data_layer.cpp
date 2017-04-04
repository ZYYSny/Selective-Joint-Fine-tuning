#include <cfloat>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "caffe/layers/merge_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MergeDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
}

template <typename Dtype>
void MergeDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape({bottom[0]->num()+bottom[1]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width()});
}

template <typename Dtype>
void MergeDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  caffe_cpu_scale(bottom[0]->count(), Dtype(1.0), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
  caffe_cpu_scale(bottom[1]->count(), Dtype(1.0), bottom[1]->cpu_data(), top[0]->mutable_cpu_data()+bottom[0]->count());
}

template <typename Dtype>
void MergeDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//TO DO
  caffe_cpu_scale(bottom[0]->count(), Dtype(1.0), top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
  caffe_cpu_scale(bottom[1]->count(), Dtype(1.0), top[0]->cpu_diff()+bottom[0]->count(), bottom[1]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(MergeDataLayer);
#endif

INSTANTIATE_CLASS(MergeDataLayer);
REGISTER_LAYER_CLASS(MergeData);

}  // namespace caffe
