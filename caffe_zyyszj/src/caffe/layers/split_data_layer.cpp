#include <cfloat>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "caffe/layers/split_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SplitDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  split_num_ = this->layer_param_.split_data_param().split_num();
  CHECK_LT(split_num_,bottom[0]->num());
  CHECK_GT(split_num_,0);
}

template <typename Dtype>
void SplitDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape({split_num_, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width()});
  top[1]->Reshape({(bottom[0]->num()-split_num_), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width()});
}

template <typename Dtype>
void SplitDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int dim=bottom[0]->channels()*bottom[0]->height()*bottom[0]->width();
  caffe_cpu_scale(split_num_*dim, Dtype(1.0), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
  caffe_cpu_scale((bottom[0]->num()-split_num_)*dim, Dtype(1.0), bottom[0]->cpu_data()+split_num_*dim, top[1]->mutable_cpu_data());
}

template <typename Dtype>
void SplitDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int dim=bottom[0]->channels()*bottom[0]->height()*bottom[0]->width();
  CHECK_EQ(split_num_*dim,top[0]->count());
  CHECK_EQ((bottom[0]->num()-split_num_)*dim,top[1]->count());
  caffe_cpu_scale(split_num_*dim, Dtype(1.0), top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
  caffe_cpu_scale((bottom[0]->num()-split_num_)*dim, Dtype(1.0), top[1]->cpu_diff(), bottom[0]->mutable_cpu_diff()+split_num_*dim);
}

#ifdef CPU_ONLY
STUB_GPU(SplitDataLayer);
#endif

INSTANTIATE_CLASS(SplitDataLayer);
REGISTER_LAYER_CLASS(SplitData);

}  // namespace caffe
