#ifndef CAFFE_NORMAL_KNN_MATCH_LAYER_HPP_
#define CAFFE_NORMAL_KNN_MATCH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class NormalKnnMatchLayer : public Layer<Dtype> {
 public:
  explicit NormalKnnMatchLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "NormalKnnMatch"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int sample_num_;
  int dim_;
  int nodes_num_;
  int orientation_;
  int scale_;
  int bin_;
  int iterations_;
  string feature_path_;
  string image_retrieval_result_;
  string channel_weight_path_;
  Blob<Dtype> features_;
  Blob<Dtype> labels_;
  Blob<Dtype> temp_;
  Blob<Dtype> max_heap_;
  Blob<Dtype> max_copy_;
  Blob<Dtype> pair_distance_;
  Blob<Dtype> channel_weights_;
};

}  // namespace caffe

#endif  // CAFFE_NORMAL_KNN_MATCH_LAYER_HPP_
