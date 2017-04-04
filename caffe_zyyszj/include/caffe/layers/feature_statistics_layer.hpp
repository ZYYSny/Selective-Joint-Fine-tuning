#ifndef CAFFE_FEATURE_STATISTICS_LAYER_HPP_
#define CAFFE_FEATURE_STATISTICS_LAYER_HPP_

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
class FeatureStatisticsLayer : public Layer<Dtype> {
 public:
  explicit FeatureStatisticsLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FeatureStatistics"; }
  virtual inline int MinNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
      
  string feature_statistics_path_;
  string previous_statistics_path_;
  string new_feature_statistics_path_;
  int total_channels_;
  int saving_signal_;
  int interval_num_;
  int bin_;
  Blob<Dtype> feature_stats_;
  Blob<Dtype> bound_stats_;
  Blob<Dtype> refined_feature_stats_;
  Blob<Dtype> features_stats_bound_; 
};

}  // namespace caffe

#endif  // CAFFE_FEATURE_STATISTICS_LAYER_HPP_
