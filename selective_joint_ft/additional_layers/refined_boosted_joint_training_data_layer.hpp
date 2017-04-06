#ifndef CAFFE_REFINED_BOOSTED_JOINT_TRAINING_DATA_LAYER_HPP_
#define CAFFE_REFINED_BOOSTED_JOINT_TRAINING_DATA_LAYER_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class RefinedBoostedJointTrainingDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit RefinedBoostedJointTrainingDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~RefinedBoostedJointTrainingDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RefinedBoostedJointTrainingData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleVectors(vector<int> &image_list);
  virtual void load_batch(Batch<Dtype>* batch);

  //Boosted Joint Training Parameters
  string source_domain_image_label_path_;
  string target_domain_image_label_path_;
  int source_domain_sample_num_;
  int source_domain_class_num_;
  int target_domain_sample_num_;
  int target_domain_class_num_;
  string target_domain_sample_weight_path_;
  string target_nn_configuration_path_;
  int target_domain_train_num_;
  int target_domain_val_num_;
  int target_domain_train_val_num_;
  int target_nn_in_source_num_;
  int base_selected_nn_num_;
  int highest_sample_level_;
  int least_sample_per_class_;
  string target_val_label_path_;
  //data structure
  vector<vector<int>> source_domain_image_sourcedomain_label_list_;
  vector<pair<string, int>> source_domain_image_global_label_list_;
  vector<vector<int>> target_domain_image_targetdomain_label_list_;
  vector<vector<pair<string, int>>> target_domain_image_global_label_list_;
  vector<vector<int>> target_nearest_neighbor_in_source_data_;
  vector<vector<int>> refined_target_nearest_neighbor_in_source_data_;
  vector<int> target_selected_sample_num_in_source_;
  vector<int> target_domain_sampling_level_;
  vector<int> target_train_val_label_;
  //
  vector<int> target_sampling_list_during_training_;
  map<int, int> refined_selected_source_domain_samples_;
  map<int, map<int,int>> refined_selected_source_domain_samples_in_newlabel_;
  int lines_id_;
};


}  // namespace caffe

#endif  // CAFFE_REFINED_BOOSTED_JOINT_TRAINING_DATA_LAYER_HPP_
