#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/refined_boosted_joint_training_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

std::vector<string> refined_boosted_joint_training_data_split(const string &s, const string &seperator) {
	vector<string> result;
	typedef string::size_type string_size;
	string_size i = 0;

	while (i != s.size()) {

		int flag = 0;
		while (i != s.size() && flag == 0) {
			flag = 1;
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[i] == seperator[x]) {
					++i;
					flag = 0;
					break;
				}
		}


		flag = 0;
		string_size j = i;
		while (j != s.size() && flag == 0) {
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[j] == seperator[x]) {
					flag = 1;
					break;
				}
			if (flag == 0)
				++j;
		}
		if (i != j) {
			result.push_back(s.substr(i, j - i));
			i = j;
		}
	}
	return result;
}

template <typename Dtype>
RefinedBoostedJointTrainingDataLayer<Dtype>::~RefinedBoostedJointTrainingDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void RefinedBoostedJointTrainingDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  //Designate each class and the corresponding images
  BoostedJointTrainingDataParameter boosted_joint_training_data_param_ = 
    this->layer_param_.boosted_joint_training_data_param();
  source_domain_image_label_path_ = boosted_joint_training_data_param_.source_domain_image_label_path();
  target_domain_image_label_path_ = boosted_joint_training_data_param_.target_domain_image_label_path();
  source_domain_sample_num_ = boosted_joint_training_data_param_.source_domain_sample_num();
  source_domain_class_num_ = boosted_joint_training_data_param_.source_domain_class_num();
  target_domain_sample_num_ = boosted_joint_training_data_param_.target_domain_sample_num();
  target_domain_class_num_ = boosted_joint_training_data_param_.target_domain_class_num();
  target_domain_sample_weight_path_ = boosted_joint_training_data_param_.target_domain_sample_weight_path();
  target_nn_configuration_path_ = boosted_joint_training_data_param_.target_nn_configuration_path();
  target_domain_train_num_ = boosted_joint_training_data_param_.target_domain_train_num();
  target_domain_val_num_ = boosted_joint_training_data_param_.target_domain_val_num();
  target_nn_in_source_num_ = boosted_joint_training_data_param_.target_nn_in_source_num();
  base_selected_nn_num_ = boosted_joint_training_data_param_.base_selected_nn_num();
  target_val_label_path_ = boosted_joint_training_data_param_.target_val_label_path();
  target_domain_train_val_num_ = boosted_joint_training_data_param_.target_domain_train_val_num();
  highest_sample_level_ = boosted_joint_training_data_param_.highest_sample_level();
  least_sample_per_class_ = boosted_joint_training_data_param_.least_sample_per_class();
  CHECK_EQ(target_domain_train_val_num_, target_domain_train_num_+target_domain_val_num_);
  //Read source domain data and label
  string path;
  int global_label, domain_label, idx;
  std::ifstream input_source_file(source_domain_image_label_path_.c_str());
  source_domain_image_sourcedomain_label_list_.resize(source_domain_class_num_);
  idx = 0;
  LOG(INFO)<<"Loading source data and labels...";
  map<int, int> source_global_domain_list;
  while(input_source_file>>path>>global_label>>domain_label){
    source_domain_image_global_label_list_.push_back(make_pair(path,global_label));
    CHECK_GE(global_label, 0);
    CHECK_GE(domain_label, 0);
    CHECK_LT(domain_label, source_domain_class_num_);
    source_domain_image_sourcedomain_label_list_[domain_label].push_back(idx);
    source_global_domain_list[global_label]=domain_label;
    idx++;
  }
  input_source_file.close();
  CHECK_EQ(idx, source_domain_sample_num_);
  CHECK_EQ(source_domain_image_global_label_list_.size(), source_domain_sample_num_);
  LOG(INFO)<<"  Loading source data and labels finished!";
  //
  std::ifstream input_target_file(target_domain_image_label_path_.c_str());
  int img_idx;
  target_domain_image_targetdomain_label_list_.resize(target_domain_class_num_);
  target_domain_image_global_label_list_.resize(target_domain_train_num_);
  map<int,int> img_idx_container;
  map<int,int>::iterator img_idx_container_iter;
  target_train_val_label_.resize(target_domain_train_val_num_);
  LOG(INFO)<<"Loading target data and labels...";
  while(input_target_file>>path>>global_label>>domain_label>>img_idx){
    target_domain_image_global_label_list_[img_idx].push_back(make_pair(path,global_label));
    img_idx_container_iter = img_idx_container.find(img_idx);
    if(img_idx_container_iter==img_idx_container.end()){
      img_idx_container[img_idx] = 1;
      target_domain_image_targetdomain_label_list_[domain_label].push_back(img_idx);
    }
    target_train_val_label_[img_idx]=domain_label;
  }
  input_target_file.close();
  LOG(INFO)<<"  Loading target data and labels finished!";
  idx=0;
  for(int k=0;k<target_domain_train_num_;k++){
	  //LOG(INFO)<<k<<"\t"<<target_domain_image_global_label_list_[k].size()<<"\t"<<idx;
    idx += target_domain_image_global_label_list_[k].size();
  }
  LOG(INFO)<<idx;
  CHECK_EQ(idx, target_domain_sample_num_);
  //
  std::ifstream input_target_val_file(target_val_label_path_.c_str());
  LOG(INFO)<<"Loading target train & val indexes and labels...";
  while(input_target_val_file>>domain_label>>img_idx){
    img_idx_container_iter = img_idx_container.find(img_idx);
    if(img_idx_container_iter==img_idx_container.end()){
      img_idx_container[img_idx] = 1;
      //target_domain_image_targetdomain_label_list_[domain_label].push_back(img_idx);
    }
    target_train_val_label_[img_idx]=domain_label;   
  }
  input_target_val_file.close();
  LOG(INFO)<<"  Loading target train & val indexes and labels finished!";
  idx=0;
  for(int k=0;k<target_domain_class_num_;k++){
    idx += target_domain_image_targetdomain_label_list_[k].size();
  }
  CHECK_EQ(idx, target_domain_train_num_); 
  //
  target_nearest_neighbor_in_source_data_.resize(target_domain_train_val_num_);
  std::ifstream input_target_nn_config_file(target_nn_configuration_path_.c_str());
  string line;
  LOG(INFO)<<"Loading target nearest neighbor indexes...";
  while(!input_target_nn_config_file.eof()){
    std::getline(input_target_nn_config_file, line);
    if(line=="")continue;
    vector<string> temp_split_str = refined_boosted_joint_training_data_split(line,"\t ");
    CHECK_EQ(temp_split_str.size(), target_nn_in_source_num_+2);
    int img_idx = stoi(temp_split_str[0]);
    for(int k=0;k<target_nn_in_source_num_;k++){
      target_nearest_neighbor_in_source_data_[img_idx].push_back(stoi(temp_split_str[k+1]));
    }
  }
  LOG(INFO)<<"  Loading target nearest neighbor indexes finished!";
  input_target_nn_config_file.close();
  //
  target_selected_sample_num_in_source_.resize(target_domain_train_val_num_, base_selected_nn_num_);
  //
  std::ifstream input_sample_weight_file(target_domain_sample_weight_path_.c_str());
  int sampling_level;
  target_domain_sampling_level_.resize(target_domain_train_val_num_,0);
  //LOG(INFO)<<"Loading target sampling weight...";
  while(input_sample_weight_file>>img_idx>>sampling_level){
    CHECK_GE(img_idx,0);
    CHECK_LT(img_idx,target_domain_train_val_num_);
    if(sampling_level>highest_sample_level_)sampling_level=highest_sample_level_;
    target_domain_sampling_level_[img_idx]=sampling_level;
    target_selected_sample_num_in_source_[img_idx] = base_selected_nn_num_*sampling_level;
    CHECK_LE(target_selected_sample_num_in_source_[img_idx], target_nn_in_source_num_);
  }
  LOG(INFO)<<"  Loading target sampling weight finished!";
  input_sample_weight_file.close();
  //
  target_sampling_list_during_training_.resize(0);
  for(int k=0;k<target_domain_train_val_num_;k++){
    for(int m=0;m<target_domain_sampling_level_[k];m++){
      target_sampling_list_during_training_.push_back(k);
    }
  }
  LOG(INFO)<<"There are "<<target_domain_train_val_num_<< "idependent training samples.";
  LOG(INFO)<<"   Source images: "<<source_domain_image_global_label_list_.size()<<",";
  LOG(INFO)<<"   Target images: "<<target_domain_sample_num_<<".";
  
  //refined joint training samples

  //get all selected training images
  map<int, map<int,int>> selected_source_domain_samples;
  map<int, map<int,int>>::iterator selected_source_domain_samples_iter;
  map<int, int>::iterator source_global_domain_list_iter;
  int count_idx = 0;
  for(int k=0;k<target_domain_train_val_num_;k++){
    for(int m=0;m<target_selected_sample_num_in_source_[k];m++){
      int source_img_idx = target_nearest_neighbor_in_source_data_[k][m];
      source_global_domain_list_iter = source_global_domain_list.find(source_img_idx);
      if(source_global_domain_list_iter == source_global_domain_list.end()){
        LOG(FATAL)<<"Can't find the source image index "<<source_img_idx;
      }
      int source_img_label = source_global_domain_list_iter->second;
      selected_source_domain_samples_iter = selected_source_domain_samples.find(source_img_label);
      if(selected_source_domain_samples_iter==selected_source_domain_samples.end()){
        map<int,int> temp_pair;
        temp_pair[source_img_idx] = 1;
        selected_source_domain_samples[source_img_label] = temp_pair;
      }else{
        selected_source_domain_samples_iter->second[source_img_idx] = 1;
      }
      count_idx++;
    }
  }
  LOG(INFO)<<selected_source_domain_samples.size()<<" classes, "<<count_idx<<" samples in the training list!";
  //
  count_idx=0;
  for(selected_source_domain_samples_iter=selected_source_domain_samples.begin();
    selected_source_domain_samples_iter!=selected_source_domain_samples.end();
    selected_source_domain_samples_iter++){
    if(selected_source_domain_samples_iter->second.size()<least_sample_per_class_){
      continue;
    }
    refined_selected_source_domain_samples_in_newlabel_[count_idx] = selected_source_domain_samples_iter->second;
    for(source_global_domain_list_iter = selected_source_domain_samples_iter->second.begin();
      source_global_domain_list_iter != selected_source_domain_samples_iter->second.end();
      source_global_domain_list_iter++){
      int source_img_idx = source_global_domain_list_iter->first;
      refined_selected_source_domain_samples_[source_img_idx] = count_idx; 
    }
    count_idx++;
  }
  LOG(INFO)<<"There are "<<refined_selected_source_domain_samples_.size()<<" training samples and "<<
    refined_selected_source_domain_samples_in_newlabel_.size() << " classes remained.";
  //
  refined_target_nearest_neighbor_in_source_data_.resize(target_domain_train_val_num_);
  for(int k=0;k<target_domain_train_val_num_;k++){
    for(int m=0;m<target_selected_sample_num_in_source_[k];m++){
      int source_img_idx = target_nearest_neighbor_in_source_data_[k][m];
      source_global_domain_list_iter = refined_selected_source_domain_samples_.find(source_img_idx);
      if(source_global_domain_list_iter == refined_selected_source_domain_samples_.end()){
        continue;
      }
      refined_target_nearest_neighbor_in_source_data_[k].push_back(source_img_idx);
      source_domain_image_global_label_list_[source_img_idx].second = source_global_domain_list_iter->second;
    }
    CHECK_GT(refined_target_nearest_neighbor_in_source_data_[k].size(),0);
  }  
  //

  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  ShuffleVectors(target_sampling_list_during_training_);
  lines_id_ = 0;

  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + source_domain_image_global_label_list_[0].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << source_domain_image_global_label_list_[0].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void RefinedBoostedJointTrainingDataLayer<Dtype>::ShuffleVectors(vector<int> &image_list) {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(image_list.begin(), image_list.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void RefinedBoostedJointTrainingDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  //LOG(INFO)<<"Construct the mini-batch!";
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  //LOG(INFO)<<"Construct the mini-batch!";
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();
  
  vector<std::pair<std::string, int> > batch_lines_;
  int target_batch_size = batch_size/2;
  CHECK_EQ(batch_size%2, 0)<<"Batch Size must can be divided by 2!";
  //Add target and source data
  batch_lines_.resize(batch_size);
  //LOG(INFO)<<"Construct the mini-batch!";
  for(int k=0;k<target_batch_size;k++){
    //LOG(INFO)<<k<<"\t"<<lines_id_;
    int target_img_idx = target_sampling_list_during_training_[lines_id_];
    //LOG(INFO)<<k<<"\t"<<lines_id_<<"\t"<<target_img_idx;
    int source_nn_location_idx = caffe_rng_rand()%refined_target_nearest_neighbor_in_source_data_[target_img_idx].size();
	  int source_nn_idx = refined_target_nearest_neighbor_in_source_data_[target_img_idx][source_nn_location_idx];
    //LOG(INFO)<<source_nn_location_idx<<"\t"<<source_nn_idx;
    batch_lines_[k+target_batch_size] = source_domain_image_global_label_list_[source_nn_idx];
	  //LOG(INFO)<<target_img_idx<<"\t"<<source_nn_location_idx<<"\t"<<source_nn_idx<<"\t"<<source_domain_image_global_label_list_[source_nn_idx].first<<"\t"<<source_domain_image_global_label_list_[source_nn_idx].second;
    if(target_img_idx>=target_domain_train_num_){
      int target_img_label = target_train_val_label_[target_img_idx];
	  //LOG(INFO)<<target_img_label;
      int temp_img_idx = caffe_rng_rand()%target_domain_image_targetdomain_label_list_[target_img_label].size();
      target_img_idx = target_domain_image_targetdomain_label_list_[target_img_label][temp_img_idx];
	  //LOG(INFO)<<target_img_idx<<"\t"<<temp_img_idx<<"\t"<<target_domain_image_targetdomain_label_list_[target_img_label].size();
    }
	//LOG(INFO)<<target_domain_image_global_label_list_.size()<<"\t"<<target_domain_image_global_label_list_[target_img_idx].size();
    int target_img_path_idx = caffe_rng_rand()%target_domain_image_global_label_list_[target_img_idx].size();
	//LOG(INFO)<<target_img_path_idx<<"\t"<<target_domain_image_global_label_list_[target_img_idx].size();
    batch_lines_[k] = target_domain_image_global_label_list_[target_img_idx][target_img_path_idx];
    batch_lines_[k].second = target_train_val_label_[target_img_idx];
	//LOG(INFO)<<target_img_idx<<"\t"<<target_img_path_idx<<"\t"<<batch_lines_[k].first<<"\t"<<batch_lines_[k].second;
    lines_id_++;
    if(lines_id_>=target_sampling_list_during_training_.size()){
      ShuffleVectors(target_sampling_list_during_training_);
      lines_id_=0;
    }
  }
  
  map<int, int> labels_int_batch;
  map<int, int>::iterator labels_int_batch_iter;
  for(int k=0;k<target_batch_size;k++){
    int img_label = batch_lines_[k].second;
    labels_int_batch_iter = labels_int_batch.find(img_label);
    if(labels_int_batch_iter==labels_int_batch.end()){
      labels_int_batch[img_label] = 1;
    }else{
      labels_int_batch_iter->second++;
    }
  }
  if(labels_int_batch.size()<=target_batch_size){
    vector<int> remained_label_list;
    for(int k=0;k<target_domain_class_num_;k++){
      labels_int_batch_iter = labels_int_batch.find(k);
      if(labels_int_batch_iter==labels_int_batch.end()){
        remained_label_list.push_back(k);
      }
    }
    ShuffleVectors(remained_label_list);
    int label_index = 0;
    for(int k=0;k<target_batch_size;k++){
      int img_label = batch_lines_[k].second;
      labels_int_batch_iter = labels_int_batch.find(img_label);
      if(labels_int_batch_iter==labels_int_batch.end()){
        LOG(FATAL)<<"Image label "<<img_label<<" is not found!";
      }else{
        if(labels_int_batch_iter->second>2){
          int target_img_label = remained_label_list[label_index];
          int temp_img_idx = caffe_rng_rand()%target_domain_image_targetdomain_label_list_[target_img_label].size();
          int target_img_idx = target_domain_image_targetdomain_label_list_[target_img_label][temp_img_idx];
          int target_img_path_idx = caffe_rng_rand()%target_domain_image_global_label_list_[target_img_idx].size();
          batch_lines_[k] = target_domain_image_global_label_list_[target_img_idx][target_img_path_idx];
          batch_lines_[k].second = target_train_val_label_[target_img_idx];
          int source_nn_location_idx = caffe_rng_rand()%refined_target_nearest_neighbor_in_source_data_[target_img_idx].size();
          int source_nn_idx = refined_target_nearest_neighbor_in_source_data_[target_img_idx][source_nn_location_idx];
          batch_lines_[k+target_batch_size] = source_domain_image_global_label_list_[source_nn_idx];
          labels_int_batch_iter->second--;
          label_index++;
        }
      }
    }
  }

  //LOG(INFO)<<"Construct the mini-batch finished!";
  CHECK_EQ(batch_lines_.size(),batch_size);
  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + batch_lines_[0].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << batch_lines_[0].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  
  // datum scales
  const int lines_size = batch_lines_.size();
  CHECK_EQ(lines_size, batch_size);
  //LOG(INFO)<<"Loading iamges...";
  for (int item_id = 0; item_id < lines_size; ++item_id) {
    //LOG(INFO)<<batch_lines_[item_id].second<<"\t"<<batch_lines_[item_id].first;
    // get a blob
    timer.Start();
    cv::Mat cv_img = ReadImageToCVMat(root_folder + batch_lines_[item_id].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << batch_lines_[item_id].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = batch_lines_[item_id].second;
  }
  //
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(RefinedBoostedJointTrainingDataLayer);
REGISTER_LAYER_CLASS(RefinedBoostedJointTrainingData);

}  // namespace caffe
#endif  // USE_OPENCV
