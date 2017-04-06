#include <cfloat>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "caffe/layers/feature_statistics_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FeatureStatisticsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  feature_statistics_path_ = this->layer_param_.feature_stats_param().feature_statistics_path();
  previous_statistics_path_ = this->layer_param_.feature_stats_param().previous_statistics_path();
  new_feature_statistics_path_ = this->layer_param_.feature_stats_param().new_feature_statistics_path();
  bin_ = this->layer_param_.feature_stats_param().bin();
  int num = bottom[0]->num();
  total_channels_ = 0;
  for(int k=0;k<bottom.size();k++){
    CHECK_EQ(num, bottom[k]->num());
    total_channels_ += bottom[k]->channels();
  }
  CHECK_EQ(total_channels_, this->layer_param_.feature_stats_param().total_channels());
  feature_stats_.Reshape({total_channels_, 7});
  for(int k =0;k<total_channels_;k++){
      feature_stats_.mutable_cpu_data()[k*7+5]= INT_MAX;
      feature_stats_.mutable_cpu_data()[k*7+6]= -INT_MAX;
  }
  saving_signal_ = this->layer_param_.feature_stats_param().saving_signal();
  //
  bound_stats_.Reshape({total_channels_, 7});
  std::ifstream input_stats(previous_statistics_path_.c_str());
  int idx = 0;
  Dtype* bound_data = bound_stats_.mutable_cpu_data();
  while(input_stats >> bound_data[idx*7+0] >> bound_data[idx*7+1]>> bound_data[idx*7+2]>> bound_data[idx*7+3]
      >> bound_data[idx*7+4]>> bound_data[idx*7+5]>> bound_data[idx*7+6]){
      idx++;
  }
  CHECK_EQ(idx, total_channels_);
  interval_num_ = 2000;
  refined_feature_stats_.Reshape({total_channels_, 2+interval_num_});
  features_stats_bound_.Reshape({total_channels_, 2+interval_num_});
  caffe_set(refined_feature_stats_.count(), Dtype(0.0), refined_feature_stats_.mutable_cpu_data());
}

template <typename Dtype>
void FeatureStatisticsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape({bottom[0]->num(), 5});
}

template <typename Dtype>
void FeatureStatisticsLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  static int feat_forward_iter = 0;
  int channel_idx=0;
  for(int m = 0; m < bottom.size(); m++){
      for(int n = 0; n < bottom[m]->channels(); n++){
          int dim = bottom[m]->count()/bottom[m]->num()/bottom[m]->channels();
          if(bottom[m]->num_axes()<3){
              CHECK_EQ(dim, 1);
          }else if(bottom[m]->num_axes()==3){
              CHECK_EQ(dim, bottom[m]->height());
          }else if(bottom[m]->num_axes()==4){
              CHECK_EQ(dim, bottom[m]->width()*bottom[m]->height());
          }else{
              LOG(FATAL)<<"Blob shape size is greater than 4!";
          }
          int num = bottom[0]->num();
          const Dtype min_ele = bound_stats_.cpu_data()[channel_idx*7+5];
          const Dtype max_ele = bound_stats_.cpu_data()[channel_idx*7+6];
          const Dtype bin_interval = (max_ele-min_ele)/Dtype(interval_num_);
          const Dtype scalar = 1;
          for(int k = 0; k < num; k++){
              for(int l =0;l<dim;l++){
                  Dtype temp = bottom[m]->cpu_data()[k*bottom[m]->channels()*dim+ n*dim + l];
                  if(temp<feature_stats_.cpu_data()[channel_idx*7+5])feature_stats_.mutable_cpu_data()[channel_idx*7+5]=temp;
                  if(temp>feature_stats_.cpu_data()[channel_idx*7+6])feature_stats_.mutable_cpu_data()[channel_idx*7+6]=temp;
                  int bin_idx = (temp - min_ele)/bin_interval;  
                  if(bin_idx < 0) bin_idx = 0;
                  if(bin_idx > (interval_num_-1)) bin_idx = (interval_num_-1);
                  refined_feature_stats_.mutable_cpu_data()[channel_idx*(2+interval_num_)+(2+bin_idx)] += scalar;
              }
          }
          feature_stats_.mutable_cpu_data()[channel_idx*7+0] = bottom.size();
          feature_stats_.mutable_cpu_data()[channel_idx*7+1] = m;
          feature_stats_.mutable_cpu_data()[channel_idx*7+2] = bottom[m]->channels();
          feature_stats_.mutable_cpu_data()[channel_idx*7+3] = n;
          feature_stats_.mutable_cpu_data()[channel_idx*7+4] = dim;
          channel_idx++;
      }
  }
  feat_forward_iter++;
  caffe_set(top[0]->count(), Dtype(0.0), top[0]->mutable_cpu_data());
  //
  if(feat_forward_iter%saving_signal_!=0)return;
  std::ofstream output_stats(feature_statistics_path_.c_str());
  std::ofstream output_new_stats(new_feature_statistics_path_.c_str());
  for(int k =0;k<total_channels_;k++){
      for(int m=0;m<7;m++){
          output_stats<<feature_stats_.cpu_data()[k*7+m]<<"\t";
      }
      output_stats<<std::endl;
      //
      Dtype asum = caffe_cpu_asum(interval_num_, refined_feature_stats_.cpu_data()+k*(2+interval_num_)+2);
      //LOG(INFO)<<asum;
      caffe_cpu_scale(interval_num_, Dtype(1/Dtype(asum+1)),
          refined_feature_stats_.cpu_data()+k*(2+interval_num_)+2,
          features_stats_bound_.mutable_cpu_data()+k*(2+interval_num_)+2);
      //Get the equalized statistical histogram of the features 
      const Dtype min_ele = bound_stats_.cpu_data()[k*7+5];
      const Dtype max_ele = bound_stats_.cpu_data()[k*7+6];
      const Dtype bin_interval = (max_ele-min_ele)/Dtype(interval_num_);
      const Dtype scalar = 1;
      output_new_stats<<min_ele<<"\t";
      output_new_stats<<max_ele<<"\t";
      ////
      Dtype start = min_ele;
      Dtype end = max_ele;
      Dtype proportion = 0;
      Dtype interval_idx = 0;
      for(int m = 0; m< interval_num_; m++){
          CHECK_LE(interval_idx, bin_);
          if(interval_idx==bin_)continue;
          Dtype prob = features_stats_bound_.cpu_data()[k*(2+interval_num_)+(2+m)];
          proportion += prob;

          int interval_len = int(proportion*bin_);
          if(interval_len>=1){
              CHECK_LE(interval_idx+interval_len, bin_);
              end = bin_interval*(m+1) + min_ele;
              Dtype interval_scalar = (end-start)/Dtype(interval_len);
              for(int n=0;n<interval_len;n++){
                  Dtype temp = start + (n+1)*interval_scalar;
                  output_new_stats<< temp <<"\t"<<proportion/Dtype(interval_len) <<"\t";
              }
              proportion = 0;
              start = end;
              interval_idx += interval_len;
          }else{
          }
          if(m==(interval_num_-1)){
              end = max_ele;
              interval_len = bin_ - interval_idx;
              Dtype interval_scalar = (end-start)/Dtype(interval_len);
              for(int n=0;n<interval_len;n++){
                  Dtype temp = start + (n+1)*interval_scalar;
                  output_new_stats<< temp <<"\t"<<proportion/Dtype(interval_len) <<"\t";
              }
              proportion = 0;
              start = end;
              interval_idx += interval_len;
          }
      }
      output_new_stats<< std::endl; 
  }
  //
}

template <typename Dtype>
void FeatureStatisticsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//TO DO
}

#ifdef CPU_ONLY
STUB_GPU(FeatureStatisticsLayer);
#endif

INSTANTIATE_CLASS(FeatureStatisticsLayer);
REGISTER_LAYER_CLASS(FeatureStatistics);

}  // namespace caffe
