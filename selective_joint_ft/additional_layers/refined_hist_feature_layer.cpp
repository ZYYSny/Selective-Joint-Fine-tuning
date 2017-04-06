#include <cfloat>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "caffe/layers/refined_hist_feature_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

vector<string> histsplit(const string &s, const string &seperator) {
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
void RefinedHistFeatureLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  HistFeatureParameter hist_feature_param = this->layer_param_.hist_feature_param();
  bin_ = hist_feature_param.bin();
  total_channels_ = hist_feature_param.total_channels();
  feature_path_ = hist_feature_param.feature_path();
  channel_weight_path_ = hist_feature_param.channel_weight_path();
  feature_stats_.Reshape({total_channels_,bin_+2});
  std::ifstream input_stats(hist_feature_param.feature_statistics_path());
  //
  Dtype* feature_stats_data = feature_stats_.mutable_cpu_data();
  int idx = 0;
  string line;
  while (!input_stats.eof()) {
    //LOG(INFO)<<idx;
	std::getline(input_stats, line);
		if (line == "")continue;
    vector<string> stats = histsplit(line, " \t");
    //std::cout<<idx<<std::endl;
    CHECK_EQ(2*bin_+2, stats.size());
    feature_stats_data[idx*(bin_+2)+0]=Dtype(stod(stats[0]));
    feature_stats_data[idx*(bin_+2)+1]=Dtype(stod(stats[1]));
    for(int k=0;k<(stats.size()/2-1);k++){
      feature_stats_data[idx*(bin_+2)+k+2]=Dtype(stod(stats[2*(k+1)]));
    }
    for(int m=0;m<(bin_+2);m++){
      std::cout<<feature_stats_data[idx*(bin_+2)+m]<<"\t";
    }
    std::cout<<"******"<<std::endl;
    idx++;
  }
  //
  CHECK_EQ(idx, total_channels_);
  int total_chs_ = 0;
  for(int k=0;k<(bottom.size()-1);k++){
    CHECK_EQ(bottom[0]->num(), bottom[k]->num());
    total_chs_ += bottom[k]->channels();
  }
  CHECK_EQ(total_channels_, total_chs_);
  
  for(int k=0;k<feature_stats_.num();k++){
      int bottom_idx = feature_stats_.cpu_data()[k*5+0];
      int channel_size = feature_stats_.cpu_data()[k*5+1];
      //CHECK_EQ(channel_size, bottom[bottom_idx]->channels());
      //int dim = feature_stats_.cpu_data()[k*5+4];
      //CHECK_EQ(dim, bottom[bottom_idx]->count()/bottom[bottom_idx]->num()/bottom[bottom_idx]->channels());
  }
  //
  std::ofstream channel_weight_file(channel_weight_path_.c_str());
  CHECK_EQ(bottom.size()-1, hist_feature_param.channel_weight_size());
  for(int k=0;k<bottom.size()-1;k++){
	  Dtype channel_weight = hist_feature_param.channel_weight(k);
	  channel_weight /= bottom[k]->channels(); 
	  for(int m=0; m<bottom[k]->channels(); m++){
		  channel_weight_file<< channel_weight << std::endl;
	  }
  }
}

template <typename Dtype>
void RefinedHistFeatureLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape({bottom[0]->num(), total_channels_, bin_});
}

template <typename Dtype>
void RefinedHistFeatureLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RefinedHistFeatureLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//TO DO
}

#ifdef CPU_ONLY
STUB_GPU(RefinedHistFeatureLayer);
#endif

INSTANTIATE_CLASS(RefinedHistFeatureLayer);
REGISTER_LAYER_CLASS(RefinedHistFeature);

}  // namespace caffe
