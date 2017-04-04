#include <algorithm>
#include <vector>
#include <fstream>

#include "caffe/layers/normal_knn_match_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormalKnnMatchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const KnnMatchParameter& knn_match_param = this->layer_param_.knn_match_param();
  sample_num_ = knn_match_param.sample_num();
  dim_ = knn_match_param.dim();
  nodes_num_ = knn_match_param.nodes_num();
  orientation_ = knn_match_param.orientation();
  scale_ = knn_match_param.scale();
  bin_ = knn_match_param.bin();
  iterations_ = knn_match_param.iterations();
  feature_path_ = knn_match_param.feature_path();
  image_retrieval_result_ = knn_match_param.image_retrieval_result();
  channel_weight_path_ = knn_match_param.channel_weight_path();
  CHECK_EQ(dim_, bottom[0]->count()/bottom[0]->num());
  int nodes_ = nodes_num_ * 2;
  max_heap_.Reshape({sample_num_, nodes_num_ * 2});
  max_copy_.Reshape({sample_num_, nodes_num_ * 2});
  caffe_set(max_heap_.count(),Dtype(1e30),max_heap_.mutable_cpu_data());
  //LOG(INFO)<<max_heap_.cpu_data()[0]<<" "<<max_heap_.cpu_data()[1];
  //
  pair_distance_.Reshape({sample_num_, bottom[0]->num()});
  //
  features_.Reshape({sample_num_, dim_});
  labels_.Reshape({sample_num_});
  temp_.Reshape({dim_});
  //
  std::ifstream features(feature_path_.c_str());
  int index = 0;
  int feature_index = 0;
  int label_index = 0;
  Dtype value(0.0);
  LOG(INFO)<<"Reading small domain features...";
  while(features>>value){
	  label_index = index/(dim_+1);
    if(label_index>(sample_num_-1)){
      label_index=(sample_num_-1);
      break;
    }
	  feature_index = index%(dim_+1);
	  if(feature_index==0){
      //LOG(INFO)<<label_index;
		  labels_.mutable_cpu_data()[label_index] = value;
	  }else{
		  features_.mutable_cpu_data()[label_index*dim_+(feature_index-1)] = value;
	  }

	  index++;
  }
  CHECK_EQ(sample_num_*(1+dim_),index);
  CHECK_EQ(sample_num_,label_index+1);
  CHECK_EQ(dim_, feature_index);
  LOG(INFO)<<"Reading small domain features finished!";
   //
  channel_weights_.Reshape({dim_});
  std::ifstream channel_weight_file(channel_weight_path_.c_str());
  index = 0;
  while(channel_weight_file>>value){
	  for(int k=0;k<bin_;k++){
		  channel_weights_.mutable_cpu_data()[index*bin_+k]=value;
	  }
	  index++;
  }
  CHECK_EQ(index*bin_, dim_);
}

template <typename Dtype>
void NormalKnnMatchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //
  //vector<int> loss_shape(1);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape({bottom[0]->num(), 10});
  //
}

template <typename Dtype>
Dtype NormalKLDistance(Dtype* anchor, Dtype* positive, int len){
	Dtype distance(0.0);
	for(int k=0;k<len;k++)distance+=0.5*(anchor[k]*log(anchor[k]/(positive[k]+1e-20))+
	    positive[k]*log(positive[k]/(anchor[k]+1e-20)));
	return distance;
}

template <typename Dtype>
Dtype NormalKLDistanceOfHistorgram(Dtype *rotated_feature_cpu, Dtype* anchor, Dtype* positive, int bin, int scale, int orientation){
	Dtype minDistance(INT_MAX);
	//
	for(int k=0;k<orientation;k++){
		for(int m=0;m<scale;m++){
			for(int n=0;n<orientation;n++){
				int index = (k+n)%orientation;
				for(int i=0;i<bin;i++){
					rotated_feature_cpu[m*orientation*bin+n*bin+i]=anchor[m*orientation*bin+index*bin+i];
				}
			}
		}
		minDistance = min(minDistance, KLDistance(rotated_feature_cpu, positive, orientation*bin));
	}
	return minDistance;
}


template <typename Dtype>
void NormalKnnMatchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      LOG(INFO)<<"KNN Match Forward_cpu";
      const Dtype* top_data = top[0]->cpu_data();
      //Dtype scale = Dtype(1/Dtype(bottom[0]->width() * bottom[0]->height()));
      //caffe_cpu_scale(top[0]->count(), scale, top[0]->cpu_data(), top_data);
      caffe_gpu_set(top[0]->count(),Dtype(1.0),top[0]->mutable_cpu_data());
}

template <typename Dtype>
void NormalKnnMatchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(NormalKnnMatchLayer);
#endif

INSTANTIATE_CLASS(NormalKnnMatchLayer);
REGISTER_LAYER_CLASS(NormalKnnMatch);

}  // namespace caffe
