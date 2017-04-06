#include <algorithm>
#include <vector>
#include <iomanip>

#include "caffe/layers/refined_hist_feature_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_functions.h>
#include <device_functions.h>

#include <opencv2/opencv.hpp>
#include <chrono>

namespace caffe {

// one thread for every positive pair
/*template <typename Dtype>
__global__ void RefinedHistFeatureForward(
    const int nthreads,
    const int num,
    const int channel,
    const int dim,
    const int bin,
	const int total_channel,
    const int channel_start,
    const Dtype	*min_max_ele,
	const Dtype *bottom, 
    Dtype *top)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
	int sample_idx = index / (channel*dim);
	int feature_idx = index % (channel*dim);
    int channel_idx = feature_idx / dim;
    int dim_idx = feature_idx % dim;

	if ((sample_idx >= num) || (feature_idx >= (channel*dim)) ||(channel_idx>= channel)||(dim_idx>=dim)) 
		return;
    int current_channel = channel_start+channel_idx;
    Dtype mean = min_max_ele[5*current_channel+3];
    Dtype vari = min_max_ele[5*current_channel+4];
	Dtype minEle = mean-(3.5*vari);
    Dtype maxEle = mean+(3.5*vari);
    Dtype histInterval = (maxEle-minEle)/bin;
    int hit;
    hit = int((bottom[sample_idx*channel*dim+channel_idx*dim+dim_idx] - minEle)/histInterval);
    if(histInterval<=0){
      printf("Error when computing historgram histInterval<=0!\n");
    }
    if(hit>=bin){
      //printf("Error when computing historgram hit>=bin!\n");
	  hit=bin-1;
    }
    if(hit<0){
      //printf("Error when computing historgram hit<0!\n");
	  hit=0;
    }
    caffe_gpu_atomic_add(Dtype(1.0/Dtype(dim)), top + sample_idx*total_channel*bin+current_channel*bin+hit);	
  }
}

//
template <typename Dtype>
void RefinedHistFeatureLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
  caffe_gpu_set(top[0]->count(), Dtype(0.0), top[0]->mutable_gpu_data());
  int channel_start=0;
  int count_idx=0;
  for(int k=0;k<(bottom.size()-1);k++){
    //
    CHECK_LE(channel_start, total_channels_);
	int dim = bottom[k]->count()/bottom[k]->num();
	dim = dim/bottom[k]->channels();
	CHECK_EQ(dim, bottom[k]->height()*bottom[k]->width());
	HistFeatureForward<<<CAFFE_GET_BLOCKS(bottom[k]->count()), CAFFE_CUDA_NUM_THREADS>>>(
	  bottom[k]->count(),
	  bottom[k]->num(),
      bottom[k]->channels(),
      dim,
      bin_, 
	  total_channels_,
	  channel_start,
	  feature_stats_.gpu_data(),
	  bottom[k]->gpu_data(), 
      top[0]->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
	//
	channel_start += bottom[k]->channels();
  }
  static int hist_forward_iter = 0;
  //LOG(INFO)<<top[0]->asum_data()<<"\t"<<hist_forward_iter;
  //
  std::ofstream feature_file(feature_path_.c_str(), ios::app);
  int feature_dim = total_channels_ * bin_;
  int label_dim=bottom[bottom.size()-1]->count()/bottom[bottom.size()-1]->num();
  CHECK_EQ(label_dim,1);
  for(int m = 0; m < top[0]->num(); m++){
    feature_file<<bottom[bottom.size()-1]->cpu_data()[m]<<"\t";
	for(int n = 0; n < feature_dim; n++){
	  feature_file<< std::setprecision(25)<<top[0]->cpu_data()[m*feature_dim+n]<<"\t";
	}
	feature_file<<std::endl;
  }
  
}
*/

template <typename Dtype>
__global__ void RefinedHistFeatureForward(
    const int nthreads,
    const int num,
    const int channel,
    const int dim,
    const int bin,
	const int total_channel,
    const int channel_start,
    const Dtype	*min_max_ele,
	const Dtype *bottom, 
    Dtype *top)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
	int sample_idx = index / (channel*dim);
	int feature_idx = index % (channel*dim);
    int channel_idx = feature_idx / dim;
    int dim_idx = feature_idx % dim;

	if ((sample_idx >= num) || (feature_idx >= (channel*dim)) ||(channel_idx>= channel)||(dim_idx>=dim)) 
		return;
	Dtype data = bottom[sample_idx*channel*dim+channel_idx*dim+dim_idx];
    int current_channel = channel_start+channel_idx;
	Dtype minEle = min_max_ele[current_channel*(bin+2)+0];
    Dtype maxEle = min_max_ele[current_channel*(bin+2)+1];
	int hit=-1;
	if(data<=minEle){
	    hit = 0;
	}else if(data>=maxEle){
	    hit = bin-1;
	}else{
	    for(int k=0;k<bin;k++){
		    if(data<min_max_ele[current_channel*(bin+2)+(2+k)]){
			   hit = k;
			   break;
			}
		}
		if(hit == -1)printf("Error during caculating the histogram!\n");
	}
    caffe_gpu_atomic_add(Dtype(1.0/Dtype(dim)), top + sample_idx*total_channel*bin+current_channel*bin+hit);	
  }
}

//
template <typename Dtype>
void RefinedHistFeatureLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
  caffe_gpu_set(top[0]->count(), Dtype(0.0), top[0]->mutable_gpu_data());
  int channel_start=0;
  int count_idx=0;
  for(int k=0;k<(bottom.size()-1);k++){
    //
    CHECK_LE(channel_start, total_channels_);
	int dim = bottom[k]->count()/bottom[k]->num();
	dim = dim/bottom[k]->channels();
	CHECK_EQ(dim, bottom[k]->height()*bottom[k]->width());
	RefinedHistFeatureForward<<<CAFFE_GET_BLOCKS(bottom[k]->count()), CAFFE_CUDA_NUM_THREADS>>>(
	  bottom[k]->count(),
	  bottom[k]->num(),
      bottom[k]->channels(),
      dim,
      bin_, 
	  total_channels_,
	  channel_start,
	  feature_stats_.gpu_data(),
	  bottom[k]->gpu_data(), 
      top[0]->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
	//
	channel_start += bottom[k]->channels();
  }
  static int hist_forward_iter = 0;
  //LOG(INFO)<<top[0]->asum_data()<<"\t"<<hist_forward_iter;
  //
  std::ofstream feature_file(feature_path_.c_str(), ios::app);
  int feature_dim = total_channels_ * bin_;
  int label_dim=bottom[bottom.size()-1]->count()/bottom[bottom.size()-1]->num();
  CHECK_EQ(label_dim,1);
  for(int m = 0; m < top[0]->num(); m++){
    feature_file<<bottom[bottom.size()-1]->cpu_data()[m]<<"\t";
	for(int n = 0; n < feature_dim; n++){
	  feature_file<< std::setprecision(25)<<top[0]->cpu_data()[m*feature_dim+n]<<"\t";
	}
	feature_file<<std::endl;
  }
  
}

//
template <typename Dtype>
void RefinedHistFeatureLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
	//TO DO
}


INSTANTIATE_LAYER_GPU_FUNCS(RefinedHistFeatureLayer);

}  // namespace caffe