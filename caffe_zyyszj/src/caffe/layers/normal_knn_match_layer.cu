#include <vector>
#include <fstream>
#include <iomanip>
#include "caffe/layers/normal_knn_match_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_functions.h>
#include <device_functions.h>

#include <opencv2/opencv.hpp>
#include <chrono>

#define eps 2.2204e-16

namespace caffe {
//
//top K
template <typename Dtype>
void NormalMaxHeapify(Dtype *A, int i, int len)  
{  
    if(i>=len)return;
    int left=2*i+1;
    int right=2*i+2;
    int largest=i;
    if(left<len&&A[left]>A[largest]){
        largest=left;
    }
    if(right<len&&A[right]>A[largest]){
        largest=right;
    }
    if(largest!=i){
        Dtype temp=A[i];
        A[i]=A[largest];
        A[largest]=temp;
		//
		Dtype temp_index=A[i+len];
		A[i+len]=A[largest+len];
        A[largest+len]=temp_index;
		//
        NormalMaxHeapify(A,largest,len);
    } 
	return;
}

//
template <typename Dtype>
void NormalBuildMaxHeap(Dtype A[], int len)
{
    for(int i=len/2-1; i>=0; --i)  
        MaxHeapify(A, i, len);
}

template <typename Dtype>
void NormaltopK(Dtype A[], int n, int k)
{
    BuildMaxHeap(A, k);  
    for(int i=k; i<n; ++i)
    {
        if(A[i] < A[0])  
        {
            Dtype tmp = A[0];
            A[0] = A[i];
            A[i] = tmp;
            MaxHeapify(A, 0, k);  
        }
    }
}

//
template <typename Dtype>
__global__ void NormalKnnMatchPairwiseDistance(
    const int nthreads,
    const int dim, 
    const int anchor_num, 
    const int positive_num, 
	const int orientation,
	const int scale,
	const int bin,
    const Dtype *anchor, 
    const Dtype *positive,
    const Dtype *weight,	
    Dtype *pdist) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
	int i = index / positive_num;
	int j = index % positive_num;

	if (i >= anchor_num || j >= positive_num) 
		return;
	int anchor_ptr_index = i*dim;
	int posi_ptr_index = j*dim;

	Dtype min_dist = Dtype(1e30);
    bool used = false;
	/*for(int k=0;k<orientation;k++){
	    Dtype sum(0.0);
		for(int m=0;m<(2*scale);m++){
			for(int n=0;n<orientation;n++){
				int index = (k+n)%orientation;
				for(int s=0;s<bin;s++){
				    Dtype temp= anchor[anchor_ptr_index+m*orientation*bin+index*bin+s];
					Dtype posi= positive[posi_ptr_index+m*orientation*bin+n*bin+s];
					sum += Dtype(0.5)*(temp*log((temp+Dtype(eps))/(posi+Dtype(eps)))+
			            posi*log((posi+Dtype(eps))/(temp+Dtype(eps)))); 
                    //sum+=abs(temp-posi);
				}
			}
		}
        if(used){
             min_dist = min(min_dist, sum);
        }else{
             min_dist = sum;
             used=true;
        }
	}*/
	min_dist = Dtype(0.0);
	for(int k=0;k<dim;k++){
	    Dtype temp= anchor[anchor_ptr_index+k];
		Dtype posi= positive[posi_ptr_index+k];
		Dtype channel_weight = weight[k];
		min_dist += channel_weight*Dtype(0.5)*(temp*log((temp+Dtype(eps))/(posi+Dtype(eps)))+
	      posi*log((posi+Dtype(eps))/(temp+Dtype(eps)))); 
	}
	pdist[i * positive_num + j] = min_dist;
  }
}

//
template <typename Dtype>
__global__ void NormalKnnMatchForward(
    const int anchor_num,
	const int j,
    const int positive_num,
	const int node_num,
    const Dtype *pair_distance,
    const Dtype *labels,
	const Dtype *max_copy,
    Dtype *max_heap) {
	//
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i = thread_idx;
	if (i >= anchor_num) 
		return;
    for(int j=0;j<positive_num;j++){
        Dtype distance = pair_distance[i*positive_num+j];
        Dtype label = labels[j];
	    //
        int index = 0;
        Dtype value = max_heap[i*(2*node_num)+ 0 * 2 + 0];
	    //
	    int start = i * node_num;
	    int end = (i+1) * node_num;
	    for(int k = start; k < end; k++){
	       if(Dtype(500)>value){
		         value=max_copy[i*(2*node_num)+ k * 2 + 0];
			     index=k;
		    }
	    }
        if(max_copy[i*(2*node_num)+ index * 2 + 0]<distance){
            return;
        }
        max_heap[i*(2*node_num)+ index * 2 + 0]=distance;
        max_heap[i*(2*node_num)+ index * 2 + 0]=label;
        
    }
}
//

template <typename Dtype>
void NormalKnnMatchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  static int forward_iter=0;
  /* if(forward_iter==0){
 std::ofstream sf1("sf1.txt");
for(int k=0;k<10;k++){
      for(int m=0;m<dim_;m++){
          sf1<<	bottom[0]->cpu_data()[k*dim_+m]<<"\t";
      }
      sf1<<std::endl;
      for(int m=0;m<dim_;m++){
          sf1<<	features_.cpu_data()[k*dim_+m]<<"\t";
      }
      sf1<<std::endl;
  }
  }
  */
  //
  //CHECK_EQ(dim_,2*orientation_*scale_*bin_);
  NormalKnnMatchPairwiseDistance<<<CAFFE_GET_BLOCKS(sample_num_ * bottom[0]->num()), CAFFE_CUDA_NUM_THREADS>>>(
    sample_num_ * bottom[0]->num(),
	dim_,  
	sample_num_,
	bottom[0]->num(),	
	orientation_, 
	scale_, 
	bin_, 
	features_.gpu_data(), 
	bottom[0]->gpu_data(),
	channel_weights_.gpu_data(),
	pair_distance_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
/*
  if(forward_iter==0){
 std::ofstream sf("sf.txt");
  for(int k=0;k<sample_num_;k++){
      for(int m=0;m<bottom[0]->num();m++){
          sf<<pair_distance_.cpu_data()[k*bottom[0]->num()+m]<<"\t";
		  std::cout<<pair_distance_.cpu_data()[k*bottom[0]->num()+m]<<"\t";
      }
      sf<<std::endl;
	  std::cout<<std::endl;
  }
  }
 */
  //
  /*
  NormalKnnMatchForward<<<CAFFE_GET_BLOCKS(sample_num_), CAFFE_CUDA_NUM_THREADS>>>(
	sample_num_, 
	1,
  bottom[0]->num(),
	nodes_num_, 
	pair_distance_.gpu_data(),
	bottom[1]->gpu_data(),
  max_copy_.gpu_data(),
  max_heap_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  */
  //
  //stringstream dk;
  //dk<<forward_iter<<".txt";
  //std::ofstream rank(dk.str());
  Dtype *max_heap = max_heap_.mutable_cpu_data();
  for(int i=0;i<sample_num_;i++){
    for(int j=0;j<bottom[0]->num();j++){
	    if(max_heap[i*(2*nodes_num_)]>pair_distance_.cpu_data()[i*bottom[0]->num()+j]){
		   max_heap[i*(2*nodes_num_)]=pair_distance_.cpu_data()[i*bottom[0]->num()+j];
		   max_heap[i*(2*nodes_num_)+nodes_num_]=bottom[1]->cpu_data()[j];
		   NormalMaxHeapify(max_heap + i*(2*nodes_num_), 0, nodes_num_);
		}
    }
	//
    /*for(int k=0;k<nodes_num_;k++){
	    rank<<max_heap[i*(2*nodes_num_)+k]<<"\t"<<max_heap[i*(2*nodes_num_)+k+nodes_num_]<<"\t";
	}
	rank<<std::endl;
	*/
  }
  //
  //LOG(INFO)<<forward_iter<<" "<<iterations_;
  if((forward_iter!=0)&&(forward_iter%iterations_==0)){
      std::ofstream retrieval_result(image_retrieval_result_.c_str());
	  for(int m=0;m<sample_num_;m++){
	      retrieval_result<<labels_.cpu_data()[m]<<"\t";
	      for(int n=0;n<2*nodes_num_;n++){
		      retrieval_result<<std::setprecision(17)<<max_heap_.cpu_data()[m*nodes_num_*2 + n]<<"\t";
		    }
		    retrieval_result<<std::endl;
	  }
  }
  caffe_gpu_set(top[0]->count(),Dtype(1.0),top[0]->mutable_gpu_data());
  forward_iter++;
}

template <typename Dtype>
void NormalKnnMatchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   caffe_gpu_set(bottom[0]->count(),Dtype(1.0),bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalKnnMatchLayer);

}  // namespace caffe
