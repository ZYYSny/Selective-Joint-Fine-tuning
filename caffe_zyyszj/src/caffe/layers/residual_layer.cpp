#include <cfloat>
#include <vector>

#include "caffe/layers/residual_layer.hpp"
#include "caffe/util/math_functions.hpp"

static bool share_data = true;
static bool share_diff = true;

namespace caffe {

template <typename Dtype>
void ResidualLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	CHECK(!(bottom.size() == 1 && top.size() == 1)) << "What the fuck are you doing?";

	// CHECK(bottom[0] == top[0]) << "Only in-place operation is supported.";
	// if (bottom.size() == 2 && top.size() == 2) 
	// 	CHECK(bottom[1] == top[1]) << "Only in-place operation is supported.";
}


template <typename Dtype>
void ResidualLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
	if (bottom.size() == 2)
		CHECK(bottom[0]->shape() == bottom[1]->shape()) << "Shape mismatch.";

	top[0]->ReshapeLike(*bottom[0]);
	if (top.size() == 2)
		top[1]->ReshapeLike(*bottom[0]);
}


template <typename Dtype>
void ResidualLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LOG(FATAL) << "Not implemented. This layer is GPU only.";
}

template <typename Dtype>
void ResidualLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, 
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	LOG(FATAL) << "Not implemented. This layer is GPU only.";
}


template <typename Dtype>
void ResidualLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	const int count = bottom[0]->count();
	// bool survive = true;
	const Dtype weight(1);
	
	if (share_data) {
		if (bottom.size() == 2) {
			top[0]->ShareData(*bottom[0]);
			caffe_gpu_axpby(count, Dtype(1), bottom[1]->gpu_data(), weight, top[0]->mutable_gpu_data());
			if (top.size() == 2) {
				top[1]->ShareData(*bottom[1]);
				caffe_copy(count, top[0]->gpu_data(), top[1]->mutable_gpu_data());
			}
		}
		else {
			top[0]->ShareData(*bottom[0]);
			top[1]->ShareData(*bottom[0]);
		}
	}
	else {
		if (bottom.size() == 2) { 
			// residual
			caffe_copy(count, bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
			caffe_gpu_axpby(count, Dtype(1), bottom[1]->gpu_data(), weight, top[0]->mutable_gpu_data());
			if (top.size() == 2)
				caffe_copy(count, top[0]->gpu_data(), top[1]->mutable_gpu_data());
		}
		else if (bottom.size() == 1 && top.size() == 2) {
			// split
			caffe_copy(count, bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
			caffe_copy(count, bottom[0]->gpu_data(), top[1]->mutable_gpu_data());
		}
		else LOG(FATAL) << "Unknown configuration.";
	}
}

template <typename Dtype>
void ResidualLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
	const int count = bottom[0]->count();
	// bool survive = true;

	if (bottom.size() == 2)
		CHECK(propagate_down[0] == propagate_down[1]);

	if (propagate_down[0]) {
		// residual 
		if (share_diff) {
			if (top.size() == 2) {
				bottom[0]->ShareDiff(*top[0]);
				caffe_gpu_axpy(count, Dtype(1), top[1]->gpu_diff(), bottom[0]->mutable_gpu_diff());
				if (bottom.size() == 2) {
					bottom[1]->ShareDiff(*top[1]);
					caffe_copy(count, bottom[0]->gpu_diff(), bottom[1]->mutable_gpu_diff());
				}
			}
			else {
				bottom[0]->ShareDiff(*top[0]);
				bottom[1]->ShareDiff(*top[0]);
			}
		}
		else {
			if (top.size() == 2) {
				caffe_copy(count, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
				caffe_gpu_axpy(count, Dtype(1), top[1]->gpu_diff(), bottom[0]->mutable_gpu_diff());
				if (bottom.size() == 2)
					caffe_copy(count, bottom[0]->gpu_diff(), bottom[1]->mutable_gpu_diff());
			}
			else if (top.size() == 1 && bottom.size() == 2) {
				caffe_copy(count, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
				caffe_copy(count, top[0]->gpu_diff(), bottom[1]->mutable_gpu_diff());
				// caffe_gpu_axpy(count, Dtype(1), top[0]->gpu_diff(), bottom[1]->mutable_gpu_diff());
			}
			else LOG(FATAL) << "Unknown configuration.";
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ResidualLayer);

INSTANTIATE_CLASS(ResidualLayer);
REGISTER_LAYER_CLASS(Residual);

}

