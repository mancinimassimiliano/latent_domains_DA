#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  // Reshape entropy blob
  vector<int> entropy_shape = prob_.shape();
  entropy_shape[softmax_axis_] = 1;
  entropy_.Reshape(entropy_shape);
  // Put ones into entropy_diff
  Dtype* entropy_diff = entropy_.mutable_cpu_diff();
  caffe_set(prob_.shape(softmax_axis_), Dtype(1.), entropy_diff);
}

template <typename Dtype>
Dtype EntropyLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_VALID:
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* ones = entropy_.cpu_diff();
  Dtype* tmp_data = prob_.mutable_cpu_diff();
  Dtype* entropy_data = entropy_.mutable_cpu_data();
  int N = prob_.count();
  int dim = prob_.count() / outer_num_;
  int cls = bottom[0]->shape(softmax_axis_);
  // Init buffers
  caffe_set(N / cls, Dtype(0), entropy_data);
  caffe_set(N, Dtype(FLT_MIN), tmp_data);
  // tmp_data = p * log(max(p, FLT_MIN))
  //caffe_cpu_eltwise_max(N, Dtype(1.), prob_data, Dtype(1.), tmp_data);
  for(int i=0; i<N; ++i) tmp_data[i]=std::max(tmp_data[i], prob_data[i]);
  caffe_log(N, tmp_data, tmp_data);
  caffe_mul(N, prob_data, tmp_data, tmp_data);
  // Compute element-wise entropy
  for (int i = 0; i < outer_num_; ++i) {
    const Dtype* plogp = tmp_data + i*dim;
    Dtype* H = entropy_data + i * inner_num_;
    caffe_cpu_gemv(CblasTrans, cls, inner_num_, Dtype(-1.), plogp, ones,
      Dtype(0.), H);
  }
  Dtype loss = 0;
  for (int i = 0; i < N / cls; ++i) {
    loss += entropy_data[i];
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_);
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* entropy_data = entropy_.cpu_data();
    const Dtype* plogp_data = prob_.cpu_diff();
    int dim = prob_.count() / outer_num_;
    int cls = bottom[0]->shape(softmax_axis_);
    for (int i = 0; i < outer_num_; ++i) {
      const Dtype* H = entropy_data + i * inner_num_;
      for (int c = 0; c < cls; ++c) {
        caffe_mul(inner_num_, H, prob_data + i * dim + c * inner_num_,
            bottom_diff + i * dim + c * inner_num_);
      }
      caffe_cpu_axpby(dim, Dtype(1.), plogp_data + i * dim, Dtype(1.),
          bottom_diff + i * dim);
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_);
    caffe_scal(prob_.count(), Dtype(-1.) * loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(EntropyLossLayer);
#endif

INSTANTIATE_CLASS(EntropyLossLayer);
REGISTER_LAYER_CLASS(EntropyLoss);

}  // namespace caffe
