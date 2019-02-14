#ifndef CAFFE_INPUT_LAYER_HPP_
#define CAFFE_INPUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net by assigning tops directly.
 *
 * This data layer is a container that merely holds the data assigned to it;
 * forward, backward, and reshape are all no-ops.
 */
template <typename Dtype>
class InputLayer : public Layer<Dtype> {	//fang: 这一层是在部署的时候会使用到的layer,特点是使用Layer基类的 forward 方法,没有自己的 Forward_xpu 和 Backward_xpu
 public:								//但是net_input_blobs_已经和数据源头的 top blob 绑定,且 net_input_blobs_ 可以通过 input_blobs()获得其引用,想怎么填充就怎么填充
  explicit InputLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "Input"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}	//fang:基类这个函数是纯虚函数,所以必须要实现,但我们不想实现他,所以就在这里搞个空的放着好了
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
};

}  // namespace caffe

#endif  // CAFFE_INPUT_LAYER_HPP_
