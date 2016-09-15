/* Copyright 2016 Waizung Taam

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

- 2016-09-15

*/

#ifndef RNN_H_
#define RNN_H_

#include "tensor/vector.h"
#include "tensor/matrix.h"
#include "tensor/util.h"

#include <iostream>
#include <type_traits>

class RNN {
public:
  using vector_t = tensor::Vector<double>;
  using matrix_t = tensor::Matrix<double>;
  using size_t = std::size_t;
  using index_t = std::make_signed<size_t>::type;
  using random_t = tensor::Random;

  RNN(const size_t& vocab_dim, const size_t& hidden_dim) :
    num_steps_(0), vocab_dim_(vocab_dim), hidden_dim_(hidden_dim),
    W_xh_(matrix_t(hidden_dim, vocab_dim, random_t::uniform_real, -1.0, 1.0)),
    W_hh_(matrix_t(hidden_dim, hidden_dim, random_t::uniform_real, -1.0, 1.0)),
    W_hy_(matrix_t(vocab_dim, hidden_dim, random_t::uniform_real, -1.0, 1.0)),
    d_W_xh_(matrix_t(hidden_dim, vocab_dim)),
    d_W_hh_(matrix_t(hidden_dim, hidden_dim)),
    d_W_hy_(matrix_t(vocab_dim, hidden_dim)),
    H_(matrix_t()) {}

  void train(const matrix_t& X, const matrix_t& T, 
             const size_t& num_epochs, double learning_rate) {
    num_steps_ = X.shape()[0];
    H_ = matrix_t(num_steps_ + 1, hidden_dim_);
    for (size_t i = 0; i < num_epochs; ++i) {
      std::cout << i << "\t";
      matrix_t Y = forward(X);
      backward(X, Y, T);
      update(learning_rate);
    }
  }
  matrix_t predict(const matrix_t& X) {
    return forward(X);
  }

private:
  matrix_t forward(const matrix_t& X) {
    matrix_t Y(num_steps_, vocab_dim_);
    for (index_t t = 0; t < num_steps_; ++t) {
      H_[t] = activ_func(W_xh_ * X[t] + W_hh_ * H_[t - 1]);
      Y[t] = output_func(W_hy_ * H_[t]);
    }
    return Y;
  }
  void backward(const matrix_t& X, const matrix_t& Y, const matrix_t& T) {
    matrix_t d_Y = Y - T;
    std::cout << tensor::util::square(d_Y).sum() << std::endl;
    matrix_t d_H_next(hidden_dim_, 1);
    for (index_t t = num_steps_ - 1; t >= 0; --t) {
      d_W_hy_ += d_Y[t] * matrix_t(H_[t]).T();
      matrix_t d_H = d_activ_func(H_[t]).times(W_hy_.T() * d_Y[t] + d_H_next);
      d_W_xh_ += d_H * matrix_t(X[t]).T();
      d_W_hh_ += d_H * matrix_t(H_[t - 1]).T();
      d_H_next = W_hh_.T() * d_H;
    }
  }
  void update(double learning_rate) {
    W_xh_ -= learning_rate * d_W_xh_;
    W_hh_ -= learning_rate * d_W_hh_;
    W_hy_ -= learning_rate * d_W_hy_;
  }

  vector_t activ_func(const matrix_t& m) {
    vector_t v = m.reshape(1, m.shape()[0])[0];
    return tensor::util::tanh(v);
  }
  matrix_t d_activ_func(const vector_t& v) {
    matrix_t m(v);
    return 1.0 - tensor::util::square(m);
  }
  vector_t output_func(const matrix_t& m) {
    vector_t v = m.reshape(1, m.shape()[0])[0];
    return tensor::util::softmax(v);
  }

  size_t num_steps_;
  size_t vocab_dim_;
  size_t hidden_dim_;

  matrix_t W_xh_;
  matrix_t W_hh_;
  matrix_t W_hy_;

  matrix_t d_W_xh_;
  matrix_t d_W_hh_;
  matrix_t d_W_hy_;

  matrix_t H_;
};

#endif  // RNN_H_