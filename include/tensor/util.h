/*
Copyright 2016 Waizung Taam

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

- 2016-08-19

- ======== tensor::util ========

*/

#ifndef TENSOR_UTIL_H_
#define TENSOR_UTIL_H_

#include "vector.h"
#include "matrix.h"

#include <cmath>
#include <random>
#include <string>
#include <type_traits>


namespace tensor {
namespace util {

template <typename Tp>
Vector<Tp> for_each(const Vector<Tp>& vec, Tp (*func_ptr)(Tp)) {
  Vector<Tp> vec_res(vec.shape()[0]);
  for (typename Vector<Tp>::size_type idx = 0;
       idx < vec_res.shape()[0]; ++idx) {
    vec_res[idx] = (*func_ptr)(vec[idx]);
  }
  return vec_res;
}
template <typename Tp>
Matrix<Tp> for_each(const Matrix<Tp>& mat, Tp (*func_ptr)(Tp)) {
  Matrix<Tp> mat_res(mat.shape()[0], mat.shape()[1]);
  for (typename Matrix<Tp>::size_type idx_row = 0;
       idx_row < mat_res.shape()[0]; ++idx_row) {
    mat_res[idx_row] = for_each(mat[idx_row], *func_ptr);
  }
  return mat_res;
}

#define FOR_EACH_IN_VEC_MAT(NAME, FUNC) \
template <typename Tp> \
Vector<Tp> NAME(const Vector<Tp>& vec) { return for_each<Tp>(vec, FUNC); } \
template <typename Tp> \
Matrix<Tp> NAME(const Matrix<Tp>& mat) { return for_each<Tp>(mat, FUNC); }

// ======== Math Functions ========

FOR_EACH_IN_VEC_MAT(abs, std::abs)
FOR_EACH_IN_VEC_MAT(ceil, std::ceil)
FOR_EACH_IN_VEC_MAT(floor, std::floor)
FOR_EACH_IN_VEC_MAT(round, std::round)

FOR_EACH_IN_VEC_MAT(exp, std::exp)
FOR_EACH_IN_VEC_MAT(exp2, std::exp2)
FOR_EACH_IN_VEC_MAT(expm1, std::expm1)

FOR_EACH_IN_VEC_MAT(log, std::log)
FOR_EACH_IN_VEC_MAT(log2, std::log2)
FOR_EACH_IN_VEC_MAT(log1p, std::log1p)

template <typename Tp, typename ExpT>
Vector<Tp> pow(const Vector<Tp>& vec, const ExpT& exp) {
  Vector<Tp> vec_pow(vec.shape()[0]);
  for (typename Vector<Tp>::index_type idx = 0; 
       idx < vec_pow.shape()[0]; ++idx) {
    vec_pow[idx] = std::pow(vec[idx], exp);
  }
  return vec_pow;
}
template <typename Tp, typename ExpT>
Matrix<Tp> pow(const Matrix<Tp>& mat, const ExpT& exp) {
  Matrix<Tp> mat_pow(mat.shape()[0], mat.shape()[1]);
  for (typename Matrix<Tp>::index_type idx_row = 0; 
       idx_row < mat_pow.shape()[0]; ++idx_row) {
    mat_pow[idx_row] = tensor::util::pow(mat[idx_row], exp);
  }
  return mat_pow;
}
template <typename Tp>
Vector<Tp> square(const Vector<Tp>& vec) { return tensor::util::pow(vec, 2); }
template <typename Tp>
Matrix<Tp> square(const Matrix<Tp>& mat) { return tensor::util::pow(mat, 2); }

FOR_EACH_IN_VEC_MAT(sqrt, std::sqrt)

template <typename Tp>
Vector<Tp> hypot(const Vector<Tp>& vec_x, const Vector<Tp>& vec_y) {
  if (vec_x.shape()[0] != vec_y.shape()[0]) {
    std::string err_msg = "Inconsistent shape for hypot: " + 
      std::to_string(vec_x.shape()[0]) + " != " +
      std::to_string(vec_y.shape()[0]) + ".";
    throw VectorException(err_msg);
  }
  Vector<Tp> vec_res(vec_x.shape()[0]);
  for (typename Vector<Tp>::size_type idx = 0; 
       idx < vec_res.shape()[0]; ++idx) {
    vec_res[idx] = std::hypot(vec_x[idx], vec_y[idx]);
  }
  return vec_res;
}
template <typename Tp>
Matrix<Tp> hypot(const Matrix<Tp>& mat_x,
                 const Matrix<Tp>& mat_y) {
  if (mat_x.shape() != mat_y.shape()) {
    std::string err_msg = "Inconsistent shape for hypot: [" + 
      std::to_string(mat_x.shape()[0]) + ", " +
      std::to_string(mat_x.shape()[1]) + "] != [" +
      std::to_string(mat_y.shape()[0]) + ", " +
      std::to_string(mat_y.shape()[1]) + "].";
    throw MatrixException(err_msg);
  }
  Matrix<Tp> mat_res(mat_x.shape());
  for (typename Matrix<Tp>::index_type idx_row = 0; 
       idx_row < mat_x.shape()[0]; ++idx_row) {
    mat_res[idx_row] = tensor::util::hypot(mat_x[idx_row], mat_y[idx_row]);
  }
  return mat_res;
}


// ======== Activation Functions ========

// ==== ReLu ====
template <typename Tp>
typename std::enable_if<std::is_floating_point<Tp>::value, Tp>::type
relu(Tp x) { return x >= 0.0 ? x : 0.0; }
FOR_EACH_IN_VEC_MAT(relu, tensor::util::relu)

template <typename Tp>
typename std::enable_if<std::is_floating_point<Tp>::value, Tp>::type
d_relu(Tp x) { return x >= 0.0 ? 1.0 : 0.0; }
FOR_EACH_IN_VEC_MAT(d_relu, tensor::util::d_relu)

// ==== Leaky ReLu ====
template <typename Tp>
typename std::enable_if<std::is_floating_point<Tp>::value, Tp>::type
leaky_relu(Tp x) { return x >= 0.0 ? x : 0.01 * x; }
FOR_EACH_IN_VEC_MAT(leaky_relu, tensor::util::leaky_relu)

template <typename Tp>
typename std::enable_if<std::is_floating_point<Tp>::value, Tp>::type
d_leaky_relu(Tp x) { return x >= 0.0 ? 1.0 : 0.01; }
FOR_EACH_IN_VEC_MAT(d_leaky_relu, tensor::util::d_leaky_relu)

// ==== ELu ====
template <typename Tp>
typename std::enable_if<std::is_floating_point<Tp>::value, Tp>::type
elu(Tp x) { return x >= 0.0 ? x : std::expm1(x); }
FOR_EACH_IN_VEC_MAT(elu, tensor::util::elu)

template <typename Tp>
typename std::enable_if<std::is_floating_point<Tp>::value, Tp>::type
d_elu(Tp x) { 
  // return x >= 0.0 ? 1.0 : std::exp(x); 
  return x >= 0.0 ? 1.0 : x + 1.0;
}
FOR_EACH_IN_VEC_MAT(d_elu, tensor::util::d_elu)

// ==== Sigmoid ====
template <typename Tp>
typename std::enable_if<std::is_floating_point<Tp>::value, Tp>::type
sigmoid(Tp x) { return 1.0 / (1.0 + std::exp(-x)); }
FOR_EACH_IN_VEC_MAT(sigmoid, tensor::util::sigmoid)

template <typename Tp>
typename std::enable_if<std::is_floating_point<Tp>::value, Tp>::type
d_sigmoid(Tp x) {
  // Tp cache = tensor::util::sigmoid(x);
  // return cache * (1.0 - cache); }
  return x * (1 - x); 
}
FOR_EACH_IN_VEC_MAT(d_sigmoid, tensor::util::d_sigmoid)

// ==== Tanh ====
FOR_EACH_IN_VEC_MAT(tanh, std::tanh)

template <typename Tp>
typename std::enable_if<std::is_floating_point<Tp>::value, Tp>::type
d_tanh(Tp x) { 
  // return std::tanh(x) * (1.0 - std::tanh(x)); 
  return x * (1.0 - x);
}
FOR_EACH_IN_VEC_MAT(d_tanh, tensor::util::d_tanh)

// ==== Softmax ====
template <typename Tp>
Vector<Tp> softmax(const Vector<Tp>& vec) {
  Vector<Tp> cache_vec = tensor::util::exp(vec - vec.max());
  return cache_vec / cache_vec.sum();
}
template <typename Tp>
Matrix<Tp> softmax(const Matrix<Tp>& mat) {
  Matrix<Tp> mat_res(mat.shape()[0], mat.shape()[1]);
  for (typename Matrix<Tp>::size_type idx_row = 0;
       idx_row < mat_res.shape()[0]; ++idx_row) {
    mat_res[idx_row] = tensor::util::softmax(mat[idx_row]);
  }
  return mat_res;
}
template <typename Tp>
Vector<Tp> d_softmax(const Vector<Tp>& vec) {
  // Vector<Tp> cache_vec = tensor::util::softmax(vec);
  // return cache_vec * (1.0 - cache_vec);
  return vec * (1.0 - vec);
}
template <typename Tp>
Matrix<Tp> d_softmax(const Matrix<Tp>& mat) {
  // Matrix<Tp> cache_mat = tensor::util::softmax(mat);
  // return cache_mat.times(1.0 - cache_mat);
  return mat.times(1.0 - mat);
}

// ==== Softplus ====
template <typename Tp>
typename std::enable_if<std::is_floating_point<Tp>::value, Tp>::type
softplus(Tp x) { return std::log1p(std::exp(x)); }  // log(1 + exp(x))
FOR_EACH_IN_VEC_MAT(softplus, tensor::util::softplus)

template <typename Tp>
typename std::enable_if<std::is_floating_point<Tp>::value, Tp>::type
d_softplus(Tp x) { 
  // return 1.0 / (1.0 + std::exp(-x)); 
  return (std::exp(x) - 1.0) / std::exp(x);
}
FOR_EACH_IN_VEC_MAT(d_softplus, tensor::util::d_softplus)


// ======== Statistics ========

template <typename Tp>
Tp mean(const Vector<Tp>& vec) {
  return vec.sum() / static_cast<Tp>(vec.shape()[0]);
}
template <typename Tp>
Tp mean(const Matrix<Tp>& mat) {
  return mat.sum() / static_cast<Tp>(mat.shape()[0] * mat.shape()[1]);
}
template <typename Tp>
Vector<Tp> mean(const Matrix<Tp>& mat, 
                const typename Matrix<Tp>::dimension_type& dim) {
  return mat.sum(dim) / static_cast<Tp>(mat.shape()[dim]);
}

template <typename Tp>
Tp variance(const Vector<Tp>& vec) {
  return tensor::util::pow(vec - tensor::util::mean(vec), 2).sum() /
         static_cast<Tp>(vec.shape()[0]);
}
template <typename Tp>
Tp variance(const Matrix<Tp>& mat) {
  return tensor::util::pow(mat - tensor::util::mean(mat), 2).sum() /
         static_cast<Tp>(mat.shape()[0] * mat.shape()[1]);
}
template <typename Tp>
Vector<Tp> variance(const Matrix<Tp>& mat,
                    const typename Matrix<Tp>::dimension_type& dim) {
  if (dim != 0 && dim != 1) {
    std::string err_msg = "Invalid dimension: " + std::to_string(dim) +
      " != 0 or 1.";
    throw MatrixException(err_msg);
  }
  Matrix<Tp> mean_mat;
  if (dim == 0) {
    mean_mat = mat;
  } else {
    mean_mat = mat.T();
  }
  for (Vector<Tp>& row : mean_mat) {
    row = tensor::util::pow(row - mean(row), 2);
  }
  return mean_mat.sum(0) / static_cast<Tp>(mat.shape()[0]);
}
template <typename Tp>
Tp stddev(const Vector<Tp>& vec) {
  return std::sqrt(tensor::util::variance(vec));
}
template <typename Tp>
Tp stddev(const Matrix<Tp>& mat) {
  return std::sqrt(tensor::util::variance(mat));
}
template <typename Tp>
Vector<Tp> stddev(const Matrix<Tp>& mat,
                  const typename Matrix<Tp>::dimension_type& dim) {
  return tensor::util::sqrt(tensor::util::variance(mat, dim));
}

// binomial_sample
// normal_sample


// ======== ConvNet ========
// zero_padding
// convolve
// max_pool
// avg_pool


// ======== Cross Entropy ========
template <typename Tp>
Vector<Tp> cross_entropy(const Vector<Tp>& prediction, 
                         const Vector<Tp>& target) {
  return Vector<Tp>(1, -(target * tensor::util::log(prediction).sum()));
}
template <typename Tp>
Matrix<Tp> cross_entropy(const Matrix<Tp>& prediction,
                         const Matrix<Tp>& target) {
  Matrix<Tp> res(prediction.shape()[0], 1);
  for (typename Matrix<Tp>::size_type idx_row = 0; 
       idx_row < res.shape()[0]; ++idx_row) {
    res[idx_row] = cross_entropy(prediction[idx_row], target[idx_row]);
  }
  return res;
}

#undef FOR_EACH_IN_VEC_MAT

}  // namespace util
}  // namespace tensor

#endif  // TENSOR_UTIL_H_