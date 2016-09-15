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

- 2016-08-12

- ======== tensor::Tensor ========

- struct TensorType
- type Tensor

*/

#ifndef TENSOR_TENSOR_H_
#define TENSOR_TENSOR_H_

#include "vector.h"
#include "matrix.h"
#include <vector>

namespace tensor {

typedef std::size_t dimension_type;

template <typename T, dimension_type Dim>
struct TensorType {
  typedef std::vector<typename TensorType<T, Dim - 1>::type> type;
};

template <typename T>
struct TensorType<T, 2> {
  typedef Matrix<T> type;
};

template <typename T>
struct TensorType<T, 1> {
  typedef Vector<T> type;
};

template <typename T, dimension_type Dim>
using Tensor = typename TensorType<T, Dim>::type;

}  // namespace tensor

#endif  // TENSOR_TENSOR_H_