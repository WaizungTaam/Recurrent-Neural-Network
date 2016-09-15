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

- 2016-08-17

- ======== tensor::Matrix ========

- namespace tensor

  - class Matrix
    - Declaration
      - type
      - constructors, etc
      - shape
      - iterators
      - accessors
      - modifiers
      - arithmetic
      - comparisons
      - io
      - helper functions
      - private data member
    - Implementation
      - Same order as declared
      - namespace internal before
        - transpose
        - arithmetic
        - comparisons

  - class MatrixException
  
*/

#ifndef TENSOR_MATRIX_H_
#define TENSOR_MATRIX_H_

#include "vector.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>
#include <x86intrin.h>

namespace tensor {

class MatrixException;


template <typename Tp>
class Matrix {
public:
  // ======== Types ========
  typedef Tp value_type;
  typedef typename std::vector<Vector<Tp>>::size_type size_type;
  typedef typename std::make_signed<size_type>::type index_type;
  typedef typename std::vector<Vector<Tp>>::difference_type difference_type;
  typedef size_type dimension_type;
  typedef typename std::vector<Vector<Tp>>::iterator iterator;
  typedef typename std::vector<Vector<Tp>>::const_iterator const_iterator;
  typedef typename std::vector<Vector<Tp>>::reverse_iterator reverse_iterator;
  typedef typename std::vector<Vector<Tp>>::const_reverse_iterator
    const const_reverse_iterator;

  // ======== Constructors, etc ========
  Matrix();

  Matrix(const size_type& num_rows, const size_type& num_cols);
  Matrix(const size_type& num_rows, const size_type& num_cols, 
         const Tp& val_init);
  template <typename OtherT>
  Matrix(const size_type& num_rows, const size_type& num_cols,
         const OtherT& val_cast);

  Matrix(const Matrix& mat_init);
  template <typename OtherT>
  Matrix(const Matrix<OtherT>& mat_cast);

  Matrix(Matrix&& mat_init);
  template <typename OtherT>
  Matrix(Matrix<OtherT>&& mat_cat);

  /*explicit */Matrix(const std::vector<std::vector<Tp>>& stdvec_init);
  template <typename OtherT>
  /*explicit */Matrix(const std::vector<std::vector<OtherT>>& stdvec_cast);

  /*explicit */Matrix(const std::initializer_list<
                      std::initializer_list<Tp>>& il_init);
  template <typename OtherT>
  /*explicit */Matrix(const std::initializer_list<
                      std::initializer_list<OtherT>>& il_cast);

  explicit Matrix(const Vector<Tp>& vec_init);
  template <typename OtherT>
  explicit Matrix(const Vector<OtherT>& vec_cast);

  template <typename ParamT1, typename ParamT2>
  Matrix(const size_type& num_rows, const size_type& num_cols, 
         Random::Distribution dis, const ParamT1& param1, 
         const ParamT2& param2);
  template <typename ParamT>
  Matrix(const size_type& num_rows, const size_type& num_cols,
         Random::Distribution dis, const ParamT& param);

  Matrix& operator=(const Tp& val_assign);
  template <typename OtherT>
  Matrix& operator=(const OtherT& val_cast);

  Matrix& operator=(const Matrix& mat_copy);
  template <typename OtherT>
  Matrix& operator=(const Matrix<OtherT>& mat_cast);

  Matrix& operator=(Matrix&& mat_move);
  template <typename OtherT>
  Matrix& operator=(Matrix<OtherT>&& mat_cast);

  Matrix& operator=(const std::vector<std::vector<Tp>>& stdvec_assign);
  template <typename OtherT>
  Matrix& operator=(const std::vector<std::vector<OtherT>>& stdvec_cast);

  Matrix& operator=(const std::initializer_list<
                    std::initializer_list<Tp>>& il_assign);
  template <typename OtherT>
  Matrix& operator=(const std::initializer_list<
                    std::initializer_list<OtherT>>& il_cast);

  Matrix& operator=(const Vector<Tp>& vec_assign);
  template <typename OtherT>
  Matrix& operator=(const Vector<OtherT>& vec_cast);

  ~Matrix();

  // ======== Shape ========
  Vector<size_type> shape() const;
  void clear();
  bool empty();

  // ======== Iterators ========
  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator cbegin() const;
  const_iterator cend() const;
  reverse_iterator rbegin();
  reverse_iterator rend();
  const_reverse_iterator rbegin() const;
  const_reverse_iterator rend() const;
  const_reverse_iterator crbegin() const;
  const_reverse_iterator crend() const;  

  // ======== Accessors ========
  Vector<Tp>& operator[](const index_type& index);
  const Vector<Tp>& operator[](const index_type& index) const;
  Matrix operator()(const index_type& idx_row) const;
  Matrix operator()(const const_iterator& cit_row) const;
  Matrix operator()(const index_type& idx_row_begin, 
                    const index_type& idx_row_end) const;
  Matrix operator()(const const_iterator& cit_row_begin,
                    const const_iterator& cit_row_end) const;
  Matrix operator()(const index_type& idx_row_begin,
                    const index_type& idx_row_end,
                    const index_type& idx_col_begin,
                    const index_type& idx_col_end) const;

  // ======== Modifiers ========
  Matrix insert(const Vector<Tp>& vec_insert, const dimension_type& dim_insert,
                const index_type& idx_insert) const;
  Matrix insert(const Matrix& mat_insert, const dimension_type& dim_insert,
                const index_type& idx_insert) const;
  Matrix remove(const dimension_type& dim_remove, const 
                index_type& idx_remove) const;
  Matrix remove(const dimension_type& dim_remove, const index_type& idx_begin,
                const index_type& idx_end) const;
  Matrix replace(const Vector<Tp>& vec_replace, 
                 const dimension_type& dim_replace,
                 const index_type& idx_row_begin, 
                 const index_type& idx_col_begin) const;
  Matrix replace(const Matrix& mat_replace, const index_type& idx_row_begin, 
                 const index_type& idx_col_begin) const;
  Matrix transpose() const;
  Matrix T() const;
  Matrix reshape(const size_type& num_rows, const size_type& num_cols) const;
  Matrix shuffle() const;

  // ======== Arithmetic ========
  template <typename AriT> 
  friend Matrix<AriT> operator+(const Matrix<AriT>& mat_lhs, 
                                const Matrix<AriT>& mat_rhs);
  template <typename AriT> 
  friend Matrix<AriT> operator+(const Matrix<AriT>& mat_lhs, 
                                const AriT& val_rhs);
  template <typename AriT> 
  friend Matrix<AriT> operator+(const AriT& val_lhs, 
                                const Matrix<AriT>& mat_rhs);
  template <typename AriT> 
  friend Matrix<AriT> operator-(const Matrix<AriT>& mat_lhs, 
                                const Matrix<AriT>& mat_rhs);
  template <typename AriT> 
  friend Matrix<AriT> operator-(const Matrix<AriT>& mat_lhs, 
                                const AriT& val_rhs);
  template <typename AriT> 
  friend Matrix<AriT> operator-(const AriT& val_lhs, 
                                const Matrix<AriT>& mat_rhs);
  template <typename AriT> 
  friend Matrix<AriT> operator*(const Matrix<AriT>& mat_lhs, 
                                const Matrix<AriT>& mat_rhs);
  template <typename AriT> 
  friend Matrix<AriT> operator*(const Matrix<AriT>& mat_lhs, 
                                const AriT& val_rhs);
  template <typename AriT>
  friend Matrix<AriT> operator*(const Matrix<AriT>& mat_lhs,
                                const Vector<AriT>& vec_rhs);
  template <typename AriT>
  friend Matrix<AriT> operator*(const Vector<AriT>& vec_lhs,
                                const Matrix<AriT>& mat_rhs);
  template <typename AriT> 
  friend Matrix<AriT> operator*(const AriT& val_lhs, 
                                const Matrix<AriT>& mat_rhs);
  template <typename AriT> 
  friend Matrix<AriT> operator/(const Matrix<AriT>& mat_lhs, 
                                const Matrix<AriT>& mat_rhs);
  template <typename AriT> 
  friend Matrix<AriT> operator/(const Matrix<AriT>& mat_lhs, 
                                const AriT& val_rhs);
  template <typename AriT> 
  friend Matrix<AriT> operator/(const AriT& val_lhs, 
                                const Matrix<AriT>& mat_rhs);
  void operator+=(const Matrix& mat_rhs);
  void operator+=(const Tp& val_rhs);
  void operator-=(const Matrix& mat_rhs);
  void operator-=(const Tp& val_rhs);
  void operator*=(const Matrix& mat_rhs);
  void operator*=(const Tp& val_rhs);
  void operator/=(const Matrix& mat_rhs);
  void operator/=(const Tp& val_rhs);
  Matrix times(const Matrix& mat_rhs) const;
  Tp sum() const;
  Vector<Tp> sum(const dimension_type& dim_sum) const;

  // ======== Comparisons ========
  template <typename CmpT> 
  friend Matrix<CmpT> operator==(const Matrix<CmpT>& mat_lhs, 
                                 const Matrix<CmpT>& mat_rhs);
  template <typename CmpT> 
  friend Matrix<CmpT> operator==(const Matrix<CmpT>& mat_lhs, 
                                 const CmpT& val_rhs);
  template <typename CmpT> 
  friend Matrix<CmpT> operator==(const CmpT& val_lhs, 
                                 const Matrix<CmpT>& mat_rhs);
  template <typename CmpT> 
  friend Matrix<CmpT> operator!=(const Matrix<CmpT>& mat_lhs, 
                         const Matrix<CmpT>& mat_rhs);  
  template <typename CmpT>
  friend Matrix<CmpT> operator!=(const Matrix<CmpT>& mat_lhs, 
                                 const CmpT& val_rhs);
  template <typename CmpT> 
  friend Matrix<CmpT> operator!=(const CmpT& val_lhs, 
                                 const Matrix<CmpT>& mat_rhs);
  template <typename CmpT>
  friend Matrix<CmpT> operator<(const Matrix<CmpT>& mat_lhs,
                                const Matrix<CmpT>& mat_rhs);
  template <typename CmpT> 
  friend Matrix<CmpT> operator<(const Matrix<CmpT>& mat_lhs, 
                                const CmpT& val_rhs);
  template <typename CmpT> 
  friend Matrix<CmpT> operator<(const CmpT& val_lhs, 
                                const Matrix<CmpT>& mat_rhs);
  template <typename CmpT>
  friend Matrix<CmpT> operator<=(const Matrix<CmpT>& mat_lhs,
                                 const Matrix<CmpT>& mat_rhs);
  template <typename CmpT> 
  friend Matrix<CmpT> operator<=(const Matrix<CmpT>& mat_lhs, 
                                 const CmpT& val_rhs);
  template <typename CmpT> 
  friend Matrix<CmpT> operator<=(const CmpT& val_lhs, 
                                 const Matrix<CmpT>& mat_rhs);
  template <typename CmpT>
  friend Matrix<CmpT> operator>(const Matrix<CmpT>& mat_lhs,
                                const Matrix<CmpT>& mat_rhs);
  template <typename CmpT> 
  friend Matrix<CmpT> operator>(const Matrix<CmpT>& mat_lhs, 
                                const CmpT& val_rhs);
  template <typename CmpT> 
  friend Matrix<CmpT> operator>(const CmpT& val_lhs, 
                                const Matrix<CmpT>& mat_rhs);
  template <typename CmpT>
  friend Matrix<CmpT> operator>=(const Matrix<CmpT>& mat_lhs,
                                 const Matrix<CmpT>& mat_rhs);
  template <typename CmpT> 
  friend Matrix<CmpT> operator>=(const Matrix<CmpT>& mat_lhs, 
                                 const CmpT& val_rhs);
  template <typename CmpT> 
  friend Matrix<CmpT> operator>=(const CmpT& val_lhs, 
                                 const Matrix<CmpT>& mat_rhs);
  bool equal(const Matrix& mat_rhs, std::size_t ulp = 1);
  bool nequal(const Matrix& mat_rhs, std::size_t ulp = 1);
  Tp max() const;
  Vector<Tp> max(const dimension_type& dim_max) const;
  Tp min() const;
  Vector<Tp> min(const dimension_type& dim_min) const;

  // ======== IO ========
  template <typename MatT, typename CharT, typename Traits>
  friend std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os, const Matrix<MatT>& mat);
  template <typename MatT, typename CharT, typename Traits>
  friend std::basic_istream<CharT, Traits>& operator>>(
    std::basic_istream<CharT, Traits>& is, Matrix<MatT>& mat);

private:
  // ======== Helper Functions ========
  static index_type to_positive_index_(const size_type& size,
                                       const index_type& index);  
  static void exclusive_range_check_(const size_type& size, 
                                     const index_type& index);
  static void exclusive_range_check_(const iterator& it_begin,
                                     const iterator& it_end,
                                     const iterator& it);
  static void exclusive_range_check_(const const_iterator& cit_begin,
                                     const const_iterator& cit_end,
                                     const const_iterator& cit);
  static void inclusive_range_check_(const size_type& size,
                                     const index_type& index);
  static void inclusive_range_check_(const iterator& it_begin,
                                     const iterator& it_end,
                                     const iterator& it);
  static void inclusive_range_check_(const const_iterator& cit_begin,
                                     const const_iterator& cit_end,
                                     const const_iterator& cit);  
  static void shape_consistence_check_(const Vector<size_type>& shape_lhs,
                                       const Vector<size_type>& shape_rhs);
  static void index_order_check_(const size_type& size,
                                 const index_type& idx_begin,
                                 const index_type& idx_end);

  index_type to_positive_index_(const index_type& index) const;
  void exclusive_range_check_(const index_type& index) const;
  void exclusive_range_check_(const iterator& it);
  void exclusive_range_check_(const const_iterator& cit) const;
  void inclusive_range_check_(const index_type& index) const;
  void inclusive_range_check_(const iterator& it);
  void inclusive_range_check_(const const_iterator& cit) const;
  void index_order_check_(const index_type& idx_begin,
                          const index_type& idx_end) const;
  void iterator_order_check_(const iterator& it_begin, 
                             const iterator& it_end);
  void const_iterator_order_check_(const const_iterator& cit_begin,
                                   const const_iterator& cit_end) const;

  // ======== Private Data Member ========  
  std::vector<Vector<Tp>> mat_;
};


// ======== Constructors, etc ========
template <typename Tp>
Matrix<Tp>::Matrix() {}

template <typename Tp>
Matrix<Tp>::Matrix(const size_type& num_rows, const size_type& num_cols) :
  mat_(std::vector<Vector<Tp>>(num_rows, Vector<Tp>(num_cols))) {}
template <typename Tp>
Matrix<Tp>::Matrix(const size_type& num_rows, const size_type& num_cols,
                   const Tp& val_init) :
  mat_(std::vector<Vector<Tp>>(num_rows, Vector<Tp>(num_cols, val_init))) {}
template <typename Tp>  template <typename OtherT>
Matrix<Tp>::Matrix(const size_type& num_rows, const size_type& num_cols,
                   const OtherT& val_cast) :
  mat_(std::vector<Vector<Tp>>(num_rows, Vector<Tp>(num_cols, val_cast))) {}

template <typename Tp>
Matrix<Tp>::Matrix(const Matrix& mat_init) : mat_(mat_init.mat_) {}
template <typename Tp>  template <typename OtherT>
Matrix<Tp>::Matrix(const Matrix<OtherT>& mat_cast) {
  mat_.resize(mat_cast.shape()[0]);
  for (size_type idx_row = 0; idx_row < mat_.size(); ++idx_row) {
    mat_[idx_row] = Vector<Tp>(mat_cast[idx_row]);
  }
}

template <typename Tp>
Matrix<Tp>::Matrix(Matrix&& mat_init) : mat_(std::move(mat_init.mat_)) {}
template <typename Tp>  template <typename OtherT>
Matrix<Tp>::Matrix(Matrix<OtherT>&& mat_cast) {
  mat_.resize(mat_cast.shape()[0]);
  Matrix<OtherT> mat_cache = std::move(mat_cast);
  for (size_type idx_row = 0; idx_row < mat_.size(); ++idx_row) {
    mat_[idx_row] = Vector<Tp>(mat_cache[idx_row]);
  }
}

template <typename Tp>
Matrix<Tp>::Matrix(const std::vector<std::vector<Tp>>& stdvec_init) {
  mat_.resize(stdvec_init.size());
  for (size_type idx_row = 0; idx_row < mat_.size(); ++idx_row) {
    mat_[idx_row] = Vector<Tp>(stdvec_init[idx_row]);
  }
}
template <typename Tp>  template <typename OtherT>
Matrix<Tp>::Matrix(const std::vector<std::vector<OtherT>>& stdvec_cast) {
  mat_.resize(stdvec_cast.size());
  for (size_type idx_row = 0; idx_row < mat_.size(); ++idx_row) {
    mat_[idx_row] = Vector<Tp>(stdvec_cast[idx_row]);
  }
}

template <typename Tp>
Matrix<Tp>::Matrix(const std::initializer_list<
                   std::initializer_list<Tp>>& il_init) {
  mat_.resize(il_init.size());
  for (size_type idx_row = 0; idx_row < mat_.size(); ++idx_row) {
    mat_[idx_row] = Vector<Tp>(*(il_init.begin() + idx_row));
  }
}
template <typename Tp>  template <typename OtherT>
Matrix<Tp>::Matrix(const std::initializer_list<
                   std::initializer_list<OtherT>>& il_cast) {
  mat_.resize(il_cast.size());
  for (size_type idx_row = 0; idx_row < mat_.size(); ++idx_row) {
    mat_[idx_row] = Vector<Tp>(*(il_cast.begin() + idx_row));
  }
}

template <typename Tp>
Matrix<Tp>::Matrix(const Vector<Tp>& vec_init) {
  mat_.resize(vec_init.shape()[0]);
  for (size_type idx_row = 0; idx_row < mat_.size(); ++idx_row) {
    mat_[idx_row] = Vector<Tp>(1, vec_init[idx_row]);
  }
}
template <typename Tp>  template <typename OtherT>
Matrix<Tp>::Matrix(const Vector<OtherT>& vec_cast) {
  mat_.resize(vec_cast.shape()[0]);
  for (size_type idx_row = 0; idx_row < mat_.size(); ++idx_row) {
    mat_[idx_row] = Vector<Tp>(1, vec_cast[idx_row]);
  }
}

template <typename Tp>  template <typename ParamT1, typename ParamT2>
Matrix<Tp>::Matrix(const size_type& num_rows, const size_type& num_cols,
                   Random::Distribution dis, const ParamT1& param1,
                   const ParamT2& param2) {
  mat_.resize(num_rows);
  for (size_type idx_row = 0; idx_row < mat_.size(); ++idx_row) {
    mat_[idx_row] = Vector<Tp>(num_cols, dis, param1, param2);
  }
}
template <typename Tp>  template <typename ParamT>
Matrix<Tp>::Matrix(const size_type& num_rows, const size_type& num_cols,
                   Random::Distribution dis, const ParamT& param) {
  mat_.resize(num_rows);
  for (size_type idx_row = 0; idx_row < mat_.size(); ++idx_row) {
    mat_[idx_row] = Vector<Tp>(num_cols, dis, param);
  }
}

template <typename Tp>
Matrix<Tp>& Matrix<Tp>::operator=(const Tp& val_assign) {
  mat_ = Matrix<Tp>(shape()[0], shape()[1], val_assign).mat_;
  return *this;
}
template <typename Tp>  template <typename OtherT>
Matrix<Tp>& Matrix<Tp>::operator=(const OtherT& val_cast) {
  mat_ = Matrix<Tp>(shape()[0], shape()[1], val_cast).mat_;
  return *this;
}

template <typename Tp>
Matrix<Tp>& Matrix<Tp>::operator=(const Matrix& mat_copy) {
  mat_ = mat_copy.mat_;
  return *this;
}
template <typename Tp>  template <typename OtherT>
Matrix<Tp>& Matrix<Tp>::operator=(const Matrix<OtherT>& mat_cast) {
  mat_ = Matrix<Tp>(mat_cast).mat_;
  return *this;
}

template <typename Tp>
Matrix<Tp>& Matrix<Tp>::operator=(Matrix&& mat_move) {
  mat_ = Matrix<Tp>(mat_move).mat_;
  return *this;
}
template <typename Tp>  template <typename OtherT>
Matrix<Tp>& Matrix<Tp>::operator=(Matrix<OtherT>&& mat_cast) {
  mat_ = Matrix<Tp>(mat_cast).mat_;
  return *this;
}

template <typename Tp>
Matrix<Tp>& Matrix<Tp>::operator=(const std::vector<
                                  std::vector<Tp>>& stdvec_assign) {
  mat_ = Matrix<Tp>(stdvec_assign).mat_;
  return *this;
}
template <typename Tp>  template <typename OtherT>
Matrix<Tp>& Matrix<Tp>::operator=(const std::vector<
                                  std::vector<OtherT>>& stdvec_cast) {
  mat_ = Matrix<Tp>(stdvec_cast).mat_;
  return *this;
}

template <typename Tp>
Matrix<Tp>& Matrix<Tp>::operator=(const std::initializer_list<
                              std::initializer_list<Tp>>& il_assign) {
  mat_ = Matrix<Tp>(il_assign).mat_;
  return *this;

}
template <typename Tp>  template <typename OtherT>
Matrix<Tp>& Matrix<Tp>::operator=(const std::initializer_list<
                              std::initializer_list<OtherT>>& il_cast) {
  mat_ = Matrix<Tp>(il_cast).mat_;
  return *this;
}

template <typename Tp>
Matrix<Tp>& Matrix<Tp>::operator=(const Vector<Tp>& vec_assign) {
  mat_ = Matrix(vec_assign).mat_;
  return *this;
}
template <typename Tp>  template <typename OtherT>
Matrix<Tp>& Matrix<Tp>::operator=(const Vector<OtherT>& vec_cast) {
  mat_ = Matrix(vec_cast).mat_;
  return *this;
}

template <typename Tp>
Matrix<Tp>::~Matrix() {}

// ======== Shape ========
template <typename Tp>
Vector<typename Matrix<Tp>::size_type> Matrix<Tp>::shape() const {
  if (mat_.size() == 0) {
    return Vector<size_type>({0, 0});
  }
  return Vector<size_type>({mat_.size(), mat_[0].shape()[0]});
}
template <typename Tp>
void Matrix<Tp>::clear() { mat_.clear(); }
template <typename Tp>
bool Matrix<Tp>::empty() { return mat_.size() == 0; }

// ======== Iterators ========
template <typename Tp> typename
Matrix<Tp>::iterator Matrix<Tp>::begin() { return mat_.begin(); }
template <typename Tp> typename
Matrix<Tp>::iterator Matrix<Tp>::end() { return mat_.end(); }
template <typename Tp> typename
Matrix<Tp>::const_iterator Matrix<Tp>::begin() const { return mat_.cbegin(); }
template <typename Tp> typename
Matrix<Tp>::const_iterator Matrix<Tp>::end() const { return mat_.cend(); }
template <typename Tp> typename
Matrix<Tp>::const_iterator Matrix<Tp>::cbegin() const { return mat_.cbegin(); }
template <typename Tp> typename
Matrix<Tp>::const_iterator Matrix<Tp>::cend() const { return mat_.cend(); }
template <typename Tp> typename
Matrix<Tp>::reverse_iterator Matrix<Tp>::rbegin() { return mat_.rbegin(); }
template <typename Tp> typename
Matrix<Tp>::reverse_iterator Matrix<Tp>::rend() { return mat_.rend(); }
template <typename Tp> typename
Matrix<Tp>::const_reverse_iterator Matrix<Tp>::rbegin() const {
  return mat_.crbegin();
}
template <typename Tp> typename
Matrix<Tp>::const_reverse_iterator Matrix<Tp>::rend() const {
  return mat_.crend();
}
template <typename Tp> typename
Matrix<Tp>::const_reverse_iterator Matrix<Tp>::crbegin() const {
  return mat_.crbegin();
}
template <typename Tp> typename
Matrix<Tp>::const_reverse_iterator Matrix<Tp>::crend() const {
  return mat_.crend();
}

// ======== Accessors ========
template <typename Tp>
Vector<Tp>& Matrix<Tp>::operator[](const index_type& index) {
  exclusive_range_check_(index);
  return mat_.at(to_positive_index_(index));
}
template <typename Tp>
const Vector<Tp>& Matrix<Tp>::operator[](const index_type& index) const {
  exclusive_range_check_(index);
  return mat_.at(to_positive_index_(index));
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::operator()(const index_type& idx_row) const {
  exclusive_range_check_(idx_row);
  Matrix row_mat(1, shape()[1]);
  row_mat[0] = mat_[to_positive_index_(idx_row)];
  return row_mat;
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::operator()(const const_iterator& cit_row) const {
  exclusive_range_check_(cit_row);
  Matrix row_mat(1, shape()[1]);
  row_mat[0] = *cit_row;
  return row_mat;
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::operator()(const index_type& idx_row_begin, 
                                  const index_type& idx_row_end) const {
  exclusive_range_check_(idx_row_begin);
  inclusive_range_check_(idx_row_end);
  index_order_check_(idx_row_begin, idx_row_end);
  size_type idx_row_begin_p = to_positive_index_(idx_row_begin);
  size_type idx_row_end_p = to_positive_index_(idx_row_end);
  Matrix mat_partial(idx_row_end_p - idx_row_begin_p, shape()[1]);
  for (size_type idx_row = 0; idx_row < mat_partial.shape()[0]; ++idx_row)  {
    mat_partial[idx_row] = mat_[idx_row_begin_p + idx_row];
  }
  return mat_partial;
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::operator()(const const_iterator& cit_row_begin,
                                  const const_iterator& cit_row_end) const {
  exclusive_range_check_(cit_row_begin);
  inclusive_range_check_(cit_row_end);
  const_iterator_order_check_(cit_row_begin, cit_row_end);
  Matrix mat_partial(static_cast<size_type>(cit_row_end - cit_row_begin),
                     shape()[1]);
  for (size_type idx_row = 0; idx_row < mat_partial.shape()[0]; ++idx_row) {
    mat_partial[idx_row] = *(cit_row_begin + idx_row);
  }
  return mat_partial;
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::operator()(const index_type& idx_row_begin,
                                  const index_type& idx_row_end,
                                  const index_type& idx_col_begin,
                                  const index_type& idx_col_end) const {
  exclusive_range_check_(idx_row_begin);
  inclusive_range_check_(idx_row_end);
  exclusive_range_check_(idx_col_begin);
  inclusive_range_check_(idx_col_end);
  index_order_check_(idx_row_begin, idx_row_end);
  index_order_check_(idx_col_begin, idx_col_end);
  size_type idx_row_begin_p = to_positive_index_(idx_row_begin);
  size_type idx_row_end_p = to_positive_index_(idx_row_end);
  size_type idx_col_begin_p = to_positive_index_(idx_col_begin);
  size_type idx_col_end_p = to_positive_index_(idx_col_end);
  Matrix mat_partial(idx_row_end_p - idx_row_begin_p, 
                     idx_col_end_p - idx_col_begin_p);
  for (size_type idx_row = 0; idx_row < mat_partial.shape()[0]; ++idx_row)  {
    mat_partial[idx_row] = mat_[idx_row_begin_p + idx_row](
                           idx_col_begin_p, idx_col_end_p);
  }
  return mat_partial;
}

// ======== Modifiers ========
template <typename Tp>
Matrix<Tp> Matrix<Tp>::insert(const Vector<Tp>& vec_insert, 
                              const dimension_type& dim_insert, 
                              const index_type& idx_insert) const {
  if (dim_insert == 0) {
    if (vec_insert.shape()[0] != shape()[1]) {
      std::string err_msg = "Inconsistent insert shape: insert size " +
        std::to_string(vec_insert.shape()[0]) + " != number of columns " +
        std::to_string(shape()[1]) + ".";
      throw MatrixException(err_msg);
    }
    inclusive_range_check_(idx_insert);
    Matrix mat_inserted = *this;
    mat_inserted.mat_.insert(
      mat_inserted.mat_.begin() + to_positive_index_(idx_insert), vec_insert);
    return mat_inserted;
  } else if (dim_insert == 1) {
    if (vec_insert.shape()[0] != shape()[0]) {
      std::string err_msg = "Inconsistent insert shape: insert size " +
        std::to_string(vec_insert.shape()[0]) + " != number of rows " +
        std::to_string(shape()[0]) + ".";
      throw MatrixException(err_msg);
    }
    Matrix mat_inserted = *this;
    for (size_type idx_row = 0; idx_row < mat_inserted.shape()[0]; ++idx_row) {
      mat_inserted.mat_[idx_row] = mat_inserted.mat_[idx_row].insert(
                                   vec_insert[idx_row], idx_insert);
    }
    return mat_inserted;
  } else {
    std::string err_msg = "Invalid Dimension: " + std::to_string(dim_insert) +
                          " != 0 or 1.";
    throw MatrixException(err_msg);
  }
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::insert(const Matrix& mat_insert, 
                              const dimension_type& dim_insert,
                              const index_type& idx_insert) const {
  if (dim_insert == 0) {
    if (mat_insert.shape()[1] != shape()[1]) {
      std::string err_msg = "Inconsistent insert shape: insert size " +
        std::to_string(mat_insert.shape()[1]) + " != number of columns " +
        std::to_string(shape()[1]) + ".";
      throw MatrixException(err_msg);
    }
    inclusive_range_check_(idx_insert);
    Matrix mat_inserted = *this;
    mat_inserted.mat_.insert(
      mat_inserted.mat_.begin() + to_positive_index_(idx_insert),
      mat_insert.mat_.begin(), mat_insert.mat_.end());
    return mat_inserted;
  } else if (dim_insert == 1) {
    if (mat_insert.shape()[0] != shape()[0]) {
      std::string err_msg = "Inconsistent insert shape: insert size " +
        std::to_string(mat_insert.shape()[0]) + " != number of rows " +
        std::to_string(shape()[0]) + ".";
      throw MatrixException(err_msg);
    }
    Matrix mat_inserted = *this;
    for (size_type idx_row = 0; idx_row < mat_inserted.shape()[0]; ++idx_row) {
      mat_inserted.mat_[idx_row] = mat_inserted.mat_[idx_row].insert(
                                   mat_insert.mat_[idx_row], idx_insert);
    }
    return mat_inserted;
  } else {
    std::string err_msg = "Invalid Dimension: " + std::to_string(dim_insert) +
                          " != 0 or 1.";
    throw MatrixException(err_msg);                        
  }
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::remove(const dimension_type& dim_remove, 
                              const index_type& idx_remove) const {
  if (dim_remove == 0) {
    exclusive_range_check_(idx_remove);
    Matrix mat_removed = *this;
    mat_removed.mat_.erase(mat_removed.mat_.begin() + 
                           to_positive_index_(idx_remove));
    return mat_removed;
  } else if (dim_remove == 1) {
    Matrix mat_removed = *this;
    for (size_type idx_row = 0; idx_row < mat_removed.shape()[0]; ++idx_row) {
      mat_removed.mat_[idx_row] = mat_removed.mat_[idx_row].remove(idx_remove);
    }
    return mat_removed;
  } else {
    std::string err_msg = "Invalid Dimension: " + std::to_string(dim_remove) +
                          " != 0 or 1.";
    throw MatrixException(err_msg);
  }
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::remove(const dimension_type& dim_remove, 
                              const index_type& idx_begin,
                              const index_type& idx_end) const {
  if (dim_remove == 0) {
    exclusive_range_check_(idx_begin);
    inclusive_range_check_(idx_end);
    index_order_check_(idx_begin, idx_end);
    Matrix mat_removed = *this;
    mat_removed.mat_.erase(
      mat_removed.mat_.begin() + to_positive_index_(idx_begin),
      mat_removed.mat_.begin() + to_positive_index_(idx_end));
    return mat_removed;
  } else if (dim_remove == 1) {
    Matrix mat_removed = *this;
    for (size_type idx_row = 0; idx_row < mat_removed.shape()[0]; ++idx_row) {
      mat_removed.mat_[idx_row] = mat_removed.mat_[idx_row].remove(
        idx_begin, idx_end);
    }
    return mat_removed;
  } else {
    std::string err_msg = "Invalid Dimension: " + std::to_string(dim_remove) +
                          " != 0 or 1.";
    throw MatrixException(err_msg);                          
  }
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::replace(const Vector<Tp>& vec_replace, 
                               const dimension_type& dim_replace,
                               const index_type& idx_row_begin, 
                               const index_type& idx_col_begin) const {
  if (dim_replace == 0) {
    exclusive_range_check_(idx_row_begin);
    Matrix mat_replaced = *this;
    mat_replaced[idx_row_begin] = mat_replaced[idx_row_begin].replace(
                                  vec_replace, idx_col_begin);
    return mat_replaced;
  } else if (dim_replace == 1) {
    exclusive_range_check_(idx_row_begin);
    size_type idx_row_begin_p = to_positive_index_(idx_row_begin);
    Matrix mat_replaced = *this;
    for (size_type idx_row = 0; idx_row < vec_replace.shape()[0] &&
         idx_row_begin_p + idx_row < mat_replaced.shape()[0]; ++idx_row) {
      mat_replaced.mat_[idx_row_begin_p + idx_row] = 
        mat_replaced.mat_[idx_row_begin_p + idx_row].replace(
        vec_replace[idx_row], idx_col_begin);
    }
    return mat_replaced;
  } else {
    std::string err_msg = "Invalid Dimension: " + std::to_string(dim_replace) +
                          " != 0 or 1.";
    throw MatrixException(err_msg);
  }
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::replace(const Matrix& mat_replace, 
                               const index_type& idx_row_begin, 
                               const index_type& idx_col_begin) const {
  exclusive_range_check_(idx_row_begin);
  size_type idx_row_begin_p = to_positive_index_(idx_row_begin);
  Matrix mat_replaced = *this;
  for (size_type idx_row = 0; idx_row < mat_replace.shape()[0] &&
       idx_row_begin_p + idx_row < mat_replaced.shape()[0]; ++idx_row) {
    mat_replaced.mat_[idx_row_begin_p + idx_row] = 
      mat_replaced.mat_[idx_row_begin_p + idx_row].replace(
      mat_replace[idx_row], idx_col_begin);
  }
  return mat_replaced;
}
// namespace internal
namespace internal {
template <typename Tp>
void transpose(const Matrix<Tp>& mat_from, Matrix<Tp>& mat_to) {
  typename Matrix<Tp>::size_type num_rows = mat_to.shape()[0],
                                 num_cols = mat_to.shape()[1],
                                 size_c_to_c_omp = 8;
  const Vector<Tp>* ptr_row_from = &mat_from[0];
  Vector<Tp>* ptr_row_to = &mat_to[0];
  if (num_rows + num_cols < 2 * size_c_to_c_omp) {
    for (typename Matrix<Tp>::size_type idx_col = 0; 
         idx_col < num_cols; ++idx_col) {
      const Tp* ptr_col_from = &ptr_row_from[idx_col][0];
      for (typename Matrix<Tp>::size_type idx_row = 0;
           idx_row < num_rows; ++idx_row) {
        Tp* ptr_col_to = &ptr_row_to[idx_row][0];
        ptr_col_to[idx_col] = ptr_col_from[idx_row];
      }
    }
  } else {
    #pragma omp parallel for shared(ptr_row_from, ptr_row_to) \
    schedule(auto) collapse(2)
    for (typename Matrix<Tp>::size_type idx_col = 0; 
         idx_col < num_cols; ++idx_col) {
      for (typename Matrix<Tp>::size_type idx_row = 0;
           idx_row < num_rows; ++idx_row) {
        const Tp* ptr_col_from = &ptr_row_from[idx_col][0];
        Tp* ptr_col_to = &ptr_row_to[idx_row][0];
        ptr_col_to[idx_col] = ptr_col_from[idx_row];
      }
    }    
  }
}
}  // namespace internal

template <typename Tp>
Matrix<Tp> Matrix<Tp>::transpose() const {
  Matrix mat_t(shape()[1], shape()[0]);
  internal::transpose(*this, mat_t);
  return mat_t;
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::T() const {
  return transpose();
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::reshape(const size_type& num_rows, 
                               const size_type& num_cols) const {
  Vector<Tp> vec_cache(num_rows * num_cols);
  bool is_end_of_vec_cache = false;
  typename Vector<Tp>::size_type idx_vec_cache = 0;
  for (size_type idx_row = 0; idx_row < shape()[0]; ++idx_row) {
    for (size_type idx_col = 0; idx_col < shape()[1]; ++idx_col) {
      vec_cache[idx_vec_cache] = mat_[idx_row][idx_col];
      ++idx_vec_cache;
      if (idx_vec_cache >= vec_cache.shape()[0]) {
        is_end_of_vec_cache = true;
        break;
      }
    }
    if (is_end_of_vec_cache) {
      break;
    }
  }
  Matrix<Tp> mat_reshaped(num_rows, num_cols);
  idx_vec_cache = 0;
  for (index_type idx_row = 0; idx_row < mat_reshaped.shape()[0]; ++idx_row) {
    for (index_type idx_col = 0; idx_col < mat_reshaped.shape()[1]; 
         ++idx_col) {
      mat_reshaped.mat_[idx_row][idx_col] = vec_cache[idx_vec_cache];
      ++idx_vec_cache;
    }
  }
  return mat_reshaped;
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::shuffle() const {
  std::random_device rd;
  std::default_random_engine gen(rd());
  Matrix mat_shuffled = *this;
  std::shuffle(mat_shuffled.mat_.begin(), mat_shuffled.mat_.end(), gen);
  return mat_shuffled;
}

// ======== Arithmetic ========
namespace internal {

#define OMP_FOR_3_MAT \
_Pragma("omp parallel for shared(mat_lhs, mat_rhs, mat_ans) schedule(auto)")
#define OMP_FOR_2_MAT_L_ANS \
_Pragma("omp parallel for shared(mat_lhs, val_rhs, mat_ans) schedule(auto)")
#define OMP_FOR_2_MAT_R_ANS \
_Pragma("omp parallel for shared(val_lhs, mat_rhs, mat_ans) schedule(auto)")

#define MAT_OP_MAT(OPERATION, OPERATOR) \
template <typename OpT> \
void OPERATION(const Matrix<OpT>& mat_lhs, const Matrix<OpT>& mat_rhs, \
               Matrix<OpT>& mat_ans) { \
  OMP_FOR_3_MAT \
  for (typename Matrix<OpT>::size_type idx_row = 0; \
       idx_row < mat_ans.shape()[0]; ++idx_row) { \
    mat_ans[idx_row] = (mat_lhs[idx_row] OPERATOR mat_rhs[idx_row]); \
  } \
}
MAT_OP_MAT(add, +)
MAT_OP_MAT(sub, -)
MAT_OP_MAT(mul, *)
MAT_OP_MAT(div, /)

#define MAT_OP_SCA(OPERATION, OPERATOR) \
template <typename OpT> \
void OPERATION(const Matrix<OpT>& mat_lhs, const OpT& val_rhs, \
               Matrix<OpT>& mat_ans) { \
  OMP_FOR_2_MAT_L_ANS \
  for (typename Matrix<OpT>::size_type idx_row = 0; \
       idx_row < mat_ans.shape()[0]; ++idx_row) { \
    mat_ans[idx_row] = (mat_lhs[idx_row] OPERATOR val_rhs); \
  } \
}
MAT_OP_SCA(add, +)
MAT_OP_SCA(sub, -)
MAT_OP_SCA(mul, *)
MAT_OP_SCA(div, /)

#define SCA_OP_MAT(OPERATION, OPERATOR) \
template <typename OpT> \
void OPERATION(const OpT& val_lhs, const Matrix<OpT>& mat_rhs, \
               Matrix<OpT>& mat_ans) { \
  OMP_FOR_2_MAT_R_ANS \
  for (typename Matrix<OpT>::size_type idx_row = 0; \
       idx_row < mat_ans.shape()[0]; ++idx_row) { \
    mat_ans[idx_row] = (val_lhs OPERATOR mat_rhs[idx_row]); \
  } \
}
SCA_OP_MAT(add, +)
SCA_OP_MAT(sub, -)
SCA_OP_MAT(mul, *)
SCA_OP_MAT(div, /)

}  // namespace internal

template <typename AriT>
Matrix<AriT> operator+(const Matrix<AriT>& mat_lhs, 
                       const Matrix<AriT>& mat_rhs) {
  Matrix<AriT>::shape_consistence_check_(mat_lhs.shape(), mat_rhs.shape());
  Matrix<AriT> mat_sum(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::add(mat_lhs, mat_rhs, mat_sum);
  return mat_sum;
}
template <typename AriT>
Matrix<AriT> operator+(const Matrix<AriT>& mat_lhs, const AriT& val_rhs) {
  Matrix<AriT> mat_sum(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::add(mat_lhs, val_rhs, mat_sum);
  return mat_sum;
}
template <typename AriT>
Matrix<AriT> operator+(const AriT& val_lhs, const Matrix<AriT>& mat_rhs) {
  Matrix<AriT> mat_sum(mat_rhs.shape()[0], mat_rhs.shape()[1]);
  internal::add(val_lhs,mat_rhs, mat_sum);
  return mat_sum;
}
template <typename AriT>
Matrix<AriT> operator-(const Matrix<AriT>& mat_lhs, 
                       const Matrix<AriT>& mat_rhs) {
  Matrix<AriT>::shape_consistence_check_(mat_lhs.shape(), mat_rhs.shape());
  Matrix<AriT> mat_diff(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::sub(mat_lhs, mat_rhs, mat_diff);
  return mat_diff;
}
template <typename AriT>
Matrix<AriT> operator-(const Matrix<AriT>& mat_lhs, const AriT& val_rhs) {
  Matrix<AriT> mat_diff(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::sub(mat_lhs, val_rhs, mat_diff);
  return mat_diff;
}
template <typename AriT>
Matrix<AriT> operator-(const AriT& val_lhs, const Matrix<AriT>& mat_rhs) {
  Matrix<AriT> mat_diff(mat_rhs.shape()[0], mat_rhs.shape()[1]);
  internal::sub(val_lhs, mat_rhs, mat_diff);
  return mat_diff;
}
// namespace internal
namespace internal {

template <typename MatT>
void mat_mul(const Matrix<MatT>& mat_lhs, const Matrix<MatT>& mat_rhs,
             Matrix<MatT>& mat_ans) {
  typename Matrix<MatT>::size_type num_rows = mat_ans.shape()[0],
                                   num_cols = mat_ans.shape()[1];
  Matrix<MatT> mat_rhs_t = mat_rhs.transpose();
  const Vector<MatT>* ptr_row_lhs = &mat_lhs[0];
  const Vector<MatT>* ptr_row_rhs = &mat_rhs_t[0];
  Vector<MatT>* ptr_row_ans = &mat_ans[0];
  #pragma omp parallel for shared(ptr_row_lhs, ptr_row_rhs, ptr_row_ans) \
  schedule(auto) collapse(2)
  for (typename Matrix<MatT>::size_type idx_row = 0;
       idx_row < num_rows; ++idx_row) {
    for (typename Matrix<MatT>::size_type idx_col = 0;
         idx_col < num_cols; ++idx_col) {
      *(&ptr_row_ans[idx_row][0] + idx_col) += 
        (ptr_row_lhs[idx_row] * ptr_row_rhs[idx_col]).sum();
    }
  }
}

}  // namespace internal

template <typename AriT>
Matrix<AriT> operator*(const Matrix<AriT>& mat_lhs, 
                       const Matrix<AriT>& mat_rhs) {
  if (mat_lhs.shape()[1] != mat_rhs.shape()[0]) {
    std::string err_msg = "Inconsistent shape for matrix multiplication: " +
      std::to_string(mat_lhs.shape()[1]) + " != " +
      std::to_string(mat_rhs.shape()[0]) + ".";
    throw MatrixException(err_msg);
  }
  Matrix<AriT> mat_prod(mat_lhs.shape()[0], mat_rhs.shape()[1]);
  internal::mat_mul(mat_lhs, mat_rhs, mat_prod);
  return mat_prod;
}
template <typename AriT>
Matrix<AriT> operator*(const Matrix<AriT>& mat_lhs, const AriT& val_rhs) {
  Matrix<AriT> mat_prod(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::mul(mat_lhs, val_rhs, mat_prod);
  return mat_prod;
}
template <typename AriT>
Matrix<AriT> operator*(const AriT& val_lhs, const Matrix<AriT>& mat_rhs) {
  Matrix<AriT> mat_prod(mat_rhs.shape()[0], mat_rhs.shape()[1]);
  internal::mul(val_lhs, mat_rhs, mat_prod);
  return mat_prod;
}
template <typename AriT>
Matrix<AriT> operator*(const Matrix<AriT>& mat_lhs, 
                       const Vector<AriT>& vec_rhs) {
  Matrix<AriT> mat_rhs(vec_rhs);
  return operator*(mat_lhs, mat_rhs);
}
template <typename AriT>
Matrix<AriT> operator*(const Vector<AriT>& vec_lhs,
                       const Matrix<AriT>& mat_rhs) {
  Matrix<AriT> mat_lhs(vec_lhs);
  return operator*(mat_lhs, mat_rhs);
}
template <typename AriT>
Matrix<AriT> operator/(const Matrix<AriT>& mat_lhs, 
                       const Matrix<AriT>& mat_rhs) {
  Matrix<AriT>::shape_consistence_check_(mat_lhs.shape(), mat_rhs.shape());
  Matrix<AriT> mat_quot(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::div(mat_lhs, mat_rhs, mat_quot);
  return mat_quot;
}
template <typename AriT>
Matrix<AriT> operator/(const Matrix<AriT>& mat_lhs, const AriT& val_rhs) {
  Matrix<AriT> mat_quot(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::div(mat_lhs, val_rhs, mat_quot);
  return mat_quot;
}
template <typename AriT>
Matrix<AriT> operator/(const AriT& val_lhs, const Matrix<AriT>& mat_rhs) {
  Matrix<AriT> mat_quot(mat_rhs.shape()[0], mat_rhs.shape()[1]);
  internal::div(val_lhs, mat_rhs, mat_quot);
  return mat_quot;
}
template <typename Tp>
void Matrix<Tp>::operator+=(const Matrix<Tp>& mat_rhs) {
  (*this) = (*this) + mat_rhs;
}
template <typename Tp>
void Matrix<Tp>::operator+=(const Tp& val_rhs) {
  (*this) = (*this) + val_rhs;
}
template <typename Tp>
void Matrix<Tp>::operator-=(const Matrix<Tp>& mat_rhs) {
  (*this) = (*this) - mat_rhs;
}
template <typename Tp>
void Matrix<Tp>::operator-=(const Tp& val_rhs) {
  (*this) = (*this) - val_rhs;
}
template <typename Tp>
void Matrix<Tp>::operator*=(const Matrix<Tp>& mat_rhs) {
  (*this) = (*this) * mat_rhs;
}
template <typename Tp>
void Matrix<Tp>::operator*=(const Tp& val_rhs) {
  (*this) = (*this) * val_rhs;
}
template <typename Tp>
void Matrix<Tp>::operator/=(const Matrix<Tp>& mat_rhs) {
  (*this) = (*this) / mat_rhs;
}
template <typename Tp>
void Matrix<Tp>::operator/=(const Tp& val_rhs) {
  (*this) = (*this) / val_rhs;
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::times(const Matrix& mat_rhs) const {
  shape_consistence_check_(shape(), mat_rhs.shape());
  Matrix mat_prod(shape()[0], shape()[1]);
  internal::mul(*this, mat_rhs, mat_prod);
  return mat_prod;
}
template <typename Tp>
Tp Matrix<Tp>::sum() const {
  Tp sum_val = Tp();
  #pragma omp parallel for schedule(auto) reduction(+ : sum_val)
  for (size_type idx_row = 0; idx_row < shape()[0]; ++idx_row) {
    sum_val = sum_val + mat_[idx_row].sum();
  }
  return sum_val;
}
// namespace internal
namespace internal {

template <typename SumT>
void sum_of_dim_one(const Matrix<SumT>& mat, Vector<SumT>& vec_sum) {
  for (typename Matrix<SumT>::size_type idx_row = 0; 
       idx_row < mat.shape()[0]; ++idx_row) {
    vec_sum[idx_row] = mat[idx_row].sum();
  }
}

}  // namespace internal

template <typename Tp>
Vector<Tp> Matrix<Tp>::sum(const dimension_type& dim_sum) const {
  if (dim_sum == 0) {
    Vector<Tp> vec_sum(shape()[1]);
    internal::sum_of_dim_one(transpose(), vec_sum);
    return vec_sum;
  } else if (dim_sum == 1) {
    Vector<Tp> vec_sum(shape()[0]);
    internal::sum_of_dim_one(*this, vec_sum);
    return vec_sum;
  } else {
    std::string err_msg = "Invalid Dimension: " + std::to_string(dim_sum) +
                          " != 0 or 1.";
    throw MatrixException(err_msg);    
  }
}

// ======== Comparisons ========
namespace internal {

MAT_OP_MAT(eq, ==)
MAT_OP_MAT(ne, !=)
MAT_OP_MAT(lt, <)
MAT_OP_MAT(le, <=)
MAT_OP_MAT(gt, >)
MAT_OP_MAT(ge, >=)

MAT_OP_SCA(eq, ==)
MAT_OP_SCA(ne, !=)
MAT_OP_SCA(lt, <)
MAT_OP_SCA(le, <=)
MAT_OP_SCA(gt, >)
MAT_OP_SCA(ge, >=)

SCA_OP_MAT(eq, ==)
SCA_OP_MAT(ne, !=)
SCA_OP_MAT(lt, <)
SCA_OP_MAT(le, <=)
SCA_OP_MAT(gt, >)
SCA_OP_MAT(ge, >=)

#undef MAT_OP_MAT
#undef MAT_OP_SCA
#undef SCA_OP_MAT

}  // namespace internal

template <typename CmpT>
Matrix<CmpT> operator==(const Matrix<CmpT>& mat_lhs,
                        const Matrix<CmpT>& mat_rhs) {
  Matrix<CmpT>::shape_consistence_check_(mat_lhs.shape(), mat_rhs.shape());
  Matrix<CmpT> mat_eq(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::eq(mat_lhs, mat_rhs, mat_eq);
  return mat_eq;
}
template <typename CmpT>
Matrix<CmpT> operator==(const Matrix<CmpT>& mat_lhs, const CmpT& val_rhs) {
  Matrix<CmpT> mat_eq(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::eq(mat_lhs, val_rhs, mat_eq);
  return mat_eq;
}
template <typename CmpT>
Matrix<CmpT> operator==(const CmpT& val_lhs, const Matrix<CmpT>& mat_rhs) {
  Matrix<CmpT> mat_eq(mat_rhs.shape()[0], mat_rhs.shape()[1]);
  internal::eq(val_lhs, mat_rhs, mat_eq);
  return mat_eq;
}
template <typename CmpT>
Matrix<CmpT> operator!=(const Matrix<CmpT>& mat_lhs,
                        const Matrix<CmpT>& mat_rhs) {
  Matrix<CmpT>::shape_consistence_check_(mat_lhs.shape(), mat_rhs.shape());
  Matrix<CmpT> mat_ne(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::ne(mat_lhs, mat_rhs, mat_ne);
  return mat_ne;
}
template <typename CmpT>
Matrix<CmpT> operator!=(const Matrix<CmpT>& mat_lhs, const CmpT& val_rhs) {
  Matrix<CmpT> mat_ne(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::ne(mat_lhs, val_rhs, mat_ne);
  return mat_ne;
}
template <typename CmpT>
Matrix<CmpT> operator!=(const CmpT& val_lhs, const Matrix<CmpT>& mat_rhs) {
  Matrix<CmpT> mat_ne(mat_rhs.shape()[0], mat_rhs.shape()[1]);
  internal::ne(val_lhs, mat_rhs, mat_ne);
  return mat_ne;
}
template <typename CmpT>
Matrix<CmpT> operator<(const Matrix<CmpT>& mat_lhs,
                        const Matrix<CmpT>& mat_rhs) {
  Matrix<CmpT>::shape_consistence_check_(mat_lhs.shape(), mat_rhs.shape());
  Matrix<CmpT> mat_lt(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::lt(mat_lhs, mat_rhs, mat_lt);
  return mat_lt;
}
template <typename CmpT>
Matrix<CmpT> operator<(const Matrix<CmpT>& mat_lhs, const CmpT& val_rhs) {
  Matrix<CmpT> mat_lt(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::lt(mat_lhs, val_rhs, mat_lt);
  return mat_lt;
}
template <typename CmpT>
Matrix<CmpT> operator<(const CmpT& val_lhs, const Matrix<CmpT>& mat_rhs) {
  Matrix<CmpT> mat_lt(mat_rhs.shape()[0], mat_rhs.shape()[1]);
  internal::lt(val_lhs, mat_rhs, mat_lt);
  return mat_lt;
}
template <typename CmpT>
Matrix<CmpT> operator<=(const Matrix<CmpT>& mat_lhs,
                        const Matrix<CmpT>& mat_rhs) {
  Matrix<CmpT>::shape_consistence_check_(mat_lhs.shape(), mat_rhs.shape());
  Matrix<CmpT> mat_le(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::le(mat_lhs, mat_rhs, mat_le);
  return mat_le;
}
template <typename CmpT>
Matrix<CmpT> operator<=(const Matrix<CmpT>& mat_lhs, const CmpT& val_rhs) {
  Matrix<CmpT> mat_le(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::le(mat_lhs, val_rhs, mat_le);
  return mat_le;
}
template <typename CmpT>
Matrix<CmpT> operator<=(const CmpT& val_lhs, const Matrix<CmpT>& mat_rhs) {
  Matrix<CmpT> mat_le(mat_rhs.shape()[0], mat_rhs.shape()[1]);
  internal::le(val_lhs, mat_rhs, mat_le);
  return mat_le;
}
template <typename CmpT>
Matrix<CmpT> operator>(const Matrix<CmpT>& mat_lhs,
                        const Matrix<CmpT>& mat_rhs) {
  Matrix<CmpT>::shape_consistence_check_(mat_lhs.shape(), mat_rhs.shape());
  Matrix<CmpT> mat_gt(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::gt(mat_lhs, mat_rhs, mat_gt);
  return mat_gt;
}
template <typename CmpT>
Matrix<CmpT> operator>(const Matrix<CmpT>& mat_lhs, const CmpT& val_rhs) {
  Matrix<CmpT> mat_gt(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::gt(mat_lhs, val_rhs, mat_gt);
  return mat_gt;
}
template <typename CmpT>
Matrix<CmpT> operator>(const CmpT& val_lhs, const Matrix<CmpT>& mat_rhs) {
  Matrix<CmpT> mat_gt(mat_rhs.shape()[0], mat_rhs.shape()[1]);
  internal::gt(val_lhs, mat_rhs, mat_gt);
  return mat_gt;
}
template <typename CmpT>
Matrix<CmpT> operator>=(const Matrix<CmpT>& mat_lhs,
                        const Matrix<CmpT>& mat_rhs) {
  Matrix<CmpT>::shape_consistence_check_(mat_lhs.shape(), mat_rhs.shape());
  Matrix<CmpT> mat_ge(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::ge(mat_lhs, mat_rhs, mat_ge);
  return mat_ge;
}
template <typename CmpT>
Matrix<CmpT> operator>=(const Matrix<CmpT>& mat_lhs, const CmpT& val_rhs) {
  Matrix<CmpT> mat_ge(mat_lhs.shape()[0], mat_lhs.shape()[1]);
  internal::ge(mat_lhs, val_rhs, mat_ge);
  return mat_ge;
}
template <typename CmpT>
Matrix<CmpT> operator>=(const CmpT& val_lhs, const Matrix<CmpT>& mat_rhs) {
  Matrix<CmpT> mat_ge(mat_rhs.shape()[0], mat_rhs.shape()[1]);
  internal::ge(val_lhs, mat_rhs, mat_ge);
  return mat_ge;
}
template <typename Tp>
bool Matrix<Tp>::equal(const Matrix& mat_rhs, std::size_t ulp) {
  if (shape()[0] != mat_rhs.shape()[0] || shape()[1] != mat_rhs.shape()[1]) {
    return false;
  }
  for (size_type idx_row = 0; idx_row < shape()[0]; ++idx_row) {
    if (mat_[idx_row].nequal(mat_rhs.mat_[idx_row])) {
      return false;
    }
  }
  return true;
}
template <typename Tp>
bool Matrix<Tp>::nequal(const Matrix& mat_rhs, std::size_t ulp) {
  return !equal(mat_rhs, ulp);
}
template <typename Tp>
Tp Matrix<Tp>::max() const {
  Tp max_element = mat_[0].max();
  for (size_type idx_row = 1; idx_row < shape()[0]; ++idx_row) {
    if (mat_[idx_row].max() > max_element) {
      max_element = mat_[idx_row].max();
    }
  }
  return max_element;
}
// namespace internal
namespace internal {

template <typename MaxT>
void max_of_dim_one(const Matrix<MaxT>& mat, Vector<MaxT>& max_vec) {
  for (typename Matrix<MaxT>::size_type idx_row = 0;
       idx_row < mat.shape()[0]; ++idx_row) {
    max_vec[idx_row] = mat[idx_row].max();
  }
}
template <typename MinT>
void min_of_dim_one(const Matrix<MinT>& mat, Vector<MinT>& min_vec) {
  for (typename Matrix<MinT>::size_type idx_row = 0;
       idx_row < mat.shape()[0]; ++idx_row) {
    min_vec[idx_row] = mat[idx_row].min();
  }
}

}  // namespace internal

template <typename Tp>
Vector<Tp> Matrix<Tp>::max(const dimension_type& dim_max) const {
  if (dim_max == 0) {
    Vector<Tp> max_vec(shape()[1]);
    internal::max_of_dim_one(transpose(), max_vec);
  } else if (dim_max == 1) {
    Vector<Tp> max_vec(shape()[0]);
    internal::max_of_dim_one(*this, max_vec);
    return max_vec;
  } else {
    std::string err_msg = "Invalid Dimension: " + std::to_string(dim_max) +
                          " != 0 or 1.";
    throw MatrixException(err_msg);
  }
}
template <typename Tp>
Tp Matrix<Tp>::min() const {
  Tp min_element = mat_[0].min();
  for (size_type idx_row = 1; idx_row < shape()[0]; ++idx_row) {
    if (mat_[idx_row].min() < min_element) {
      min_element = mat_[idx_row].min();
    }
  }
  return min_element;
}
template <typename Tp>
Vector<Tp> Matrix<Tp>::min(const dimension_type& dim_min) const {
  if (dim_min == 0) {
    Vector<Tp> min_vec(shape()[1]);
    internal::min_of_dim_one(transpose(), min_vec);
    return min_vec;
  } else if (dim_min == 1) {
    Vector<Tp> min_vec(shape()[0]);
    internal::min_of_dim_one(*this, min_vec);
    return min_vec;
  } else {
    std::string err_msg = "Invalid Dimension: " + std::to_string(dim_min) +
                          " != 0 or 1.";
    throw MatrixException(err_msg);
  }
}

// ======== IO ========
template <typename MatT, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(
  std::basic_ostream<CharT, Traits>& os, const Matrix<MatT>& mat) {
  if (mat.shape()[0] == 0) {
    os << "[[]]";
    return os;
  }
  if (mat.shape()[0] == 1) {
    os << "[" << mat.mat_[0] << "]";
    return os;
  }
  os << "[" << mat.mat_[0] << "\n";
  for (typename Matrix<MatT>::size_type idx_row = 1;
       idx_row < mat.shape()[0] - 1; ++idx_row) {
    os << " " << mat.mat_[idx_row] << "\n";
  }
  os << " " << mat.mat_[mat.shape()[0] - 1] << "]";
  return os;
}
template <typename MatT, typename CharT, typename Traits>
std::basic_istream<CharT, Traits>& operator>>(
  std::basic_istream<CharT, Traits>& is, Matrix<MatT>& mat) {
  for (typename Matrix<MatT>::size_type idx_row = 0;
       idx_row < mat.shape()[0]; ++idx_row) {
    is >> mat.mat_[idx_row];
  }
  return is;
}

// ======== Helper Functions ========
template <typename Tp>
typename Matrix<Tp>::index_type Matrix<Tp>::to_positive_index_(
  const size_type& size, const index_type& index) {
  return index >= 0 ? index : size + index;
}
template <typename Tp>
void Matrix<Tp>::exclusive_range_check_(const size_type& size, 
                                        const index_type& index) {
  size_type pos_index = to_positive_index_(size, index);
  if (pos_index >= size) {
    std::string err_msg = "Out-of-Range: row index " + 
      std::to_string(index) + " is out of range [0, " + 
      std::to_string(size) + ").";    
    throw MatrixException(err_msg);     
  }
}
template <typename Tp>
void Matrix<Tp>::exclusive_range_check_(const iterator& it_begin,
                                        const iterator& it_end,
                                        const iterator& it) {
  if (it < it_begin || it >= it_end) {
    std::string err_msg = 
      "Out-of-Range: row iterator is out of the range [begin(), end()).";
    throw MatrixException(err_msg);
  }
}
template <typename Tp>
void Matrix<Tp>::exclusive_range_check_(const const_iterator& cit_begin,
                                        const const_iterator& cit_end,
                                        const const_iterator& cit) {
  if (cit < cit_begin || cit >= cit_end) {
    std::string err_msg = 
      "Out-of-Range: row const_iterator is out of the range [begin(), end()).";
    throw MatrixException(err_msg);
  }
}
template <typename Tp>
void Matrix<Tp>::inclusive_range_check_(const size_type& size,
                                        const index_type& index) {
  size_type pos_index = to_positive_index_(size, index);
  if (pos_index > size) {
    std::string err_msg = "Out-of-Range: row index " + 
      std::to_string(index) + " is out of range [0, " + 
      std::to_string(size) + "].";    
    throw MatrixException(err_msg);     
  }
}
template <typename Tp>
void Matrix<Tp>::inclusive_range_check_(const iterator& it_begin,
                                        const iterator& it_end,
                                        const iterator& it) {
  if (it < it_begin || it > it_end) {
    std::string err_msg = 
      "Out-of-Range: row iterator is out of the range [begin(), end()].";
    throw MatrixException(err_msg);
  }
}
template <typename Tp>
void Matrix<Tp>::inclusive_range_check_(const const_iterator& cit_begin,
                                        const const_iterator& cit_end,
                                        const const_iterator& cit) {
  if (cit < cit_begin || cit > cit_end) {
    std::string err_msg = 
      "Out-of-Range: row const_iterator is out of the range [cbegin(), cend()].";
    throw MatrixException(err_msg);
  }
}
template <typename Tp>
void Matrix<Tp>::shape_consistence_check_(const Vector<size_type>& shape_lhs,
                                          const Vector<size_type>& shape_rhs) {
  if (shape_lhs[0] != shape_rhs[0] || shape_lhs[1] != shape_rhs[1]) {
    std::string err_msg = "Inconsistent shape: [" +
      std::to_string(shape_lhs[0]) + ", " +
      std::to_string(shape_lhs[1]) + "] != [" +
      std::to_string(shape_rhs[0]) + ", " +
      std::to_string(shape_rhs[1]) + "].";
    throw MatrixException(err_msg);
  }
}
template <typename Tp>
void Matrix<Tp>::index_order_check_(const size_type& size,
                                    const index_type& idx_begin,
                                    const index_type& idx_end) {
  if (to_positive_index_(size, idx_begin) >
      to_positive_index_(size, idx_end)) {
    std::string err_msg = "Invalid Row Index Order: begin " +
      std::to_string(to_positive_index_(size, idx_begin)) + " > end " +
      std::to_string(to_positive_index_(size, idx_end)) + ".";
    throw MatrixException(err_msg);
  }
}

template <typename Tp>
typename Matrix<Tp>::index_type Matrix<Tp>::to_positive_index_(
  const index_type& index) const {
  return to_positive_index_(mat_.size(), index);
}
template <typename Tp>
void Matrix<Tp>::exclusive_range_check_(const index_type& index) const {
  exclusive_range_check_(mat_.size(), index);
}
template <typename Tp>
void Matrix<Tp>::exclusive_range_check_(const iterator& it) {
  exclusive_range_check_(begin(), end(), it);
}
template <typename Tp>
void Matrix<Tp>::exclusive_range_check_(const const_iterator& cit) const {
  exclusive_range_check_(cbegin(), cend(), cit);
}
template <typename Tp>
void Matrix<Tp>::inclusive_range_check_(const index_type& index) const {
  inclusive_range_check_(mat_.size(), index);
}
template <typename Tp>
void Matrix<Tp>::inclusive_range_check_(const iterator& it) {
  inclusive_range_check_(begin(), end(), it);
}
template <typename Tp>
void Matrix<Tp>::inclusive_range_check_(const const_iterator& cit)  const {
  inclusive_range_check_(cbegin(), cend(), cit);
}
template <typename Tp>
void Matrix<Tp>::index_order_check_(const index_type& idx_begin, 
                                   const index_type& idx_end) const {
  index_order_check_(mat_.size(), idx_begin, idx_end);
}
template <typename Tp>
void Matrix<Tp>::iterator_order_check_(const iterator& it_begin,
                                      const iterator& it_end) {
  index_order_check_(mat_.size(), 
    static_cast<index_type>(it_begin - mat_.begin()),
    static_cast<index_type>(it_end - mat_.begin()));
}
template <typename Tp>
void Matrix<Tp>::const_iterator_order_check_(
  const const_iterator& cit_begin, const const_iterator& cit_end) const {
  index_order_check_(mat_.size(),
    static_cast<index_type>(cit_begin - mat_.cbegin()),
    static_cast<index_type>(cit_end - mat_.cbegin()));
}
// ======== ENd of class Vector ========


class MatrixException {
public:
  MatrixException() noexcept {};
  MatrixException(const MatrixException& other) noexcept : msg_(other.msg_) {}
  explicit MatrixException(const std::string& message) noexcept : 
    msg_(message) {}
  explicit MatrixException(const char* message) noexcept : msg_(message) {}
  MatrixException& operator=(const MatrixException& other) noexcept {
    msg_ = other.msg_;
    return *this;
  }
  MatrixException& operator=(const std::string& msg_copy) noexcept { 
    msg_ = msg_copy;
    return *this;
  }
  MatrixException& operator=(const char* msg_copy) noexcept { 
    msg_ = msg_copy;
    return *this;
  }
  ~MatrixException() noexcept {};
  const char* what() const noexcept { return msg_.c_str(); }

protected:
  std::string msg_;  
};

}  // namespace tensor

#endif  // TENSOR_MATRIX_H_