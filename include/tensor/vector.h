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

- ======== tensor::Vector ========

- namespace tensor

  - struct Random

  - class Vector
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
        - random constructors
        - arithmetic
        - comparisons

  - class VectorException

*/

#ifndef TENSOR_VECTOR_H_
#define TENSOR_VECTOR_H_

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>
#include <x86intrin.h>


namespace tensor {

struct Random {
  enum Generator {
    default_random_engine, minstd_rand0, minstd_rand, mt19937, mt19937_64, 
    ranlux24_base, ranlux48_base, ranlux24, ranlux48, knuth_b    
  };
  enum Distribution {
    uniform_int, uniform_real, 
    bernoulli, binomial, negative_binomial, geometric,
    poisson, exponential, gamma, weibull, extreme_value,
    normal, lognormal, chi_squared, cauchy, fisher_f, student_t
  };
};

class VectorException;


template <typename T>
class Vector {

public:
  // ======== Types ========
  typedef T value_type;
  typedef typename std::vector<T>::size_type size_type;
  typedef typename std::make_signed<size_type>::type index_type;
  typedef typename std::vector<T>::difference_type difference_type;
  typedef typename std::vector<T>::iterator iterator;
  typedef typename std::vector<T>::const_iterator const_iterator;
  typedef typename std::vector<T>::reverse_iterator reverse_iterator;
  typedef typename std::vector<T>::const_reverse_iterator
    const const_reverse_iterator;

  // ======== Constructors, etc ========
  Vector();

  explicit Vector(const size_type& size_init);
  Vector(const size_type& size_init, const T& val_init);
  template <typename OtherT>
  Vector(const size_type& size_init, const OtherT& val_cast);

  Vector(const Vector& vec_init);
  template <typename OtherT>
  Vector(const Vector<OtherT>& vec_cast);

  Vector(Vector&& vec_init);
  template <typename OtherT>
  Vector(Vector<OtherT>&& vec_cast);

  /*explicit */Vector(const std::vector<T>& stdvec_init);
  template <typename OtherT>
  /*explicit */Vector(const std::vector<OtherT>& stdvec_cast);

  /*explicit */Vector(const std::initializer_list<T>& il_init);
  template <typename OtherT>
  /*explicit */Vector(const std::initializer_list<OtherT>& il_cast);

  /* TODO */
  template <typename ParamT1, typename ParamT2>
  Vector(const size_type& size_init, Random::Generator gen, 
         Random::Distribution dis, const ParamT1& param1, 
         const ParamT2& param2);
  /* TODO */
  template <typename ParamT>
  Vector(const size_type& size_init, Random::Generator gen, 
         Random::Distribution dis, const ParamT& param);
  template <typename ParamT1, typename ParamT2>
  Vector(const size_type& size_init, Random::Distribution dis,
         const ParamT1& param1, const ParamT2& param2);
  template <typename ParamT>
  Vector(const size_type& size_init, Random::Distribution dis, 
         const ParamT& param);

  Vector& operator=(const T& val_assign);
  template <typename OtherT>
  Vector& operator=(const OtherT& val_cast);

  Vector& operator=(const Vector& vec_copy);
  template <typename OtherT>
  Vector& operator=(const Vector<OtherT>& vec_cast);

  Vector& operator=(Vector&& vec_move);
  template <typename OtherT>
  Vector& operator=(Vector<OtherT>&& vec_cast);

  Vector& operator=(const std::vector<T>& stdvec_assign);
  template <typename OtherT>
  Vector& operator=(const std::vector<OtherT>& stdvec_cast);

  Vector& operator=(const std::initializer_list<T>& il_assign);
  template <typename OtherT>
  Vector& operator=(const std::initializer_list<OtherT>& il_cast);

  ~Vector();

  // ======== Shape ========
  Vector<size_type> shape() const;
  void clear();
  bool empty() const;

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
  T& operator[](const index_type& index);
  const T& operator[](const index_type& index) const;
  Vector operator()(const index_type& index) const;
  Vector operator()(const const_iterator& cit) const;
  Vector operator()(const index_type& idx_begin, 
                    const index_type& idx_end) const;
  Vector operator()(const const_iterator& cit_begin, 
                    const const_iterator& cit_end) const;

  // ======== Modifiers ========
  Vector insert(const T& val_insert, const index_type& index) const;
  Vector insert(const T& val_insert, const const_iterator& cit) const;
  Vector insert(const Vector& vec_insert, const index_type& index) const;
  Vector insert(const Vector& vec_insert, const const_iterator& cit) const;
  Vector remove(const index_type& index) const;
  Vector remove(const const_iterator& cit) const;
  Vector remove(const index_type& idx_begin, const index_type& idx_end) const;
  Vector remove(const const_iterator& cit_begin, 
                const const_iterator& cit_end) const;
  Vector replace(const T& val_replace, const index_type& index) const;
  Vector replace(const T& val_replace, const const_iterator& cit) const;
  Vector replace(const Vector& vec_replace, const index_type& index) const;
  Vector replace(const Vector& vec_replace, const const_iterator& cit) const;
  Vector reshape(const size_type& size) const;
  Vector reverse() const;
  Vector shuffle() const;

  // ======== Arithmetic ========
  template <typename AriT> 
  friend Vector<AriT> operator+(const Vector<AriT>& vec_lhs, 
                                const Vector<AriT>& vec_rhs);
  template <typename AriT> 
  friend Vector<AriT> operator+(const Vector<AriT>& vec_lhs, 
                                const AriT& val_rhs);
  template <typename AriT> 
  friend Vector<AriT> operator+(const AriT& val_lhs, 
                                const Vector<AriT>& vec_rhs);
  template <typename AriT> 
  friend Vector<AriT> operator-(const Vector<AriT>& vec_lhs, 
                                const Vector<AriT>& vec_rhs);
  template <typename AriT> 
  friend Vector<AriT> operator-(const Vector<AriT>& vec_lhs, 
                                const AriT& val_rhs);
  template <typename AriT> 
  friend Vector<AriT> operator-(const AriT& val_lhs, 
                                const Vector<AriT>& vec_rhs);
  template <typename AriT> 
  friend Vector<AriT> operator*(const Vector<AriT>& vec_lhs, 
                                const Vector<AriT>& vec_rhs);
  template <typename AriT> 
  friend Vector<AriT> operator*(const Vector<AriT>& vec_lhs, 
                                const AriT& val_rhs);
  template <typename AriT> 
  friend Vector<AriT> operator*(const AriT& val_lhs, 
                                const Vector<AriT>& vec_rhs);
  template <typename AriT> 
  friend Vector<AriT> operator/(const Vector<AriT>& vec_lhs, 
                                const Vector<AriT>& vec_rhs);
  template <typename AriT> 
  friend Vector<AriT> operator/(const Vector<AriT>& vec_lhs, 
                                const AriT& val_rhs);
  template <typename AriT> 
  friend Vector<AriT> operator/(const AriT& val_lhs, 
                                const Vector<AriT>& vec_rhs);
  void operator+=(const Vector& vec_rhs);
  void operator+=(const T& val_rhs);
  void operator-=(const Vector& vec_rhs);
  void operator-=(const T& val_rhs);
  void operator*=(const Vector& vec_rhs);
  void operator*=(const T& val_rhs);
  void operator/=(const Vector& vec_rhs);
  void operator/=(const T& val_rhs);
  T sum() const;

  // ======== Comparisons ========
  template <typename CmpT> 
  friend Vector<CmpT> operator==(const Vector<CmpT>& vec_lhs, 
                                const Vector<CmpT>& vec_rhs);
  template <typename CmpT> 
  friend Vector<CmpT> operator==(const Vector<CmpT>& vec_lhs, 
                                 const CmpT& val_rhs);
  template <typename CmpT> 
  friend Vector<CmpT> operator==(const CmpT& val_lhs, 
                                 const Vector<CmpT>& vec_rhs);
  template <typename CmpT> 
  friend Vector<CmpT> operator!=(const Vector<CmpT>& vec_lhs, 
                         const Vector<CmpT>& vec_rhs);  
  template <typename CmpT>
  friend Vector<CmpT> operator!=(const Vector<CmpT>& vec_lhs, 
                                 const CmpT& val_rhs);
  template <typename CmpT> 
  friend Vector<CmpT> operator!=(const CmpT& val_lhs, 
                                 const Vector<CmpT>& vec_rhs);
  template <typename CmpT>
  friend Vector<CmpT> operator<(const Vector<CmpT>& vec_lhs,
                                 const Vector<CmpT>& vec_rhs);
  template <typename CmpT> 
  friend Vector<CmpT> operator<(const Vector<CmpT>& vec_lhs, 
                                const CmpT& val_rhs);
  template <typename CmpT> 
  friend Vector<CmpT> operator<(const CmpT& val_lhs, 
                                const Vector<CmpT>& vec_rhs);
  template <typename CmpT>
  friend Vector<CmpT> operator<=(const Vector<CmpT>& vec_lhs,
                                 const Vector<CmpT>& vec_rhs);
  template <typename CmpT> 
  friend Vector<CmpT> operator<=(const Vector<CmpT>& vec_lhs, 
                                 const CmpT& val_rhs);
  template <typename CmpT> 
  friend Vector<CmpT> operator<=(const CmpT& val_lhs, 
                                 const Vector<CmpT>& vec_rhs);
  template <typename CmpT>
  friend Vector<CmpT> operator>(const Vector<CmpT>& vec_lhs,
                                const Vector<CmpT>& vec_rhs);
  template <typename CmpT> 
  friend Vector<CmpT> operator>(const Vector<CmpT>& vec_lhs, 
                                const CmpT& val_rhs);
  template <typename CmpT> 
  friend Vector<CmpT> operator>(const CmpT& val_lhs, 
                                const Vector<CmpT>& vec_rhs);
  template <typename CmpT>
  friend Vector<CmpT> operator>=(const Vector<CmpT>& vec_lhs,
                                 const Vector<CmpT>& vec_rhs);
  template <typename CmpT> 
  friend Vector<CmpT> operator>=(const Vector<CmpT>& vec_lhs, 
                                 const CmpT& val_rhs);
  template <typename CmpT> 
  friend Vector<CmpT> operator>=(const CmpT& val_lhs, 
                                 const Vector<CmpT>& vec_rhs);
  bool equal(const Vector& vec_rhs, std::size_t ulp = 1);
  bool nequal(const Vector& vec_rhs, std::size_t ulp = 1);
  T max() const;
  T min() const;

  // ======== IO ========
  template <typename VecT, typename CharT, typename Traits>
  friend std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os, const Vector<VecT>& vec);
  template <typename VecT, typename CharT, typename Traits>
  friend std::basic_istream<CharT, Traits>& operator>>(
    std::basic_istream<CharT, Traits>& is, Vector<VecT>& vec);

private:
  // ======== Helper Functions ========
  static bool fequal_(const T& x, const T& y, std::size_t ulp = 1);
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
  std::vector<T> vec_;
};


// ======== Constructors, etc ========
template <typename T>
Vector<T>::Vector() {}

template <typename T>
Vector<T>::Vector(const size_type& size_init) :
  vec_(std::vector<T>(size_init, T())) {}

template <typename T>
Vector<T>::Vector(const size_type& size_init, const T& val_init) :
  vec_(std::vector<T>(size_init, val_init)) {}
template <typename T>  template <typename OtherT>
Vector<T>::Vector(const size_type& size_init, const OtherT& val_cast) :
  vec_(std::vector<T>(size_init, static_cast<T>(val_cast))) {}

template <typename T>
Vector<T>::Vector(const Vector& vec_init) : vec_(vec_init.vec_) {}
template <typename T>  template <typename OtherT>
Vector<T>::Vector(const Vector<OtherT>& vec_cast) {
  vec_.resize(vec_cast.shape()[0]);
  for (size_type idx = 0; idx < vec_.size(); ++idx) {
    vec_[idx] = static_cast<T>(vec_cast[idx]);
  }
}

template <typename T>
Vector<T>::Vector(Vector&& vec_init) : vec_(std::move(vec_init.vec_)) {}
template <typename T>  template <typename OtherT>
Vector<T>::Vector(Vector<OtherT>&& vec_cast) {
  vec_.resize(vec_cast.shape()[0]);
  Vector<OtherT> vec_cache = std::move(vec_cast);
  for (size_type idx = 0; idx < vec_.size(); ++idx) {
    vec_[idx] = static_cast<T>(vec_cache[idx]);
  }
}

template <typename T>
Vector<T>::Vector(const std::vector<T>& stdvec_init) : vec_(stdvec_init) {}
template <typename T>  template <typename OtherT>
Vector<T>::Vector(const std::vector<OtherT>& stdvec_cast) {
  vec_.resize(stdvec_cast.size());
  for (size_type idx = 0; idx < vec_.size(); ++idx) {
    vec_[idx] = static_cast<T>(stdvec_cast[idx]);
  }
}

template <typename T>
Vector<T>::Vector(const std::initializer_list<T>& il_init) : vec_(il_init) {}
template <typename T>  template <typename OtherT>
Vector<T>::Vector(const std::initializer_list<OtherT>& il_cast) {
  vec_.resize(il_cast.size());
  for (size_type idx = 0; idx < vec_.size(); ++idx) {
    vec_[idx] = static_cast<T>(*(il_cast.begin() + idx));
  }
}

// namespace internal
namespace internal {

struct RandomNumberGenerator {
#define RNG(NAME) \
static std::NAME NAME(std::NAME::result_type seed) { \
  return std::NAME(seed); \
}
RNG(default_random_engine)
RNG(minstd_rand0)
RNG(minstd_rand)
RNG(mt19937)
RNG(mt19937_64)
RNG(ranlux24_base)
RNG(ranlux48_base)
RNG(ranlux24)
RNG(ranlux48)
RNG(knuth_b)
#undef RNG
};

template <typename T>
struct RandomDistribution {

#define DIS_2_PARAM_INTEGRAL(NAME) \
template <typename Tp = T, typename ParamT1, typename ParamT2> \
static std::NAME##_distribution< typename \
std::enable_if<std::is_integral<Tp>::value, Tp>::type> \
NAME(const ParamT1& param1, const ParamT2& param2) { \
  return std::NAME##_distribution<Tp>(param1, param2); \
} \
template <typename Tp = T, typename ParamT1, typename ParamT2> \
static std::NAME##_distribution< typename \
std::enable_if<!std::is_integral<Tp>::value, int>::type> \
NAME(const ParamT1& param1, const ParamT2& param2) { \
  return std::NAME##_distribution<int>(param1, param2); \
}

#define DIS_2_PARAM_FLOATING(NAME) \
template <typename Tp = T, typename ParamT1, typename ParamT2> \
static std::NAME##_distribution< typename \
std::enable_if<std::is_floating_point<Tp>::value, Tp>::type> \
NAME(const ParamT1& param1, const ParamT2& param2) { \
  return std::NAME##_distribution<Tp>(param1, param2); \
} \
template <typename Tp = T, typename ParamT1, typename ParamT2> \
static std::NAME##_distribution< typename \
std::enable_if<!std::is_floating_point<Tp>::value, double>::type> \
NAME(const ParamT1& param1, const ParamT2& param2) { \
  return std::NAME##_distribution<double>(param1, param2); \
}

#define DIS_1_PARAM_INTEGRAL(NAME) \
template <typename Tp = T, typename ParamT> \
static std::NAME##_distribution< typename \
std::enable_if<std::is_integral<Tp>::value, Tp>::type> \
NAME(const ParamT& param) { \
  return std::NAME##_distribution<Tp>(param); \
} \
template <typename Tp = T, typename ParamT> \
static std::NAME##_distribution< typename \
std::enable_if<!std::is_integral<Tp>::value, int>::type> \
NAME(const ParamT& param) { \
  return std::NAME##_distribution<int>(param); \
}

#define DIS_1_PARAM_FLOATING(NAME) \
template <typename Tp = T, typename ParamT> \
static std::NAME##_distribution< typename \
std::enable_if<std::is_floating_point<Tp>::value, Tp>::type> \
NAME(const ParamT& param) { \
  return std::NAME##_distribution<Tp>(param); \
} \
template <typename Tp = T, typename ParamT> \
static std::NAME##_distribution< typename \
std::enable_if<!std::is_floating_point<Tp>::value, double>::type> \
NAME(const ParamT& param) { \
  return std::NAME##_distribution<double>(param); \
}

DIS_2_PARAM_INTEGRAL(uniform_int)
DIS_2_PARAM_FLOATING(uniform_real)

DIS_2_PARAM_INTEGRAL(binomial)
DIS_2_PARAM_INTEGRAL(negative_binomial)
DIS_1_PARAM_INTEGRAL(geometric)

DIS_1_PARAM_INTEGRAL(poisson)
DIS_1_PARAM_FLOATING(exponential)
DIS_2_PARAM_FLOATING(gamma)
DIS_2_PARAM_FLOATING(weibull)
DIS_2_PARAM_FLOATING(extreme_value)

DIS_2_PARAM_FLOATING(normal)
DIS_2_PARAM_FLOATING(lognormal)
DIS_1_PARAM_FLOATING(chi_squared)
DIS_2_PARAM_FLOATING(cauchy)
DIS_2_PARAM_FLOATING(fisher_f)
DIS_1_PARAM_FLOATING(student_t)

#undef DIS_2_PARAM_INTEGRAL
#undef DIS_2_PARAM_FLOATING
#undef DIS_1_PARAM_INTEGRAL
#undef DIS_1_PARAM_FLOATING

template <typename ParamT>
static std::bernoulli_distribution bernoulli(const ParamT& param) {
  return std::bernoulli_distribution(param);
}
};

}  // namespace internal

// Random Constructor
template <typename T>  template <typename ParamT1, typename ParamT2>
Vector<T>::Vector(const size_type& size_init, Random::Generator gen, 
                  Random::Distribution dis, const ParamT1& param1, 
                  const ParamT2& param2) {
  /* TODO */
}
template <typename T>  template <typename ParamT>
Vector<T>::Vector(const size_type& size_init, Random::Generator gen, 
                  Random::Distribution dis, const ParamT& param) {
  /* TODO */
}
template <typename T>  template <typename ParamT1, typename ParamT2>
Vector<T>::Vector(const size_type& size_init, Random::Distribution dis,
                  const ParamT1& param1, const ParamT2& param2) {
  vec_.resize(size_init);
  auto gen = internal::RandomNumberGenerator::default_random_engine(
             std::random_device()());
  switch (dis) {
    case Random::Distribution::uniform_int: {
      if(!std::is_integral<T>::value) {
        throw VectorException(
          "Result type of uniform_int should be a integral type.");
      }
      auto dis = internal::RandomDistribution<T>::uniform_int(param1, param2);
      for (T& element : vec_) element = dis(gen);  break; }
    case Random::Distribution::uniform_real: {
      if(!std::is_floating_point<T>::value) {
        throw VectorException(
          "Result type of uniform_real should be a floating point type.");
      }
      auto dis = internal::RandomDistribution<T>::uniform_real(param1, param2);
      for (T& element : vec_) element = dis(gen);  break; }
    case Random::Distribution::binomial: {
      if(!std::is_integral<T>::value) {
        throw VectorException(
          "Result type of binomial should be a integral type.");
      }
      auto dis = internal::RandomDistribution<T>::binomial(param1, param2);
      for (T& element : vec_) element = dis(gen);  break; }
    case Random::Distribution::negative_binomial: {
      if(!std::is_integral<T>::value) {
        throw VectorException(
          "Result type of negative_binomial should be a integral type.");
      }
      auto dis = internal::RandomDistribution<T>::negative_binomial(
        param1, param2);
      for (T& element : vec_) element = dis(gen);  break; }
    case Random::Distribution::gamma: {
      if(!std::is_floating_point<T>::value) {
        throw VectorException(
          "Result type of gamma should be a floating point type.");
      }
      auto dis = internal::RandomDistribution<T>::gamma(param1, param2);
      for (T& element : vec_) element = dis(gen);  break; }
    case Random::Distribution::weibull: {
      if(!std::is_floating_point<T>::value) {
        throw VectorException(
          "Result type of weibull should be a floating point type.");
      }
      auto dis = internal::RandomDistribution<T>::weibull(param1, param2);
      for (T& element : vec_) element = dis(gen);  break; }
    case Random::Distribution::extreme_value: {
      if(!std::is_floating_point<T>::value) {
        throw VectorException(
          "Result type of extreme_value should be a floating point type.");
      }
      auto dis = internal::RandomDistribution<T>::extreme_value(
        param1, param2);
      for (T& element : vec_) element = dis(gen);  break; }
    case Random::Distribution::normal: {
      if(!std::is_floating_point<T>::value) {
        throw VectorException(
          "Result type of normal should be a floating point type.");
      }
      auto dis = internal::RandomDistribution<T>::normal(param1, param2);
      for (T& element : vec_) element = dis(gen);  break; }
    case Random::Distribution::lognormal: {
      if(!std::is_floating_point<T>::value) {
        throw VectorException(
          "Result type of lognormal should be a floating point type.");
      }
      auto dis = internal::RandomDistribution<T>::lognormal(param1, param2);
      for (T& element : vec_) element = dis(gen);  break; }
    case Random::Distribution::cauchy: {
      if(!std::is_floating_point<T>::value) {
        throw VectorException(
          "Result type of cauchy should be a floating point type.");
      }
      auto dis = internal::RandomDistribution<T>::cauchy(param1, param2);
      for (T& element : vec_) element = dis(gen);  break; }
    case Random::Distribution::fisher_f: {
      if(!std::is_floating_point<T>::value) {
        throw VectorException(
          "Result type of fisher_f should be a floating point type.");
      }
      auto dis = internal::RandomDistribution<T>::fisher_f(param1, param2);
      for (T& element : vec_) element = dis(gen);  break; }
    default:
      throw VectorException("Unsupported random distribution type.");
  }
}
template <typename T>  template <typename ParamT>
Vector<T>::Vector(const size_type& size_init, Random::Distribution dis, 
                  const ParamT& param) {
  vec_.resize(size_init);
  auto gen = internal::RandomNumberGenerator::default_random_engine(
             std::random_device()());
  switch (dis) {
    case Random::Distribution::bernoulli: {
      auto dis = internal::RandomDistribution<T>::bernoulli(param);
      for (T& element : vec_) element = static_cast<T>(dis(gen));  break; }
    case Random::Distribution::geometric: {
      if(!std::is_integral<T>::value) {
        throw VectorException(
          "Result type of geometric should be a integral type."); 
      }
      auto dis = internal::RandomDistribution<T>::geometric(param);
      for (T& element : vec_) element = dis(gen);  break; }
    case Random::Distribution::poisson: {
      if(!std::is_integral<T>::value) {
        throw VectorException(
          "Result type of poisson should be a integral type."); 
      }
      auto dis = internal::RandomDistribution<T>::poisson(param);
      for (T& element : vec_) element = dis(gen);  break; }
    case Random::Distribution::exponential: {
      if(!std::is_floating_point<T>::value) {
        throw VectorException(
          "Result type of exponential should be a floating point type."); 
      }
      auto dis = internal::RandomDistribution<T>::exponential(param);
      for (T& element : vec_) element = dis(gen);  break; }
    case Random::Distribution::chi_squared: {
      if(!std::is_floating_point<T>::value) {
        throw VectorException(
          "Result type of chi_squared should be a floating point type."); 
      }
      auto dis = internal::RandomDistribution<T>::chi_squared(param);
      for (T& element : vec_) element = dis(gen);  break; }
    case Random::Distribution::student_t: {
      if(!std::is_floating_point<T>::value) {
        throw VectorException(
          "Result type of student_t should be a floating point type."); 
      }
      auto dis = internal::RandomDistribution<T>::student_t(param);
      for (T& element : vec_) element = dis(gen);  break; }
    default:
      throw VectorException("Unsupported random distribution type.");
  }
}

template <typename T>
Vector<T>& Vector<T>::operator=(const T& val_assign) {
  vec_ = std::vector<T>(vec_.size(), val_assign);
  return *this;
}
template <typename T>  template <typename OtherT>
Vector<T>& Vector<T>::operator=(const OtherT& val_cast) {
  T value = static_cast<T>(val_cast);
  vec_ = std::vector<T>(vec_.size(), value);
  return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator=(const Vector& vec_copy) {
  vec_ = vec_copy.vec_;
  return *this;
}
template <typename T>  template <typename OtherT>
Vector<T>& Vector<T>::operator=(const Vector<OtherT>& vec_cast) {
  vec_ = Vector<T>(vec_cast).vec_;
  return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator=(Vector&& vec_move) {
  vec_ = std::move(vec_move.vec_);
  return *this;
}
template <typename T>  template <typename OtherT>
Vector<T>& Vector<T>::operator=(Vector<OtherT>&& vec_cast) {
  vec_ = Vector<T>(vec_cast).vec_;
  return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator=(const std::vector<T>& stdvec_assign) {
  vec_ = stdvec_assign;
  return *this;
}
template <typename T>  template <typename OtherT>
Vector<T>& Vector<T>::operator=(const std::vector<OtherT>& stdvec_cast) {
  vec_ = Vector<T>(stdvec_cast).vec_;
  return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator=(const std::initializer_list<T>& il_assign) {
  vec_ = il_assign;
  return *this;
}
template <typename T>  template <typename OtherT>
Vector<T>& Vector<T>::operator=(const std::initializer_list<OtherT>& il_cast) {
  vec_ = Vector<T>(il_cast).vec_;
  return *this;
}

template <typename T>
Vector<T>::~Vector() {}


// ======== Shape ========
template <typename T>
Vector<typename Vector<T>::size_type> Vector<T>::shape() const {
  return Vector<size_type>(1, vec_.size());
}
template <typename T>
void Vector<T>::clear() {
  vec_.clear();
}
template <typename T>
bool Vector<T>::empty() const {
  return vec_.size() == 0;
}

// ======== Iterators ========
template <typename T> typename
Vector<T>::iterator Vector<T>::begin() { return vec_.begin(); }
template <typename T> typename
Vector<T>::iterator Vector<T>::end() { return vec_.end(); }
template <typename T> typename
Vector<T>::const_iterator Vector<T>::begin() const { return vec_.cbegin(); }
template <typename T> typename
Vector<T>::const_iterator Vector<T>::end() const { return vec_.cend(); }
template <typename T> typename
Vector<T>::const_iterator Vector<T>::cbegin() const { return vec_.cbegin(); }
template <typename T> typename
Vector<T>::const_iterator Vector<T>::cend() const { return vec_.cend(); }
template <typename T> typename
Vector<T>::reverse_iterator Vector<T>::rbegin() { return vec_.rbegin(); }
template <typename T> typename
Vector<T>::reverse_iterator Vector<T>::rend() { return vec_.rend(); }
template <typename T> typename
Vector<T>::const_reverse_iterator Vector<T>::rbegin() const {
  return vec_.crbegin();
}
template <typename T> typename
Vector<T>::const_reverse_iterator Vector<T>::rend() const {
  return vec_.crend();
}
template <typename T> typename
Vector<T>::const_reverse_iterator Vector<T>::crbegin() const {
  return vec_.crbegin();
}
template <typename T> typename
Vector<T>::const_reverse_iterator Vector<T>::crend() const {
  return vec_.crend();
}

// ======== Accessors ========
template <typename T>
T& Vector<T>::operator[](const index_type& index) {
  exclusive_range_check_(index);
  return vec_.at(to_positive_index_(index));
}
template <typename T>
const T& Vector<T>::operator[](const index_type& index) const {
  exclusive_range_check_(index);
  return vec_.at(to_positive_index_(index));
}
template <typename T>
Vector<T> Vector<T>::operator()(const index_type& index) const {
  exclusive_range_check_(index);
  return Vector(1, vec_.at(to_positive_index_(index)));
}
template <typename T>
Vector<T> Vector<T>::operator()(const const_iterator& cit) const {
  exclusive_range_check_(cit);
  return Vector(1, *cit);
}
template <typename T>
Vector<T> Vector<T>::operator()(const index_type& idx_begin,
                                const index_type& idx_end) const {
  exclusive_range_check_(idx_begin);
  inclusive_range_check_(idx_end);
  index_order_check_(idx_begin, idx_end);
  return Vector(std::vector<T>(vec_.begin() + to_positive_index_(idx_begin),
                               vec_.begin() + to_positive_index_(idx_end)));
}
template <typename T>
Vector<T> Vector<T>::operator()(const const_iterator& cit_begin,
                                const const_iterator& cit_end) const {
  exclusive_range_check_(cit_begin);
  inclusive_range_check_(cit_end);
  const_iterator_order_check_(cit_begin, cit_end);
  return Vector(std::vector<T>(cit_begin, cit_end));
}

// ======== Modifiers ========
template <typename T>
Vector<T> Vector<T>::insert(const T& val_insert, 
                            const index_type& index) const {
  inclusive_range_check_(index);
  Vector vec_inserted = *this;
  vec_inserted.vec_.insert(
    vec_inserted.vec_.begin() + to_positive_index_(index), val_insert);
  return vec_inserted;
}
template <typename T>
Vector<T> Vector<T>::insert(const T& val_insert,
                            const const_iterator& cit) const {
  inclusive_range_check_(cit);
  return insert(val_insert, static_cast<index_type>(cit - vec_.cbegin()));
}
template <typename T>
Vector<T> Vector<T>::insert(const Vector& vec_insert,
                            const index_type& index) const {
  inclusive_range_check_(index);
  Vector vec_inserted = *this;
  vec_inserted.vec_.insert(
    vec_inserted.vec_.begin() + to_positive_index_(index),
    vec_insert.vec_.begin(), vec_insert.vec_.end());
  return vec_inserted;
}
template <typename T>
Vector<T> Vector<T>::insert(const Vector& vec_insert,
                            const const_iterator& cit) const {
  inclusive_range_check_(cit);
  return insert(vec_insert, static_cast<index_type>(cit - vec_.cbegin()));
}
template <typename T>
Vector<T> Vector<T>::remove(const index_type& index) const {
  exclusive_range_check_(index);
  Vector vec_removed = *this;
  vec_removed.vec_.erase(vec_removed.vec_.begin() + to_positive_index_(index));
  return vec_removed;
}
template <typename T>
Vector<T> Vector<T>::remove(const const_iterator& cit) const {
  exclusive_range_check_(cit);
  return remove(static_cast<size_type>(cit - vec_.cbegin()));
}
template <typename T>
Vector<T> Vector<T>::remove(const index_type& idx_begin, 
                            const index_type& idx_end) const {
  exclusive_range_check_(idx_begin);
  inclusive_range_check_(idx_end);
  index_order_check_(idx_begin, idx_end);
  Vector vec_removed = *this;
  vec_removed.vec_.erase(
    vec_removed.vec_.begin() + to_positive_index_(idx_begin),
    vec_removed.vec_.begin() + to_positive_index_(idx_end));
  return vec_removed;
}
template <typename T>
Vector<T> Vector<T>::remove(const const_iterator& cit_begin,
                            const const_iterator& cit_end) const {
  exclusive_range_check_(cit_begin);
  inclusive_range_check_(cit_end);
  const_iterator_order_check_(cit_begin, cit_end);
  return remove(static_cast<index_type>(cit_begin - vec_.cbegin()),
                static_cast<index_type>(cit_end - vec_.cbegin()));
}
template <typename T>
Vector<T> Vector<T>::replace(const T& val_replace, 
                             const index_type& index) const {
  exclusive_range_check_(index);
  Vector vec_replaced = *this;
  vec_replaced.vec_.at(to_positive_index_(index)) = val_replace;
  return vec_replaced;
}
template <typename T>
Vector<T> Vector<T>::replace(const T& val_replace,
                             const const_iterator& cit) const {
  exclusive_range_check_(cit);
  return replace(val_replace, static_cast<index_type>(cit - vec_.cbegin()));
}
template <typename T>
Vector<T> Vector<T>::replace(const Vector& vec_replace,
                             const index_type& index) const {
  exclusive_range_check_(index);
  index_type pos_index = to_positive_index_(index);
  Vector vec_replaced = *this;
  for (index_type idx_rep = 0; idx_rep < vec_replace.vec_.size() &&
       pos_index + idx_rep < vec_replaced.vec_.size(); ++idx_rep) {
    vec_replaced.vec_.at(pos_index + idx_rep) = vec_replace.vec_.at(idx_rep);
  }
  return vec_replaced;
}
template <typename T>
Vector<T> Vector<T>::replace(const Vector& vec_replace,
                             const const_iterator& cit) const {
  exclusive_range_check_(cit);
  return replace(vec_replace, static_cast<index_type>(cit - vec_.cbegin()));
}
template <typename T>
Vector<T> Vector<T>::reshape(const size_type& size) const {
  std::vector<T> vec_cache = vec_;
  vec_cache.resize(size);
  return Vector(vec_cache);
}
template <typename T>
Vector<T> Vector<T>::reverse() const {
  return Vector(std::vector<T>(vec_.crbegin(), vec_.crend()));
}
template <typename T>
Vector<T> Vector<T>::shuffle() const {
  std::random_device rd;
  std::default_random_engine gen(rd());
  Vector vec_shuffled = *this;
  std::shuffle(vec_shuffled.vec_.begin(), vec_shuffled.vec_.end(), gen);
  return vec_shuffled;
}

// ======== Arithmetic ========
// namespace internal
namespace internal {

#define OMP_FOR_3_PTR \
_Pragma("omp parallel for shared(ptr_lhs, ptr_rhs, ptr_ans) schedule(auto)")

#define OMP_FOR_2_PTR_L_ANS \
_Pragma("omp parallel for shared(ptr_lhs, rhs, ptr_ans) schedule(auto)")

#define OMP_FOR_2_PTR_R_ANS \
_Pragma("omp parallel for shared(lhs, ptr_rhs, ptr_ans) schedule(auto)")

#define ARITHMETIC_VEC_VEC(OPERATION, OPERATOR) \
template <typename T> \
void OPERATION(const Vector<T>& lhs, const Vector<T>& rhs, \
               Vector<T>& ans) { \
  const typename Vector<T>::size_type size = ans.shape()[0]; \
  const T* ptr_lhs = &lhs[0]; \
  const T* ptr_rhs = &rhs[0]; \
  T* ptr_ans = &ans[0]; \
  for (typename Vector<T>::size_type idx = 0; idx < size; ++idx) { \
    ptr_ans[idx] = ptr_lhs[idx] OPERATOR ptr_rhs[idx]; \
  } \
}
ARITHMETIC_VEC_VEC(add, +)
ARITHMETIC_VEC_VEC(sub, -)
ARITHMETIC_VEC_VEC(mul, *)
ARITHMETIC_VEC_VEC(div, /)
#undef ARITHMETIC_VEC_VEC

#define ARITHMETIC_DOUBLE_VEC_VEC(OPERATION, OPERATOR, SSE_OPERATION) \
template <> \
void OPERATION<double>(const Vector<double>& lhs, const Vector<double>& rhs, \
               Vector<double>& ans) { \
  const Vector<double>::size_type size_c_to_c_omp_sse = 476; \
  const Vector<double>::size_type size = ans.shape()[0]; \
  const double* ptr_lhs = &lhs[0]; \
  const double* ptr_rhs = &rhs[0]; \
  double* ptr_ans = &ans[0]; \
  if (size < size_c_to_c_omp_sse) { \
    for (Vector<double>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = ptr_lhs[idx] OPERATOR ptr_rhs[idx]; \
    } \
  } else { \
    OMP_FOR_3_PTR \
    for (Vector<double>::size_type idx = 0; idx < size / 2; ++idx) { \
      _mm_store_pd(ptr_ans + 2 * idx, SSE_OPERATION( \
        _mm_load_pd(ptr_lhs + 2 * idx), _mm_load_pd(ptr_rhs + 2 * idx))); \
    } \
    if (size % 2 != 0) { \
      ptr_ans[size - 1] = ptr_lhs[size - 1] + ptr_rhs[size - 1]; \
    } \
  } \
}
ARITHMETIC_DOUBLE_VEC_VEC(add, +, _mm_add_pd)
ARITHMETIC_DOUBLE_VEC_VEC(sub, -, _mm_sub_pd)
ARITHMETIC_DOUBLE_VEC_VEC(mul, *, _mm_mul_pd)
ARITHMETIC_DOUBLE_VEC_VEC(div, /, _mm_div_pd)
#undef ARITHMETIC_DOUBLE_VEC_VEC

#define ARITHMETIC_FLOAT_VEC_VEC(OPERATION, OPERATOR, SSE_OPERATION) \
template <> \
void OPERATION<float>(const Vector<float>& lhs, const Vector<float>& rhs, \
               Vector<float>& ans) { \
  const Vector<float>::size_type size_c_sse_to_c_omp_sse = 826; \
  const Vector<float>::size_type size = ans.shape()[0]; \
  const float* ptr_lhs = &lhs[0]; \
  const float* ptr_rhs = &rhs[0]; \
  float* ptr_ans = &ans[0]; \
  if (size < size_c_sse_to_c_omp_sse) { \
    for (Vector<float>::size_type idx = 0; idx < size / 4; ++idx) { \
      _mm_store_ps(ptr_ans + 4 * idx, SSE_OPERATION( \
        _mm_load_ps(ptr_lhs + 4 * idx), _mm_load_ps(ptr_rhs + 4 * idx))); \
    } \
  } else { \
    OMP_FOR_3_PTR \
    for (Vector<float>::size_type idx = 0; idx < size / 4; ++idx) { \
      _mm_store_ps(ptr_ans + 4 * idx, SSE_OPERATION( \
        _mm_load_ps(ptr_lhs + 4 * idx), _mm_load_ps(ptr_rhs + 4 * idx))); \
    } \
  } \
  if (size % 4 == 1) { \
    ptr_ans[size - 1] = ptr_lhs[size - 1] OPERATOR ptr_rhs[size - 1]; \
  } else if (size % 4 == 2) { \
    ptr_ans[size - 2] = ptr_lhs[size - 2] OPERATOR ptr_rhs[size - 2]; \
    ptr_ans[size - 1] = ptr_lhs[size - 1] OPERATOR ptr_rhs[size - 1]; \
  } else if (size % 4 == 3) { \
    ptr_ans[size - 3] = ptr_lhs[size - 3] OPERATOR ptr_rhs[size - 3]; \
    ptr_ans[size - 2] = ptr_lhs[size - 2] OPERATOR ptr_rhs[size - 2]; \
    ptr_ans[size - 1] = ptr_lhs[size - 1] OPERATOR ptr_rhs[size - 1]; \
  } \
}
ARITHMETIC_FLOAT_VEC_VEC(add, +, _mm_add_ps)
ARITHMETIC_FLOAT_VEC_VEC(sub, -, _mm_sub_ps)
ARITHMETIC_FLOAT_VEC_VEC(mul, *, _mm_mul_ps)
ARITHMETIC_FLOAT_VEC_VEC(div, /, _mm_div_ps)
#undef ARITHMETIC_FLOAT_VEC_VEC


#define ARITHMETIC_VEC_SCA(OPERATION, OPERATOR) \
template <typename T> \
void OPERATION(const Vector<T>& lhs, const T& rhs, \
               Vector<T>& ans) { \
  const typename Vector<T>::size_type size = ans.shape()[0]; \
  const T* ptr_lhs = &lhs[0]; \
  T* ptr_ans = &ans[0]; \
  for (typename Vector<T>::index_type idx = 0; idx < size; ++idx) { \
    ptr_ans[idx] = ptr_lhs[idx] OPERATOR rhs; \
  } \
}
ARITHMETIC_VEC_SCA(add, +)
ARITHMETIC_VEC_SCA(sub, -)
ARITHMETIC_VEC_SCA(mul, *)
ARITHMETIC_VEC_SCA(div, /)
#undef ARITHMETIC_VEC_SCA

#define ARITHMETIC_DOUBLE_VEC_SCA(OPERATION, OPERATOR, SSE_OPERATION) \
template <> \
void OPERATION<double>(const Vector<double>& lhs, const double& rhs, \
               Vector<double>& ans) { \
  const Vector<double>::size_type size_c_to_c_omp = 595; \
  const Vector<double>::size_type size_c_omp_to_c_omp_sse = 6726; \
  const Vector<double>::size_type size = ans.shape()[0]; \
  const double* ptr_lhs = &lhs[0]; \
  double* ptr_ans = &ans[0]; \
  if (size < size_c_to_c_omp) { \
    for (Vector<double>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = ptr_lhs[idx] OPERATOR rhs; \
    } \
  } else if (size < size_c_omp_to_c_omp_sse) { \
    OMP_FOR_2_PTR_L_ANS \
    for (Vector<double>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = ptr_lhs[idx] OPERATOR rhs; \
    } \
  } else { \
    const double* ptr_rhs = &rhs; \
    OMP_FOR_3_PTR \
    for (Vector<double>::size_type idx = 0; idx < size / 2; ++idx) { \
      _mm_store_pd(ptr_ans + 2 * idx, SSE_OPERATION( \
        _mm_load_pd(ptr_lhs + 2 * idx), _mm_load1_pd(ptr_rhs))); \
    } \
    if (size % 2 != 0) { \
      ptr_ans[size - 1] = ptr_lhs[size - 1] OPERATOR rhs; \
    } \
  }\
}
ARITHMETIC_DOUBLE_VEC_SCA(add, +, _mm_add_pd)
ARITHMETIC_DOUBLE_VEC_SCA(sub, -, _mm_sub_pd)
ARITHMETIC_DOUBLE_VEC_SCA(mul, *, _mm_mul_pd)
ARITHMETIC_DOUBLE_VEC_SCA(div, /, _mm_div_pd)
#undef ARITHMETIC_DOUBLE_VEC_SCA

#define ARITHMETIC_FLOAT_VEC_SCA(OPERATION, OPERATOR, SSE_OPERATION) \
template <> \
void OPERATION<float>(const Vector<float>& lhs, const float& rhs, \
               Vector<float>& ans) { \
  const Vector<float>::size_type size_c_sse_to_c_omp_sse = 917; \
  const Vector<float>::size_type size = ans.shape()[0]; \
  const float* ptr_lhs = &lhs[0]; \
  const float* ptr_rhs = &rhs; \
  float* ptr_ans = &ans[0]; \
  if (size < size_c_sse_to_c_omp_sse) { \
    for (Vector<float>::size_type idx = 0; idx < size / 4; ++idx) { \
      _mm_store_ps(ptr_ans + 4 * idx, SSE_OPERATION( \
        _mm_load_ps(ptr_lhs + 4 * idx), _mm_load1_ps(ptr_rhs))); \
    } \
  } else { \
    OMP_FOR_3_PTR \
    for (Vector<float>::size_type idx = 0; idx < size / 4; ++idx) { \
      _mm_store_ps(ptr_ans + 4 * idx, SSE_OPERATION( \
        _mm_load_ps(ptr_lhs + 4 * idx), _mm_load1_ps(ptr_rhs))); \
    } \
  } \
  if (size % 4 == 1) { \
    ptr_ans[size - 1] = ptr_lhs[size - 1] OPERATOR rhs; \
  } else if (size % 4 == 2) { \
    ptr_ans[size - 2] = ptr_lhs[size - 2] OPERATOR rhs; \
    ptr_ans[size - 1] = ptr_lhs[size - 1] OPERATOR rhs; \
  } else if (size % 4 == 3) { \
    ptr_ans[size - 3] = ptr_lhs[size - 3] OPERATOR rhs; \
    ptr_ans[size - 2] = ptr_lhs[size - 2] OPERATOR rhs; \
    ptr_ans[size - 1] = ptr_lhs[size - 1] OPERATOR rhs; \
  } \
}
ARITHMETIC_FLOAT_VEC_SCA(add, +, _mm_add_ps)
ARITHMETIC_FLOAT_VEC_SCA(sub, -, _mm_sub_ps)
ARITHMETIC_FLOAT_VEC_SCA(mul, *, _mm_mul_ps)
ARITHMETIC_FLOAT_VEC_SCA(div, /, _mm_div_ps)
#undef ARITHMETIC_FLOAT_VEC_SCA


#define ARITHMETIC_SCA_VEC(OPERATION, OPERATOR) \
template <typename T> \
void OPERATION(const T& lhs, const Vector<T>& rhs, \
               Vector<T>& ans) { \
  const typename Vector<T>::size_type size = ans.shape()[0]; \
  const T* ptr_rhs = &rhs[0]; \
  T* ptr_ans = &ans[0]; \
  for (typename Vector<T>::index_type idx = 0; idx < size; ++idx) { \
    ptr_ans[idx] = lhs OPERATOR ptr_rhs[idx]; \
  } \
}
ARITHMETIC_SCA_VEC(add, +)
ARITHMETIC_SCA_VEC(sub, -)
ARITHMETIC_SCA_VEC(mul, *)
ARITHMETIC_SCA_VEC(div, /)
#undef ARITHMETIC_SCA_VEC

#define ARITHMETIC_DOUBLE_SCA_VEC(OPERATION, OPERATOR, SSE_OPERATION) \
template <> \
void OPERATION<double>(const double& lhs, const Vector<double>& rhs, \
               Vector<double>& ans) { \
  const Vector<double>::size_type size_c_to_c_omp = 595; \
  const Vector<double>::size_type size_c_omp_to_c_omp_sse = 6726; \
  const Vector<double>::size_type size = ans.shape()[0]; \
  const double* ptr_rhs = &rhs[0]; \
  double* ptr_ans = &ans[0]; \
  if (size < size_c_to_c_omp) { \
    for (Vector<double>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = lhs OPERATOR ptr_rhs[idx]; \
    } \
  } else if (size < size_c_omp_to_c_omp_sse) { \
    OMP_FOR_2_PTR_R_ANS \
    for (Vector<double>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = lhs OPERATOR ptr_rhs[idx]; \
    } \
  } else { \
    const double* ptr_lhs = &lhs; \
    OMP_FOR_3_PTR \
    for (Vector<double>::size_type idx = 0; idx < size / 2; ++idx) { \
      _mm_store_pd(ptr_ans + 2 * idx, SSE_OPERATION( \
        _mm_load1_pd(ptr_lhs), _mm_load_pd(ptr_rhs + 2 * idx))); \
    } \
    if (size % 2 != 0) { \
      ptr_ans[size - 1] = lhs OPERATOR ptr_rhs[size - 1]; \
    } \
  }\
}
ARITHMETIC_DOUBLE_SCA_VEC(add, +, _mm_add_pd)
ARITHMETIC_DOUBLE_SCA_VEC(sub, -, _mm_sub_pd)
ARITHMETIC_DOUBLE_SCA_VEC(mul, *, _mm_mul_pd)
ARITHMETIC_DOUBLE_SCA_VEC(div, /, _mm_div_pd)
#undef ARITHMETIC_DOUBLE_SCA_VEC

#define ARITHMETIC_FLOAT_SCA_VEC(OPERATION, OPERATOR, SSE_OPERATION) \
template <> \
void OPERATION<float>(const float& lhs, const Vector<float>& rhs, \
               Vector<float>& ans) { \
  const Vector<float>::size_type size_c_sse_to_c_omp_sse = 917; \
  const Vector<float>::size_type size = ans.shape()[0]; \
  const float* ptr_lhs = &lhs; \
  const float* ptr_rhs = &rhs[0]; \
  float* ptr_ans = &ans[0]; \
  if (size < size_c_sse_to_c_omp_sse) { \
    for (Vector<float>::size_type idx = 0; idx < size / 4; ++idx) { \
      _mm_store_ps(ptr_ans + 4 * idx, SSE_OPERATION( \
        _mm_load1_ps(ptr_lhs), _mm_load_ps(ptr_rhs + 4 * idx))); \
    } \
  } else { \
    OMP_FOR_3_PTR \
    for (Vector<float>::size_type idx = 0; idx < size / 4; ++idx) { \
      _mm_store_ps(ptr_ans + 4 * idx, SSE_OPERATION( \
        _mm_load1_ps(ptr_lhs), _mm_load_ps(ptr_rhs + 4 * idx))); \
    } \
  } \
  if (size % 4 == 1) { \
    ptr_ans[size - 1] = lhs OPERATOR ptr_rhs[size - 1]; \
  } else if (size % 4 == 2) { \
    ptr_ans[size - 2] = lhs OPERATOR ptr_rhs[size - 2]; \
    ptr_ans[size - 1] = lhs OPERATOR ptr_rhs[size - 1]; \
  } else if (size % 4 == 3) { \
    ptr_ans[size - 3] = lhs OPERATOR ptr_rhs[size - 3]; \
    ptr_ans[size - 2] = lhs OPERATOR ptr_rhs[size - 2]; \
    ptr_ans[size - 1] = lhs OPERATOR ptr_rhs[size - 1]; \
  } \
}
ARITHMETIC_FLOAT_SCA_VEC(add, +, _mm_add_ps)
ARITHMETIC_FLOAT_SCA_VEC(sub, -, _mm_sub_ps)
ARITHMETIC_FLOAT_SCA_VEC(mul, *, _mm_mul_ps)
ARITHMETIC_FLOAT_SCA_VEC(div, /, _mm_div_ps)
#undef ARITHMETIC_FLOAT_SCA_VEC


template <typename T>
void sum(const Vector<T>& v, T& s) {
  const typename Vector<T>::size_type size = v.shape()[0];
  const T* ptr = &v[0];
  s = T();
  for (typename Vector<T>::size_type idx = 0; idx < size; ++idx) {
    s = s + ptr[idx];
  }
}

template <>
void sum<double>(const Vector<double>& v, double& s) {
  const Vector<double>::size_type size_c_to_c_sse = 18;
  const Vector<double>::size_type size_c_sse_to_c_omp = 856;
  const Vector<double>::size_type size = v.shape()[0];
  const double* ptr = &v[0];
  s = 0.0;
  if (size < size_c_to_c_sse) {
    for (Vector<double>::size_type idx = 0; idx < size; ++idx) {
      s = s + ptr[idx];
    }
  } else if (size < size_c_sse_to_c_omp) {
    __m128d s_reg = _mm_set1_pd(0.0);
    for (Vector<double>::size_type idx = 0; idx < size / 2; ++idx) {
      s_reg = _mm_add_pd(s_reg, _mm_load_pd(ptr + 2 * idx));
    }
    s_reg = _mm_hadd_pd(s_reg, s_reg);
    _mm_store_sd(&s, s_reg);
    if (size % 2 != 0) {
      s = s + ptr[size - 1];
    }
  } else {
    #pragma omp parallel for schedule(auto) reduction(+ : s)
    for (Vector<double>::size_type idx = 0; idx < size; ++idx) {
      s = s + ptr[idx];
    }
  }
}

template <>
void sum<float>(const Vector<float>& v, float& s) {
  const Vector<float>::size_type size_c_to_c_sse = 10;
  const Vector<float>::size_type size = v.shape()[0];
  const float* ptr = &v[0];
  s = 0.0;
  if (size < size_c_to_c_sse) {
    for (Vector<float>::size_type idx = 0; idx < size; ++idx) {
      s = s + ptr[idx];
    }
  } else {
    __m128 s_reg = _mm_set1_ps(0.0);
    for (std::size_t idx = 0; idx < size / 4; ++idx) {
      s_reg = _mm_add_ps(s_reg, _mm_load_ps(ptr + 4 * idx));
    }
    s_reg = _mm_hadd_ps(s_reg, s_reg);
    s_reg = _mm_hadd_ps(s_reg, s_reg);
    _mm_store_ss(&s, s_reg);
    if (size % 4 == 1) {
      s = s + ptr[size - 1];
    } else if (size % 4 == 2) {
      s = s + ptr[size - 1] + ptr[size - 2];
    } else if (size % 4 == 3) {
      s = s + ptr[size - 1] + ptr[size - 2] + ptr[size - 3];
    }
  }
}

}  // namespace internal

template <typename AriT>
Vector<AriT> operator+(const Vector<AriT>& vec_lhs, 
                       const Vector<AriT>& vec_rhs) {
  Vector<AriT>::shape_consistence_check_(vec_lhs.shape(), vec_rhs.shape());
  Vector<AriT> vec_sum(vec_lhs.shape()[0]);
  internal::add(vec_lhs, vec_rhs, vec_sum);
  return vec_sum;
}
template <typename AriT>
Vector<AriT> operator+(const Vector<AriT>& vec_lhs,
                       const AriT& val_rhs) {
  Vector<AriT> vec_sum(vec_lhs.shape()[0]);
  internal::add(vec_lhs, val_rhs, vec_sum);
  return vec_sum;
}
template <typename AriT>
Vector<AriT> operator+(const AriT& val_lhs,
                       const Vector<AriT>& vec_rhs) {
  Vector<AriT> vec_sum(vec_rhs.shape()[0]);
  internal::add(val_lhs, vec_rhs, vec_sum);
  return vec_sum;
}
template <typename AriT>
Vector<AriT> operator-(const Vector<AriT>& vec_lhs,
                       const Vector<AriT>& vec_rhs) {
  Vector<AriT>::shape_consistence_check_(vec_lhs.shape(), vec_rhs.shape());
  Vector<AriT> vec_diff(vec_lhs.shape()[0]);
  internal::sub(vec_lhs, vec_rhs, vec_diff);
  return vec_diff;
}
template <typename AriT>
Vector<AriT> operator-(const Vector<AriT>& vec_lhs,
                       const AriT& val_rhs) {
  Vector<AriT> vec_diff(vec_lhs.shape()[0]);
  internal::sub(vec_lhs, val_rhs, vec_diff);
  return vec_diff;
}
template <typename AriT>
Vector<AriT> operator-(const AriT& val_lhs,
                       const Vector<AriT>& vec_rhs) {
  Vector<AriT> vec_diff(vec_rhs.shape()[0]);
  internal::sub(val_lhs, vec_rhs, vec_diff);
  return vec_diff;
}
template <typename AriT>
Vector<AriT> operator*(const Vector<AriT>& vec_lhs, 
                       const Vector<AriT>& vec_rhs) {
  Vector<AriT>::shape_consistence_check_(vec_lhs.shape(), vec_rhs.shape());
  Vector<AriT> vec_prod(vec_lhs.shape()[0]);
  internal::mul(vec_lhs, vec_rhs, vec_prod);
  return vec_prod;
}
template <typename AriT>
Vector<AriT> operator*(const Vector<AriT>& vec_lhs,
                       const AriT& val_rhs) {
  Vector<AriT> vec_prod(vec_lhs.shape()[0]);
  internal::mul(vec_lhs, val_rhs, vec_prod);
  return vec_prod;
}
template <typename AriT>
Vector<AriT> operator*(const AriT& val_lhs,
                       const Vector<AriT>& vec_rhs) {
  Vector<AriT> vec_prod(vec_rhs.shape()[0]);
  internal::mul(val_lhs, vec_rhs, vec_prod);
  return vec_prod;
}
template <typename AriT>
Vector<AriT> operator/(const Vector<AriT>& vec_lhs, 
                       const Vector<AriT>& vec_rhs) {
  Vector<AriT>::shape_consistence_check_(vec_lhs.shape(), vec_rhs.shape());
  Vector<AriT> vec_quot(vec_lhs.shape()[0]);
  internal::div(vec_lhs, vec_rhs, vec_quot);
  return vec_quot;
}
template <typename AriT>
Vector<AriT> operator/(const Vector<AriT>& vec_lhs,
                       const AriT& val_rhs) {
  Vector<AriT> vec_quot(vec_lhs.shape()[0]);
  internal::div(vec_lhs, val_rhs, vec_quot);
  return vec_quot;
}
template <typename AriT>
Vector<AriT> operator/(const AriT& val_lhs,
                       const Vector<AriT>& vec_rhs) {
  Vector<AriT> vec_quot(vec_rhs.shape()[0]);
  internal::div(val_lhs, vec_rhs, vec_quot);
  return vec_quot;
}
template <typename T>
void Vector<T>::operator+=(const Vector<T>& vec_rhs) {
  (*this) = (*this) + vec_rhs;
}
template <typename T>
void Vector<T>::operator+=(const T& val_rhs) {
  (*this) = (*this) + val_rhs;
}
template <typename T>
void Vector<T>::operator-=(const Vector<T>& vec_rhs) {
  (*this) = (*this) - vec_rhs;
}
template <typename T>
void Vector<T>::operator-=(const T& val_rhs) {
  (*this) = (*this) - val_rhs;
}
template <typename T>
void Vector<T>::operator*=(const Vector<T>& vec_rhs) {
  (*this) = (*this) * vec_rhs;
}
template <typename T>
void Vector<T>::operator*=(const T& val_rhs) {
  (*this) = (*this) * val_rhs;
}
template <typename T>
void Vector<T>::operator/=(const Vector<T>& vec_rhs) {
  (*this) = (*this) / vec_rhs;
}
template <typename T>
void Vector<T>::operator/=(const T& val_rhs) {
  (*this) = (*this) / val_rhs;
}
template <typename T>
T Vector<T>::sum() const {
  T sum_res = T();
  internal::sum(*this, sum_res);
  return sum_res;
}


// ======== Comparisons ========
// namespace internal
namespace internal {

#define COMPARISON_VEC_VEC(OPERATION, OPERATOR) \
template <typename T> \
void OPERATION(const Vector<T>& lhs, const Vector<T>& rhs, \
               Vector<T>& ans) { \
  const typename Vector<T>::size_type size = ans.shape()[0]; \
  const T* ptr_lhs = &lhs[0]; \
  const T* ptr_rhs = &rhs[0]; \
  T* ptr_ans = &ans[0]; \
  for (typename Vector<T>::size_type idx = 0; idx < size; ++idx) { \
    ptr_ans[idx] = static_cast<T>(ptr_lhs[idx] OPERATOR ptr_rhs[idx]); \
  } \
}
COMPARISON_VEC_VEC(equal, ==)
COMPARISON_VEC_VEC(nequal, !=)
COMPARISON_VEC_VEC(less, <)
COMPARISON_VEC_VEC(leq, <=)
COMPARISON_VEC_VEC(greater, >)
COMPARISON_VEC_VEC(geq, >=)
#undef COMPARISON_VEC_VEC

#define COMPARISON_DOUBLE_VEC_VEC(OPERATION, OPERATOR) \
template <> \
void OPERATION<double>(const Vector<double>& lhs, const Vector<double>& rhs, \
                       Vector<double>& ans) { \
  const Vector<double>::size_type size_c_to_c_omp = 550; \
  const Vector<double>::size_type size_c_omp_to_c = 3600; \
  const Vector<double>::size_type size = ans.shape()[0]; \
  const double* ptr_lhs = &lhs[0]; \
  const double* ptr_rhs = &rhs[0]; \
  double* ptr_ans = &ans[0]; \
  if (size < size_c_to_c_omp || size >= size_c_omp_to_c) { \
    for (Vector<double>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = static_cast<double>(ptr_lhs[idx] OPERATOR ptr_rhs[idx]); \
    } \
  } else { \
    OMP_FOR_3_PTR \
    for (Vector<double>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = static_cast<double>(ptr_lhs[idx] OPERATOR ptr_rhs[idx]); \
    } \
  } \
}
COMPARISON_DOUBLE_VEC_VEC(equal, ==)
COMPARISON_DOUBLE_VEC_VEC(nequal, !=)
COMPARISON_DOUBLE_VEC_VEC(less, <)
COMPARISON_DOUBLE_VEC_VEC(leq, <=)
COMPARISON_DOUBLE_VEC_VEC(greater, >)
COMPARISON_DOUBLE_VEC_VEC(geq, >=)
#undef COMPARISON_DOUBLE_VEC_VEC

#define COMPARISON_FLOAT_VEC_VEC(OPERATION, OPERATOR) \
template <> \
void OPERATION<float>(const Vector<float>& lhs, const Vector<float>& rhs, \
                      Vector<float>& ans) { \
  const Vector<float>::size_type size_c_to_c_omp = 452; \
  const Vector<float>::size_type size_c_omp_to_c = 4098; \
  const Vector<float>::size_type size = ans.shape()[0]; \
  const float* ptr_lhs = &lhs[0]; \
  const float* ptr_rhs = &rhs[0]; \
  float* ptr_ans = &ans[0]; \
  if (size < size_c_to_c_omp || size >= size_c_omp_to_c) { \
    for (Vector<float>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = static_cast<float>(ptr_lhs[idx] OPERATOR ptr_rhs[idx]); \
    } \
  } else { \
    OMP_FOR_3_PTR \
    for (Vector<float>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = static_cast<float>(ptr_lhs[idx] OPERATOR ptr_rhs[idx]); \
    } \
  } \
}
COMPARISON_FLOAT_VEC_VEC(equal, ==)
COMPARISON_FLOAT_VEC_VEC(nequal, !=)
COMPARISON_FLOAT_VEC_VEC(less, <)
COMPARISON_FLOAT_VEC_VEC(leq, <=)
COMPARISON_FLOAT_VEC_VEC(greater, >)
COMPARISON_FLOAT_VEC_VEC(geq, >=)
#undef COMPARISON_FLOAT_VEC_VEC


#define COMPARISON_VEC_SCA(OPERATION, OPERATOR) \
template <typename T> \
void OPERATION(const Vector<T>& lhs, const T& rhs, Vector<T>& ans) {\
  const typename Vector<T>::size_type size = ans.shape()[0]; \
  const T* ptr_lhs = &lhs[0]; \
  T* ptr_ans = &ans[0]; \
  for (typename Vector<T>::size_type idx = 0; idx < size; ++idx) { \
    ptr_ans[idx] = static_cast<T>(ptr_lhs[idx] OPERATOR rhs); \
  } \
}
COMPARISON_VEC_SCA(equal, ==)
COMPARISON_VEC_SCA(nequal, !=)
COMPARISON_VEC_SCA(less, <)
COMPARISON_VEC_SCA(leq, <=)
COMPARISON_VEC_SCA(greater, >)
COMPARISON_VEC_SCA(geq, >=)
#undef COMPARISON_VEC_SCA

#define COMPARISON_DOUBLE_VEC_SCA(OPERATION, OPERATOR) \
template <> \
void OPERATION<double>(const Vector<double>& lhs, const double& rhs, \
                       Vector<double>& ans) { \
  const Vector<double>::size_type size_c_to_c_omp = 554; \
  const Vector<double>::size_type size_c_omp_to_c = 7170; \
  const Vector<double>::size_type size = ans.shape()[0]; \
  const double* ptr_lhs = &lhs[0]; \
  double* ptr_ans = &ans[0]; \
  if (size < size_c_to_c_omp || size >= size_c_omp_to_c) { \
    for (Vector<double>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = static_cast<double>(ptr_lhs[idx] OPERATOR rhs); \
    } \
  } else { \
    OMP_FOR_2_PTR_L_ANS \
    for (Vector<double>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = static_cast<double>(ptr_lhs[idx] OPERATOR rhs); \
    } \
  } \
}
COMPARISON_DOUBLE_VEC_SCA(equal, ==)
COMPARISON_DOUBLE_VEC_SCA(nequal, !=)
COMPARISON_DOUBLE_VEC_SCA(less, <)
COMPARISON_DOUBLE_VEC_SCA(leq, <=)
COMPARISON_DOUBLE_VEC_SCA(greater, >)
COMPARISON_DOUBLE_VEC_SCA(geq, >=)
#undef COMPARISON_DOUBLE_VEC_SCA

#define COMPARISON_FLOAT_VEC_SCA(OPERATION, OPERATOR) \
template <> \
void OPERATION<float>(const Vector<float>& lhs, const float& rhs, \
                      Vector<float>& ans) { \
  const Vector<float>::size_type size_c_to_c_omp = 514; \
  const Vector<float>::size_type size_c_omp_to_c = 7243; \
  const Vector<float>::size_type size = ans.shape()[0]; \
  const float* ptr_lhs = &lhs[0]; \
  float* ptr_ans = &ans[0]; \
  if (size < size_c_to_c_omp || size >= size_c_omp_to_c) { \
    for (Vector<float>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = static_cast<float>(ptr_lhs[idx] OPERATOR rhs); \
    } \
  } else { \
    OMP_FOR_2_PTR_L_ANS \
    for (Vector<float>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = static_cast<float>(ptr_lhs[idx] OPERATOR rhs); \
    } \
  } \
}
COMPARISON_FLOAT_VEC_SCA(equal, ==)
COMPARISON_FLOAT_VEC_SCA(nequal, !=)
COMPARISON_FLOAT_VEC_SCA(less, <)
COMPARISON_FLOAT_VEC_SCA(leq, <=)
COMPARISON_FLOAT_VEC_SCA(greater, >)
COMPARISON_FLOAT_VEC_SCA(geq, >=)
#undef COMPARISON_FLOAT_VEC_SCA


#define COMPARISON_SCA_VEC(OPERATION, OPERATOR) \
template <typename T> \
void OPERATION(const T& lhs, const Vector<T>& rhs, Vector<T>& ans) {\
  const typename Vector<T>::size_type size = ans.shape()[0]; \
  const T* ptr_rhs = &rhs[0]; \
  T* ptr_ans = &ans[0]; \
  for (typename Vector<T>::size_type idx = 0; idx < size; ++idx) { \
    ptr_ans[idx] = static_cast<T>(lhs OPERATOR ptr_rhs[idx]); \
  } \
}
COMPARISON_SCA_VEC(equal, ==)
COMPARISON_SCA_VEC(nequal, !=)
COMPARISON_SCA_VEC(less, <)
COMPARISON_SCA_VEC(leq, <=)
COMPARISON_SCA_VEC(greater, >)
COMPARISON_SCA_VEC(geq, >=)
#undef COMPARISON_SCA_VEC

#define COMPARISON_DOUBLE_SCA_VEC(OPERATION, OPERATOR) \
template <> \
void OPERATION<double>(const double& lhs, const Vector<double>& rhs, \
                       Vector<double>& ans) { \
  const Vector<double>::size_type size_c_to_c_omp = 554; \
  const Vector<double>::size_type size_c_omp_to_c = 7170; \
  const Vector<double>::size_type size = ans.shape()[0]; \
  const double* ptr_rhs = &rhs[0]; \
  double* ptr_ans = &ans[0]; \
  if (size < size_c_to_c_omp || size >= size_c_omp_to_c) { \
    for (Vector<double>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = static_cast<double>(lhs OPERATOR ptr_rhs[idx]); \
    } \
  } else { \
    OMP_FOR_2_PTR_R_ANS \
    for (Vector<double>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = static_cast<double>(lhs OPERATOR ptr_rhs[idx]); \
    } \
  } \
}
COMPARISON_DOUBLE_SCA_VEC(equal, ==)
COMPARISON_DOUBLE_SCA_VEC(nequal, !=)
COMPARISON_DOUBLE_SCA_VEC(less, <)
COMPARISON_DOUBLE_SCA_VEC(leq, <=)
COMPARISON_DOUBLE_SCA_VEC(greater, >)
COMPARISON_DOUBLE_SCA_VEC(geq, >=)
#undef COMPARISON_DOUBLE_SCA_VEC

#define COMPARISON_FLOAT_SCA_VEC(OPERATION, OPERATOR) \
template <> \
void OPERATION<float>(const float& lhs, const Vector<float>& rhs, \
                      Vector<float>& ans) { \
  const Vector<float>::size_type size_c_to_c_omp = 514; \
  const Vector<float>::size_type size_c_omp_to_c = 7243; \
  const Vector<float>::size_type size = ans.shape()[0]; \
  const float* ptr_rhs = &rhs[0]; \
  float* ptr_ans = &ans[0]; \
  if (size < size_c_to_c_omp || size >= size_c_omp_to_c) { \
    for (Vector<float>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = static_cast<float>(lhs OPERATOR ptr_rhs[idx]); \
    } \
  } else { \
    OMP_FOR_2_PTR_R_ANS \
    for (Vector<float>::size_type idx = 0; idx < size; ++idx) { \
      ptr_ans[idx] = static_cast<float>(lhs OPERATOR ptr_rhs[idx]); \
    } \
  } \
}
COMPARISON_FLOAT_SCA_VEC(equal, ==)
COMPARISON_FLOAT_SCA_VEC(nequal, !=)
COMPARISON_FLOAT_SCA_VEC(less, <)
COMPARISON_FLOAT_SCA_VEC(leq, <=)
COMPARISON_FLOAT_SCA_VEC(greater, >)
COMPARISON_FLOAT_SCA_VEC(geq, >=)
#undef COMPARISON_FLOAT_SCA_VEC

}  // namespace internal

template <typename CmpT>
Vector<CmpT> operator==(const Vector<CmpT>& vec_lhs, 
                        const Vector<CmpT>& vec_rhs) {
  Vector<CmpT>::shape_consistence_check_(vec_lhs.shape(), vec_rhs.shape());
  Vector<CmpT> vec_eq(vec_lhs.shape()[0]);
  internal::equal(vec_lhs, vec_rhs, vec_eq);
  return vec_eq;
}
template <typename CmpT>
Vector<CmpT> operator==(const Vector<CmpT>& vec_lhs, const CmpT& val_rhs) {
  Vector<CmpT> vec_eq(vec_lhs.shape()[0]);
  internal::equal(vec_lhs, val_rhs, vec_eq);
  return vec_eq;
}
template <typename CmpT>
Vector<CmpT> operator==(const CmpT& val_lhs, const Vector<CmpT>& vec_rhs) {
  Vector<CmpT> vec_eq(vec_rhs.shape()[0]);
  internal::equal(val_lhs, vec_rhs, vec_eq);
  return vec_eq;
}
template <typename CmpT>
Vector<CmpT> operator!=(const Vector<CmpT>& vec_lhs, 
                        const Vector<CmpT>& vec_rhs) {
  Vector<CmpT>::shape_consistence_check_(vec_lhs.shape(), vec_rhs.shape());
  Vector<CmpT> vec_neq(vec_lhs.shape()[0]);
  internal::nequal(vec_lhs, vec_rhs, vec_neq);
  return vec_neq;
}
template <typename CmpT>
Vector<CmpT> operator!=(const Vector<CmpT>& vec_lhs, const CmpT& val_rhs) {
  Vector<CmpT> vec_neq(vec_lhs.shape()[0]);
  internal::nequal(vec_lhs, val_rhs, vec_neq);
  return vec_neq;
}
template <typename CmpT>
Vector<CmpT> operator!=(const CmpT& val_lhs, const Vector<CmpT>& vec_rhs) {
  Vector<CmpT> vec_neq(vec_rhs.shape()[0]);
  internal::nequal(val_lhs, vec_rhs, vec_neq);
  return vec_neq;
}
template <typename CmpT>
Vector<CmpT> operator<(const Vector<CmpT>& vec_lhs, 
                       const Vector<CmpT>& vec_rhs) {
  Vector<CmpT>::shape_consistence_check_(vec_lhs.shape(), vec_rhs.shape());
  Vector<CmpT> vec_le(vec_lhs.shape()[0]);
  internal::less(vec_lhs, vec_rhs, vec_le);
  return vec_le;
}
template <typename CmpT>
Vector<CmpT> operator<(const Vector<CmpT>& vec_lhs, const CmpT& val_rhs) {
  Vector<CmpT> vec_le(vec_lhs.shape()[0]);
  internal::less(vec_lhs, val_rhs, vec_le);
  return vec_le;
}
template <typename CmpT>
Vector<CmpT> operator<(const CmpT& val_lhs, const Vector<CmpT>& vec_rhs) {
  Vector<CmpT> vec_le(vec_rhs.shape()[0]);
  internal::less(val_lhs, vec_rhs, vec_le);
  return vec_le;
}
template <typename CmpT>
Vector<CmpT> operator<=(const Vector<CmpT>& vec_lhs, 
                        const Vector<CmpT>& vec_rhs) {
  Vector<CmpT>::shape_consistence_check_(vec_lhs.shape(), vec_rhs.shape());
  Vector<CmpT> vec_leq(vec_lhs.shape()[0]);
  internal::leq(vec_lhs, vec_rhs, vec_leq);
  return vec_leq;
}
template <typename CmpT>
Vector<CmpT> operator<=(const Vector<CmpT>& vec_lhs, const CmpT& val_rhs) {
  Vector<CmpT> vec_leq(vec_lhs.shape()[0]);
  internal::leq(vec_lhs, val_rhs, vec_leq);
  return vec_leq;
}
template <typename CmpT>
Vector<CmpT> operator<=(const CmpT& val_lhs, const Vector<CmpT>& vec_rhs) {
  Vector<CmpT> vec_leq(vec_rhs.shape()[0]);
  internal::leq(val_lhs, vec_rhs, vec_leq);
  return vec_leq;
}
template <typename CmpT>
Vector<CmpT> operator>(const Vector<CmpT>& vec_lhs, 
                       const Vector<CmpT>& vec_rhs) {
  Vector<CmpT>::shape_consistence_check_(vec_lhs.shape(), vec_rhs.shape());
  Vector<CmpT> vec_ge(vec_lhs.shape()[0]);
  internal::greater(vec_lhs, vec_rhs, vec_ge);
  return vec_ge;
}
template <typename CmpT>
Vector<CmpT> operator>(const Vector<CmpT>& vec_lhs, const CmpT& val_rhs) {
  Vector<CmpT> vec_ge(vec_lhs.shape()[0]);
  internal::greater(vec_lhs, val_rhs, vec_ge);
  return vec_ge;
}
template <typename CmpT>
Vector<CmpT> operator>(const CmpT& val_lhs, const Vector<CmpT>& vec_rhs) {
  Vector<CmpT> vec_ge(vec_rhs.shape()[0]);
  internal::greater(val_lhs, vec_rhs, vec_ge);
  return vec_ge;
}
template <typename CmpT>
Vector<CmpT> operator>=(const Vector<CmpT>& vec_lhs, 
                        const Vector<CmpT>& vec_rhs) {
  Vector<CmpT>::shape_consistence_check_(vec_lhs.shape(), vec_rhs.shape());
  Vector<CmpT> vec_geq(vec_lhs.shape()[0]);
  internal::geq(vec_lhs, vec_rhs, vec_geq);
  return vec_geq;
}
template <typename CmpT>
Vector<CmpT> operator>=(const Vector<CmpT>& vec_lhs, const CmpT& val_rhs) {
  Vector<CmpT> vec_geq(vec_lhs.shape()[0]);
  internal::geq(vec_lhs, val_rhs, vec_geq);
  return vec_geq;
}
template <typename CmpT>
Vector<CmpT> operator>=(const CmpT& val_lhs, const Vector<CmpT>& vec_rhs) {
  Vector<CmpT> vec_geq(vec_rhs.shape()[0]);
  internal::geq(val_lhs, vec_rhs, vec_geq);
  return vec_geq;
}

// namespace internal
namespace internal {

template <typename T>
bool fequal(const T& x, const T& y, std::size_t ulp) {
  return std::fabs(x - y) <= ulp * std::numeric_limits<T>::epsilon() *
                             std::fabs((x + y) / 2.0)
      || std::fabs(x - y) <= std::numeric_limits<T>::min();
}
template <typename T>
bool is_equal(const Vector<T>& vec_lhs, const Vector<T>& vec_rhs,
              std::size_t ulp = 1) {
  const typename Vector<T>::size_type size = vec_lhs.shape()[0];
  const T* ptr_lhs = &vec_lhs[0];
  const T* ptr_rhs = &vec_rhs[0];
  if (std::is_floating_point<T>::value) {
    for (typename Vector<T>::size_type idx = 0; idx < size; ++idx) {
      if (!fequal(ptr_lhs[idx], ptr_rhs[idx], ulp)) {
        return false;
      }
    }
  } else {
    for (typename Vector<T>::size_type idx = 0; idx < size; ++idx) {
      if (ptr_lhs[idx] != ptr_rhs[idx]) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace internal

template <typename T>
bool Vector<T>::equal(const Vector& vec_rhs, std::size_t ulp) {
  if (shape()[0] != vec_rhs.shape()[0]) {
    return false;
  }
  if (shape()[0] == 0 && vec_rhs.shape()[0] == 0) {
    return true;
  }
  return internal::is_equal(*this, vec_rhs, ulp);
}
template <typename T>
bool Vector<T>::nequal(const Vector& vec_rhs, std::size_t ulp) {
  return !equal(vec_rhs, ulp);
}

// namespace internal
namespace internal {

template <typename T>
void max(const Vector<T>& v, T& m) {
  typename Vector<T>::size_type size = v.shape()[0];
  const T* ptr = &v[0];
  m = ptr[0];
  for (typename Vector<T>::size_type idx = 1; idx < size; ++idx) {
    if (ptr[idx] > m) {
      m = ptr[idx];
    }
  }
}
template <typename T>
void min(const Vector<T>& v, T& m) {
  typename Vector<T>::size_type size = v.shape()[0];
  const T* ptr = &v[0];
  m = ptr[0];
  for (typename Vector<T>::size_type idx = 1; idx < size; ++idx) {
    if (ptr[idx] < m) {
      m = ptr[idx];
    }
  }
}

}  // namespace internal

template <typename T>
T Vector<T>::max() const {
  T max_val;
  internal::max(*this, max_val);
  return max_val;
}
template <typename T>
T Vector<T>::min() const {
  T min_val;
  internal::min(*this, min_val);
  return min_val;
}


// ======== IO ========
template <typename VecT, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(
  std::basic_ostream<CharT, Traits>& os, const Vector<VecT>& vec) {
  if (vec.shape()[0] == 0) {
    os << "[]";
    return os;
  }
  os << "[";
  for (typename Vector<VecT>::size_type idx = 0; 
       idx < vec.shape()[0] - 1; ++idx) {
    os << vec[idx] << ", ";
  }
  os << vec[vec.shape()[0] - 1] << "]";
  return os;
}
template <typename VecT, typename CharT, typename Traits>
std::basic_istream<CharT, Traits>& operator>>(
  std::basic_istream<CharT, Traits>& is, Vector<VecT>& vec) {
  if (vec.vec_.size() == 0) {
    VecT element;
    while (is >> element) {
      vec.vec_.push_back(element);
    }
  } else {
    for (VecT& element : vec.vec_) {
      is >> element;
    }
  }
  return is;
}

// ======== Helper Functions ========
template <typename T>
bool Vector<T>::fequal_(const T& x, const T& y, std::size_t ulp) {
  // static_assert(std::is_floating_point<T>::value, 
  //   "Vector<T>::fequal_ only compares floating point numbers.");
  return std::fabs(x - y) <= ulp * std::numeric_limits<T>::epsilon() *
                             std::fabs((x + y) / 2.0)
      || std::fabs(x - y) <= std::numeric_limits<T>::min();
}
template <typename T>
typename Vector<T>::index_type Vector<T>::to_positive_index_(
  const size_type& size, const index_type& index) {
  return index >= 0 ? index : size + index;
}
template <typename T>
void Vector<T>::exclusive_range_check_(const size_type& size, 
                                       const index_type& index) {
  size_type pos_index = to_positive_index_(size, index);
  if (pos_index >= size) {
    std::string err_msg = "Out-of-Range: index " + 
      std::to_string(index) + " is out of range [0, " + 
      std::to_string(size) + ").";    
    throw VectorException(err_msg);     
  }
}
template <typename T>
void Vector<T>::exclusive_range_check_(const iterator& it_begin,
                                       const iterator& it_end,
                                       const iterator& it) {
  if (it < it_begin || it >= it_end) {
    std::string err_msg = 
      "Out-of-Range: iterator is out of the range [begin(), end()).";
    throw VectorException(err_msg);
  }
}
template <typename T>
void Vector<T>::exclusive_range_check_(const const_iterator& cit_begin,
                                       const const_iterator& cit_end,
                                       const const_iterator& cit) {
  if (cit < cit_begin || cit >= cit_end) {
    std::string err_msg = 
      "Out-of-Range: const_iterator is out of the range [begin(), end()).";
    throw VectorException(err_msg);
  }
}
template <typename T>
void Vector<T>::inclusive_range_check_(const size_type& size,
                                       const index_type& index) {
  size_type pos_index = to_positive_index_(size, index);
  if (pos_index > size) {
    std::string err_msg = "Out-of-Range: index " + 
      std::to_string(index) + " is out of range [0, " + 
      std::to_string(size) + "].";    
    throw VectorException(err_msg);     
  }
}
template <typename T>
void Vector<T>::inclusive_range_check_(const iterator& it_begin,
                                       const iterator& it_end,
                                       const iterator& it) {
  if (it < it_begin || it > it_end) {
    std::string err_msg = 
      "Out-of-Range: iterator is out of the range [begin(), end()].";
    throw VectorException(err_msg);
  }
}
template <typename T>
void Vector<T>::inclusive_range_check_(const const_iterator& cit_begin,
                                       const const_iterator& cit_end,
                                       const const_iterator& cit) {
  if (cit < cit_begin || cit > cit_end) {
    std::string err_msg = 
      "Out-of-Range: const_iterator is out of the range [cbegin(), cend()].";
    throw VectorException(err_msg);
  }
}
template <typename T>
void Vector<T>::shape_consistence_check_(const Vector<size_type>& shape_lhs,
                                         const Vector<size_type>& shape_rhs) {
  if (shape_lhs[0] != shape_rhs[0]) {
    std::string err_msg = "Inconsistent shape: [" +
      std::to_string(shape_lhs[0]) + "] != [" +
      std::to_string(shape_rhs[0]) + "].";
    throw VectorException(err_msg);
  }
}
template <typename T>
void Vector<T>::index_order_check_(const size_type& size,
                                   const index_type& idx_begin,
                                   const index_type& idx_end) {
  if (to_positive_index_(size, idx_begin) >
      to_positive_index_(size, idx_end)) {
    std::string err_msg = "Invalid Index Order: begin " +
      std::to_string(to_positive_index_(size, idx_begin)) + " > end " +
      std::to_string(to_positive_index_(size, idx_end)) + ".";
    throw VectorException(err_msg);
  }
}

template <typename T>
typename Vector<T>::index_type Vector<T>::to_positive_index_(
  const index_type& index) const {
  return to_positive_index_(vec_.size(), index);
}
template <typename T>
void Vector<T>::exclusive_range_check_(const index_type& index) const {
  exclusive_range_check_(vec_.size(), index);
}
template <typename T>
void Vector<T>::exclusive_range_check_(const iterator& it) {
  exclusive_range_check_(begin(), end(), it);
}
template <typename T>
void Vector<T>::exclusive_range_check_(const const_iterator& cit) const {
  exclusive_range_check_(cbegin(), cend(), cit);
}
template <typename T>
void Vector<T>::inclusive_range_check_(const index_type& index) const {
  inclusive_range_check_(vec_.size(), index);
}
template <typename T>
void Vector<T>::inclusive_range_check_(const iterator& it) {
  inclusive_range_check_(begin(), end(), it);
}
template <typename T>
void Vector<T>::inclusive_range_check_(const const_iterator& cit)  const {
  inclusive_range_check_(cbegin(), cend(), cit);
}
template <typename T>
void Vector<T>::index_order_check_(const index_type& idx_begin, 
                                   const index_type& idx_end) const {
  index_order_check_(vec_.size(), idx_begin, idx_end);
}
template <typename T>
void Vector<T>::iterator_order_check_(const iterator& it_begin,
                                      const iterator& it_end) {
  index_order_check_(vec_.size(), 
    static_cast<index_type>(it_begin - begin()),
    static_cast<index_type>(it_end - begin()));
}
template <typename T>
void Vector<T>::const_iterator_order_check_(
  const const_iterator& cit_begin, const const_iterator& cit_end) const {
  index_order_check_(vec_.size(),
    static_cast<index_type>(cit_begin - cbegin()),
    static_cast<index_type>(cit_end - cbegin()));
}
// ======== End of class Vector ========


class VectorException : public std::exception {
public:
  VectorException() noexcept {};
  VectorException(const VectorException& other) noexcept : msg_(other.msg_) {}
  explicit VectorException(const std::string& message) noexcept : 
    msg_(message) {}
  explicit VectorException(const char* message) noexcept : msg_(message) {}
  VectorException& operator=(const VectorException& other) noexcept {
    msg_ = other.msg_;
    return *this;
  }
  VectorException& operator=(const std::string& msg_copy) noexcept { 
    msg_ = msg_copy;
    return *this;
  }
  VectorException& operator=(const char* msg_copy) noexcept { 
    msg_ = msg_copy;
    return *this;
  }
  ~VectorException() noexcept {};
  const char* what() const noexcept { return msg_.c_str(); }

protected:
  std::string msg_;  
};

}  // namespace tensor

#endif  // TENSOR_VECTOR_H_