// Copyright (c) 2020, Massachusetts Institute of Technology,
// Cambridge, MA 02139
// All Rights Reserved
// Authors: Jingnan Shi, et al. (see THANKS for the full author list)
// See LICENSE for the license information

#include "catch.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <set>

#include <robin/utils.hpp>

TEST_CASE("prefix sum") {
  SECTION("all zeros") {
    size_t N = 10;
    auto* input = new size_t[N];
    auto* output = new size_t[N];
    memset(input, 0, N*sizeof(size_t));
    robin::PrefixSum<size_t>(input, output, N);
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(output[i] == 0);
    }
    delete[] input;
    delete[] output;
  }
  SECTION("all ones") {
    size_t N = 10;
    auto* input = new size_t[N];
    std::fill_n(input, N, 1);
    auto* output = new size_t[N];
    robin::PrefixSum<size_t>(input, output, N);
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(output[i] == i+1);
    }
    delete[] input;
    delete[] output;
  }
  SECTION("small arrays") {
    size_t N = 6;
    size_t input[] = {1,2,3,4,5,6};
    auto* output = new size_t[N];
    size_t output_expected[] = {1,3,6,10,15,21};
    robin::PrefixSum<size_t>(&(input[0]), output, 6);
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(output[i] == output_expected[i]);
    }
    delete[] output;
  }
  SECTION("in-place") {
    size_t N = 6;
    size_t input[] = {1,2,3,4,5,6};
    size_t output_expected[] = {1,3,6,10,15,21};
    robin::PrefixSum<size_t>(&(input[0]), &(input[0]), 6);
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(input[i] == output_expected[i]);
    }
  }
}