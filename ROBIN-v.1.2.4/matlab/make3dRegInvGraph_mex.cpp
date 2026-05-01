// Copyright (c) 2020, Massachusetts Institute of Technology,
// Cambridge, MA 02139
// All Rights Reserved
// Authors: Jingnan Shi, et al. (see THANKS for the full author list)
// See LICENSE for the license information

#include <robin/robin.hpp>

#include <chrono>
#include <iostream>
#include <map>

#include "mex.h"

#include "mex_utils.hpp"

enum class INPUT_PARAMS : int {
  src_3d_points = 0,
  dst_3d_points = 1,
  noise_bound = 2,
};

enum class OUTPUT_PARAMS : int {
  adj_mat = 0,
  time_taken = 1,
};

typedef bool (*mexTypeCheckFunction)(const mxArray*);
const std::map<INPUT_PARAMS, mexTypeCheckFunction> INPUT_PARMS_MAP{
    {INPUT_PARAMS::src_3d_points, &isRealDoubleMatrix},
    {INPUT_PARAMS::dst_3d_points, &isRealDoubleMatrix},
    {INPUT_PARAMS::noise_bound, &isRealDoubleScalar},
};

const std::map<OUTPUT_PARAMS, mexTypeCheckFunction> OUTPUT_PARMS_MAP{
    {OUTPUT_PARAMS::adj_mat, &isRealDoubleMatrix},
    {OUTPUT_PARAMS::time_taken, &isRealDoubleScalar},
};

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  // Check for proper number of arguments
  if (nrhs != INPUT_PARMS_MAP.size()) {
    mexErrMsgIdAndTxt("teaserSolve:nargin", "Wrong number of input arguments.");
  }
  if (nlhs != OUTPUT_PARMS_MAP.size()) {
    mexErrMsgIdAndTxt("teaserSolve:nargin", "Wrong number of output arguments.");
  }

  // Check for proper input types
  for (const auto& pair : INPUT_PARMS_MAP) {
    if (!pair.second(prhs[toUType(pair.first)])) {
      std::stringstream error_msg;
      error_msg << "Argument " << toUType(pair.first) + 1 << " has the wrong type.\n";
      mexErrMsgIdAndTxt("teaserSolve:nargin", error_msg.str().c_str());
    }
  }

  mexPrintf("Arguments type checks passed.\n");
  mexEvalString("drawnow;");

  // Prepare parameters
  Eigen::Matrix3Xd src_pc, dst_pc;
  mexPointMatrixToEigenMatrix(prhs[toUType(INPUT_PARAMS::src_3d_points)], &src_pc);
  mexPointMatrixToEigenMatrix(prhs[toUType(INPUT_PARAMS::dst_3d_points)], &dst_pc);

  auto noise_bound = static_cast<double>(*mxGetPr(prhs[toUType(INPUT_PARAMS::noise_bound)]));

  // Start the timer
  auto start = std::chrono::high_resolution_clock::now();

  // Call robin API
  auto g = robin::Make3dRegInvGraph(src_pc, dst_pc, noise_bound);

  // Stop the timer
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  double duration_in_milliseconds = static_cast<double>(duration.count()) / 1000.0;
  double duration_in_seconds = duration_in_milliseconds / 1000.0;

  // Return to MATLAB
  // populate the adjacency matrix
  plhs[toUType(OUTPUT_PARAMS::adj_mat)] =
      mxCreateDoubleMatrix(g.VertexCount(), g.VertexCount(), mxREAL);
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> g_map(
      mxGetPr(plhs[toUType(OUTPUT_PARAMS::adj_mat)]), g.VertexCount(), g.VertexCount());
  g_map = g.GetAdjMat();

  // populate the runtime
  plhs[toUType(OUTPUT_PARAMS::time_taken)] = mxCreateDoubleScalar(duration_in_seconds);
}