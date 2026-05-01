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
#include <Eigen/Core>

#include "mex_utils.hpp"

enum class INPUT_PARAMS : int {
  adj_mat = 0,
  graph_structure = 1,
};

enum class OUTPUT_PARAMS : int {
  vertices = 0,
  time_taken = 1,
};

typedef bool (*mexTypeCheckFunction)(const mxArray*);
const std::map<INPUT_PARAMS, mexTypeCheckFunction> INPUT_PARMS_MAP{
    {INPUT_PARAMS::adj_mat, &isSquareMatrix},
    {INPUT_PARAMS::graph_structure, &isRealDoubleScalar},
};

const std::map<OUTPUT_PARAMS, mexTypeCheckFunction> OUTPUT_PARMS_MAP{
    {OUTPUT_PARAMS::vertices, &isRealDoubleScalar},
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
  // adjacency matrix
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> adj_mat;
  mexMatrixToEigenMatrix(prhs[toUType(INPUT_PARAMS::adj_mat)], &adj_mat);

  // make a robin graph
  robin::Graph g;
  g.PopulateVertices(adj_mat.rows());
  for (size_t col_idx = 0; col_idx < adj_mat.cols(); ++col_idx) {
    for (size_t row_idx = 0; row_idx < adj_mat.rows(); ++row_idx) {
      if (adj_mat(row_idx, col_idx) != 0) {
        g.AddEdge(col_idx, row_idx);
      }
    }
  }

  // graph structure type
  auto graph_structure =
      static_cast<double>(*mxGetPr(prhs[toUType(INPUT_PARAMS::graph_structure)]));
  robin::InlierGraphStructure gs;
  if (graph_structure == 0) {
    // MAX_CORE
    gs = robin::InlierGraphStructure::MAX_CORE;
  } else {
    // MAX_CLIQUE
    gs = robin::InlierGraphStructure::MAX_CLIQUE;
  }

  // Start the timer
  auto start = std::chrono::high_resolution_clock::now();

  // call robin API
  mexPrintf("Calling graph structure solver.\n");
  mexEvalString("drawnow;");

  auto indices = robin::FindInlierStructure(g, gs);
  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] +=1;
  }

  // Stop the timer
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  double duration_in_milliseconds = static_cast<double>(duration.count()) / 1000.0;
  double duration_in_seconds = duration_in_milliseconds / 1000.0;

  mexPrintf("Graph structure found.\n");
  mexEvalString("drawnow;");

  // return to matlab
  // populate the indices
  plhs[toUType(OUTPUT_PARAMS::vertices)] = mxCreateDoubleMatrix(1, indices.size(), mxREAL);
  for (size_t i = 0; i < indices.size(); ++i) {
    mxGetPr(plhs[toUType(OUTPUT_PARAMS::vertices)])[i] = static_cast<double>(indices[i]);
  }
  // populate the runtime
  plhs[toUType(OUTPUT_PARAMS::time_taken)] = mxCreateDoubleScalar(duration_in_seconds);
}