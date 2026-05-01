// Copyright (c) 2020, Massachusetts Institute of Technology,
// Cambridge, MA 02139
// All Rights Reserved
// Authors: Jingnan Shi, et al. (see THANKS for the full author list)
// See LICENSE for the license information

#include <fstream>

#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#include <robin/graph_io.hpp>

namespace robin {

AdjListGraph MatrixMarketReader::readAdjListGraphFromFile(const std::string& filepath) {
  AdjListGraph g;

  // read the second line
  std::ifstream mtx_file(filepath);
  std::string line;
  // first line
  std::getline(mtx_file, line);
  // second line
  // get the numbers separately
  std::string rows, cols, num_edges;
  mtx_file >> rows;
  mtx_file >> cols;
  mtx_file >> num_edges;

  if (std::stoi(num_edges) == 0) {
    // a graph with no edges
    int num_rows = std::stoi(rows);
    int num_cols = std::stoi(cols);
    if (num_rows != num_cols) {
      ROBIN_DEBUG_ERROR_MSG("Input matrix (matrix market format) is not square.");
      return g;
    }
    g.PopulateVertices(num_rows);
    return g;
  } else {
    // a graph with non-zero edges
    Eigen::SparseMatrix<float, Eigen::RowMajor> data;
    Eigen::loadMarket(data, filepath);
    if (data.rows() != data.cols()) {
      ROBIN_DEBUG_ERROR_MSG("Input matrix (matrix market format) is not square.");
      return g;
    }

    g.PopulateVertices(data.rows());
    for (size_t k = 0; k <data.outerSize(); ++k) {
      for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(data, k); it; ++it){
        g.AddEdge(it.col(), it.row());
      }
    }
  }
  return g;
}
}