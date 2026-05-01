// Copyright (c) 2020, Massachusetts Institute of Technology,
// Cambridge, MA 02139
// All Rights Reserved
// Authors: Jingnan Shi, et al. (see THANKS for the full author list)
// See LICENSE for the license information

#include "catch.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <iterator>
#include <omp.h>
#include <set>

#include <robin/graph.hpp>
#include <robin/graph_io.hpp>
#include <robin/utils.hpp>

TEST_CASE("CSR from edges") {
  {
    // 3-clique
    std::vector<std::pair<size_t, size_t>> edge_list;
    edge_list.push_back({0, 1});
    edge_list.push_back({1, 2});
    edge_list.push_back({0, 2});

    robin::CSRGraph csr_graph(edge_list);

    REQUIRE(csr_graph.VertexCount() == 3);
    REQUIRE(csr_graph.EdgeCount() == 3);

    for (size_t i = 0; i < 2; ++i) {
      REQUIRE(csr_graph.GetVertexDegree(i) == 2);
    }

    // check vertex 0 edges
    auto v0_deg = csr_graph.GetVertexDegree(0);
    std::set<size_t> v0_edge_set;
    for (size_t i = 0; i < v0_deg; ++i) {
      auto e = csr_graph.GetVertexEdge(0, i);
      v0_edge_set.insert(e);
    }
    REQUIRE(v0_edge_set.count(1) == 1);
    REQUIRE(v0_edge_set.count(2) == 1);

    // check vertex 1 edges
    auto v1_deg = csr_graph.GetVertexDegree(1);
    std::set<size_t> v1_edge_set;
    for (size_t i = 0; i < v1_deg; ++i) {
      auto e = csr_graph.GetVertexEdge(1, i);
      v1_edge_set.insert(e);
    }
    REQUIRE(v1_edge_set.count(0) == 1);
    REQUIRE(v1_edge_set.count(2) == 1);

    // check vertex 2 edges
    auto v2_deg = csr_graph.GetVertexDegree(2);
    std::set<size_t> v2_edge_set;
    for (size_t i = 0; i < v2_deg; ++i) {
      auto e = csr_graph.GetVertexEdge(2, i);
      v2_edge_set.insert(e);
    }
    REQUIRE(v2_edge_set.count(0) == 1);
    REQUIRE(v2_edge_set.count(1) == 1);
  }
  {
    // star graph
    std::vector<std::pair<size_t, size_t>> edge_list;
    edge_list.push_back({0, 1});
    edge_list.push_back({0, 2});
    edge_list.push_back({0, 3});
    edge_list.push_back({0, 4});
    edge_list.push_back({0, 5});

    robin::CSRGraph csr_graph(edge_list);
    REQUIRE(csr_graph.VertexCount() == 6);
    REQUIRE(csr_graph.EdgeCount() == 5);

    REQUIRE(csr_graph.GetVertexDegree(0) == 5);
    for (size_t i = 1; i < 6; ++i) {
      REQUIRE(csr_graph.GetVertexDegree(i) == 1);
    }

    // check vertex 0 edges
    // auto v0_edges = csr_graph.GetVertexEdges(0);
    auto v0_deg = csr_graph.GetVertexDegree(0);
    std::set<size_t> v0_edge_set;
    for (size_t i = 0; i < v0_deg; ++i) {
      auto e = csr_graph.GetVertexEdge(0, i);
      v0_edge_set.insert(e);
    }
    REQUIRE(v0_edge_set.count(0) == 0);
    REQUIRE(v0_edge_set.count(1) == 1);
    REQUIRE(v0_edge_set.count(2) == 1);
    REQUIRE(v0_edge_set.count(3) == 1);
    REQUIRE(v0_edge_set.count(4) == 1);
    REQUIRE(v0_edge_set.count(5) == 1);

    // check the other edges
    for (size_t v = 1; v < 6; ++v) {
      auto v_deg = csr_graph.GetVertexDegree(v);
      std::set<size_t> v_edges_set;
      for (size_t i = 0; i < v_deg; ++i) {
        auto e = csr_graph.GetVertexEdge(v, i);
        v_edges_set.insert(e);
      }
      REQUIRE(v_edges_set.count(0) == 1);
    }
  }
}

TEST_CASE("small CSR graph") {
  // test a K_3 graph
  size_t vertex_count = 3;
  size_t* offsets = new size_t[vertex_count + 1];
  // because this is an undirected graph, we store every edge 2 times
  size_t* edges = new size_t[vertex_count * 2];
  // populate offsets and edges array
  offsets[0] = 0;
  offsets[1] = 2;
  offsets[2] = 4;
  offsets[3] = 6;
  edges[0] = 1;
  edges[1] = 2;
  edges[2] = 0;
  edges[3] = 2;
  edges[4] = 0;
  edges[5] = 1;

  robin::CSRGraph csr_graph;
  csr_graph.Allocate(3, 3);
  csr_graph.CopyOffsetsFrom(offsets);
  csr_graph.CopyEdgesFrom(edges);

  // check vertex and edge attributes
  for (size_t i = 0; i < vertex_count; ++i) {
    auto v_deg = csr_graph.GetVertexDegree(i);
    REQUIRE(v_deg == 2);
  }

  // check vertex 0 edges
  auto v0_deg = csr_graph.GetVertexDegree(0);
  std::set<size_t> v0_edge_set;
  for (size_t i = 0; i < v0_deg; ++i) {
    auto e = csr_graph.GetVertexEdge(0, i);
    v0_edge_set.insert(e);
  }
  REQUIRE(v0_edge_set.count(1) == 1);
  REQUIRE(v0_edge_set.count(2) == 1);

  // check vertex 1 edges
  auto v1_deg = csr_graph.GetVertexDegree(1);
  std::set<size_t> v1_edge_set;
  for (size_t i = 0; i < v1_deg; ++i) {
    auto e = csr_graph.GetVertexEdge(1, i);
    v1_edge_set.insert(e);
  }
  REQUIRE(v1_edge_set.count(0) == 1);
  REQUIRE(v1_edge_set.count(2) == 1);

  // check vertex 2 edges
  auto v2_deg = csr_graph.GetVertexDegree(2);
  std::set<size_t> v2_edge_set;
  for (size_t i = 0; i < v2_deg; ++i) {
    auto e = csr_graph.GetVertexEdge(2, i);
    v2_edge_set.insert(e);
  }
  REQUIRE(v2_edge_set.count(0) == 1);
  REQUIRE(v2_edge_set.count(1) == 1);

  delete[] offsets;
  delete[] edges;
}

TEST_CASE("graph methods") {
  robin::AdjListGraph g;

  g.AddVertex(0);
  g.AddVertex(1);
  g.AddVertex(2);
  g.AddVertex(3);
  g.AddEdge(0, 1);
  g.AddEdge(0, 2);
  g.AddEdge(2, 3);

  // graph g now has 4 vertices and 3 edges
  REQUIRE(g.VertexCount() == 4);
  REQUIRE(g.EdgeCount() == 3);

  SECTION("add vertices") {
    g.AddVertex(4);
    REQUIRE(g.VertexCount() == 5);
    g.AddVertex(5);
    REQUIRE(g.VertexCount() == 6);

    // check vertex existences
    REQUIRE(g.HasVertex(4));
    REQUIRE(g.HasVertex(5));
    REQUIRE_FALSE(g.HasVertex(100));
  }
  SECTION("add edges") {
    // add new edge
    g.AddEdge(0, 3);
    REQUIRE(g.HasEdge(0, 3));
    REQUIRE(g.HasEdge(3, 0));

    // add new edge
    g.AddEdge(1, 3);
    REQUIRE(g.HasEdge(1, 3));
    REQUIRE(g.HasEdge(3, 1));

    // add edge to non-existing vertices
    g.AddEdge(100, 3);
    REQUIRE_FALSE(g.HasEdge(100, 3));

    // add existing edge
    g.AddEdge(1, 3);
    REQUIRE(g.HasEdge(1, 3));
    REQUIRE(g.HasEdge(3, 1));

    // check non-existence of an edge
    REQUIRE_FALSE(g.HasEdge(100, 1));
  }
  SECTION("remove edges") {
    // remove existing edge
    REQUIRE(g.HasEdge(0, 1));
    g.RemoveEdge(0, 1);
    REQUIRE_FALSE(g.HasEdge(0, 1));

    // remove existing edge
    REQUIRE(g.HasEdge(0, 2));
    g.RemoveEdge(0, 2);
    REQUIRE_FALSE(g.HasEdge(0, 2));

    // remove non-existing edge
    REQUIRE_FALSE(g.HasEdge(100, 2));
    g.RemoveEdge(100, 2);
    REQUIRE_FALSE(g.HasEdge(100, 2));
  }
  SECTION("check vertex degrees") {
    REQUIRE(g.GetVertexDegree(0) == 2);
    REQUIRE(g.GetVertexDegree(1) == 1);
    REQUIRE(g.GetVertexDegree(2) == 2);
    REQUIRE(g.GetVertexDegree(3) == 1);
  }
  SECTION("clear") {
    g.Clear();
    REQUIRE(g.EdgeCount() == 0);
    REQUIRE(g.VertexCount() == 0);
  }
  SECTION("adjacency matrix") {
    // 3 edges
    Eigen::MatrixXd exp_adj_mat(4, 4);
    // clang-format off
    exp_adj_mat << 0, 1, 1, 0,
                   1, 0, 0, 0,
                   1, 0, 0, 1,
                   0, 0, 1, 0;
    // clang-format on
    auto act_adj_mat = g.GetAdjMat();
    REQUIRE(act_adj_mat.isApprox(exp_adj_mat));

    // 5 edges
    g.AddEdge(0, 3);
    g.AddEdge(1, 3);
    // clang-format off
    exp_adj_mat << 0, 1, 1, 1,
                   1, 0, 0, 1,
                   1, 0, 0, 1,
                   1, 1, 1, 0;
    // clang-format on
    act_adj_mat = g.GetAdjMat();
    REQUIRE(act_adj_mat.isApprox(exp_adj_mat));
  }
}

TEST_CASE("CSR from edges to edges") {
  {
    // 3-clique
    std::vector<std::pair<size_t, size_t>> edge_list;
    edge_list.push_back({0, 1});
    edge_list.push_back({1, 2});
    edge_list.push_back({0, 2});
    std::set<std::pair<size_t, size_t>> expected_edge_set(edge_list.begin(), edge_list.end());

    robin::CSRGraph csr_graph(edge_list);

    REQUIRE(csr_graph.VertexCount() == 3);
    REQUIRE(csr_graph.EdgeCount() == 3);

    auto output_edge_list = csr_graph.ToEdgeList();
    std::set<std::pair<size_t, size_t>> output_edge_set(output_edge_list.begin(),
                                                        output_edge_list.end());

    REQUIRE(output_edge_list.size() == 3);
    for (const auto& e : expected_edge_set) {
      auto count1 = output_edge_set.count({e.first, e.second});
      auto count2 = output_edge_set.count({e.second, e.first});
      bool flag1 = count1 == 1;
      bool flag2 = count2 == 1;
      REQUIRE((count1 == 1 || count2 == 1));
      // xor
      REQUIRE(flag1 != flag2);
    }
  }
  {
    // star graph
    std::vector<std::pair<size_t, size_t>> edge_list;
    edge_list.push_back({0, 1});
    edge_list.push_back({0, 2});
    edge_list.push_back({0, 3});
    edge_list.push_back({0, 4});
    edge_list.push_back({0, 5});
    std::set<std::pair<size_t, size_t>> expected_edge_set(edge_list.begin(), edge_list.end());

    robin::CSRGraph csr_graph(edge_list);
    REQUIRE(csr_graph.VertexCount() == 6);
    REQUIRE(csr_graph.EdgeCount() == 5);
    auto output_edge_list = csr_graph.ToEdgeList();
    std::set<std::pair<size_t, size_t>> output_edge_set(output_edge_list.begin(),
                                                        output_edge_list.end());

    REQUIRE(output_edge_list.size() == 5);
    for (const auto& e : expected_edge_set) {
      auto count1 = output_edge_set.count({e.first, e.second});
      auto count2 = output_edge_set.count({e.second, e.first});
      bool flag1 = count1 == 1;
      bool flag2 = count2 == 1;
      REQUIRE((count1 == 1 || count2 == 1));
      // xor
      REQUIRE(flag1 != flag2);
    }
  }
}
TEST_CASE("csr to CSR arrays") {
  // test a K_3 graph
  size_t vertex_count = 3;
  size_t* offsets = new size_t[vertex_count + 1];
  // because this is an undirected graph, we store every edge 2 times
  size_t* edges = new size_t[vertex_count * 2];
  // populate offsets and edges array
  offsets[0] = 0;
  offsets[1] = 2;
  offsets[2] = 4;
  offsets[3] = 6;
  edges[0] = 1;
  edges[1] = 2;
  edges[2] = 0;
  edges[3] = 2;
  edges[4] = 0;
  edges[5] = 1;

  robin::CSRGraph csr_graph;
  csr_graph.Allocate(3, 3);
  csr_graph.CopyOffsetsFrom(offsets);
  csr_graph.CopyEdgesFrom(edges);

  auto output_csr_arrays = csr_graph.ToCSRArrays();
  REQUIRE(output_csr_arrays.first.size() == 4);
  REQUIRE(output_csr_arrays.second.size() == 6);
  // check for offsets array
  for (size_t i = 0; i < output_csr_arrays.first.size(); ++i) {
    REQUIRE(output_csr_arrays.first[i] == offsets[i]);
  }
  // check for edges array
  for (size_t i = 0; i < output_csr_arrays.second.size(); ++i) {
    REQUIRE(output_csr_arrays.second[i] == edges[i]);
  }

  delete[] offsets;
  delete[] edges;
}

TEST_CASE("atomic csr to CSR arrays") {
  // test a K_3 graph
  size_t vertex_count = 3;
  auto* offsets = new std::atomic<size_t>[vertex_count + 1];
  // because this is an undirected graph, we store every edge 2 times
  auto* edges = new std::atomic<size_t>[vertex_count * 2];
  // populate offsets and edges array
  offsets[0] = 0;
  offsets[1] = 2;
  offsets[2] = 4;
  offsets[3] = 6;
  edges[0] = 1;
  edges[1] = 2;
  edges[2] = 0;
  edges[3] = 2;
  edges[4] = 0;
  edges[5] = 1;

  robin::AtomicCSRGraph csr_graph;
  csr_graph.Allocate(3, 3);
  csr_graph.CopyOffsetsFrom(offsets);
  csr_graph.CopyEdgesFrom(edges);

  auto output_csr_arrays = csr_graph.ToCSRArrays();
  REQUIRE(output_csr_arrays.first.size() == 4);
  REQUIRE(output_csr_arrays.second.size() == 6);
  // check for offsets array
  for (size_t i = 0; i < output_csr_arrays.first.size(); ++i) {
    REQUIRE(output_csr_arrays.first[i] == offsets[i]);
  }
  // check for edges array
  for (size_t i = 0; i < output_csr_arrays.second.size(); ++i) {
    REQUIRE(output_csr_arrays.second[i] == edges[i]);
  }

  delete[] offsets;
  delete[] edges;
}
