// Copyright (c) 2020, Massachusetts Institute of Technology,
// Cambridge, MA 02139
// All Rights Reserved
// Authors: Jingnan Shi, et al. (see THANKS for the full author list)
// See LICENSE for the license information

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <robin/robin.hpp>

namespace py = pybind11;

/**
 * Python interface with pybind11
 */
PYBIND11_MODULE(spark_robin, m) {
  m.doc() = "Python binding for Robin";

  // GraphStorageType
  py::enum_<robin::GraphsStorageType>(m, "GraphStorageType")
      .value("ATOMIC_CSR", robin::GraphsStorageType::ATOMIC_CSR)
      .value("CSR", robin::GraphsStorageType::CSR)
      .value("ADJ_LIST", robin::GraphsStorageType::ADJ_LIST);

  // IGraph class (abstract virtual)
  py::class_<robin::IGraph>(m, "IGraph")
      .def("VertexCount", &robin::IGraph::VertexCount)
      .def("GetVertexDegree", &robin::IGraph::GetVertexDegree)
      .def("GetVertexEdge", &robin::IGraph::GetVertexEdge)
      .def("ToEdgeList", &robin::IGraph::ToEdgeList)
      .def("ToCSRArrays", &robin::IGraph::ToCSRArrays);

  // AdjListGraph class
  py::class_<robin::AdjListGraph, robin::IGraph>(m, "AdjListGraph")
      .def(py::init<>())
      .def("AddVertex", &robin::AdjListGraph::AddVertex)
      .def("AddEdge", &robin::AdjListGraph::AddEdge)
      .def("HasVertex", &robin::AdjListGraph::HasVertex)
      .def("HasEdge", &robin::AdjListGraph::HasEdge)
      .def("RemoveEdge", &robin::AdjListGraph::RemoveEdge)
      .def("ToCSRArrays", &robin::AdjListGraph::ToCSRArrays)
      .def("ToEdgeList", &robin::AdjListGraph::ToEdgeList)
      .def("VertexCount", &robin::AdjListGraph::VertexCount)
      .def("EdgeCount", &robin::AdjListGraph::EdgeCount)
      .def("reserve", &robin::AdjListGraph::reserve)
      .def("Clear", &robin::AdjListGraph::Clear)
      .def("Random", &robin::AdjListGraph::Random)
      .def("GetAdjMat", &robin::AdjListGraph::GetAdjMat);

  // AtomicCSRGraph class
  py::class_<robin::AtomicCSRGraph, robin::IGraph>(m, "AtomicCSRGraph")
      .def(py::init<>())
      .def("Allocate", &robin::AtomicCSRGraph::Allocate)
      .def("GetVertexOffset", &robin::AtomicCSRGraph::GetVertexOffset)
      .def("SetVertexOffset", &robin::AtomicCSRGraph::SetVertexOffset)
      .def("CopyOffsetsFrom", &robin::AtomicCSRGraph::CopyOffsetsFrom)
      .def("GetVertexEdge", &robin::AtomicCSRGraph::GetVertexEdge)
      .def("SetEdge", &robin::AtomicCSRGraph::SetEdge)
      .def("CopyEdgesFrom", &robin::AtomicCSRGraph::CopyEdgesFrom)
      .def("GetVertexDegree", &robin::AtomicCSRGraph::GetVertexDegree)
      .def("ToCSRArrays", &robin::AtomicCSRGraph::ToCSRArrays)
      .def("ToEdgeList", &robin::AtomicCSRGraph::ToEdgeList)
      .def("VertexCount", &robin::AtomicCSRGraph::VertexCount)
      .def("EdgeCount", &robin::AtomicCSRGraph::EdgeCount)
      .def("PrintEdgeArray", &robin::AtomicCSRGraph::PrintEdgeArray)
      .def("PrintOffsetsArray", &robin::AtomicCSRGraph::PrintOffsetsArray);

  // CSRGraph class
  py::class_<robin::CSRGraph, robin::IGraph>(m, "CSRGraph")
      .def(py::init<>())
      .def("Allocate", &robin::CSRGraph::Allocate)
      .def("GetVertexOffset", &robin::CSRGraph::GetVertexOffset)
      .def("SetVertexOffset", &robin::CSRGraph::SetVertexOffset)
      .def("CopyOffsetsFrom", &robin::CSRGraph::CopyOffsetsFrom)
      .def("GetVertexEdge", &robin::CSRGraph::GetVertexEdge)
      .def("SetEdge", &robin::CSRGraph::SetEdge)
      .def("CopyEdgesFrom", &robin::CSRGraph::CopyEdgesFrom)
      .def("GetVertexDegree", &robin::CSRGraph::GetVertexDegree)
      .def("ToCSRArrays", &robin::CSRGraph::ToCSRArrays)
      .def("ToEdgeList", &robin::CSRGraph::ToEdgeList)
      .def("VertexCount", &robin::CSRGraph::VertexCount)
      .def("EdgeCount", &robin::CSRGraph::EdgeCount);

  // InlierGraphStructure enum
  py::enum_<robin::InlierGraphStructure>(m, "InlierGraphStructure")
      .value("MAX_CORE", robin::InlierGraphStructure::MAX_CORE)
      .value("MAX_CLIQUE", robin::InlierGraphStructure::MAX_CLIQUE);

  // SO3 distance enum
  py::enum_<robin::So3Distance>(m, "So3Distance")
      .value("CHORDAL_DISTANCE", robin::So3Distance::CHORDAL_DISTANCE)
      .value("GEODESIC_DISTANCE", robin::So3Distance::GEODESIC_DISTANCE);

  // FindInlierStructure
  m.def("FindInlierStructure", &robin::FindInlierStructure, "Find inlier structures",
        py::arg("graph"), py::arg("graph_structure_type"));

  // MakeVecAvgInvGraph
  m.def("MakeVecAvgInvGraph", &robin::MakeVecAvgInvGraph, "Make inv graph for vec avg problems",
        py::arg("measurements"), py::arg("noise_bound"), py::arg("graph_storage_type"));

  // MakeRotAvgInvGraph
  m.def("MakeRotAvgInvGraph", &robin::MakeRotAvgInvGraph, "Make inv graph for rot avg problems",
        py::arg("measurements"), py::arg("so3_dist_type"), py::arg("noise_bound"),
        py::arg("graph_storage_type"));

  // Make3DRegInvGraph
  m.def("Make3dRegInvGraph", &robin::Make3dRegInvGraph, "Make inv graph for 3d reg problems",
        py::arg("src_3d_points"), py::arg("dst_3d_points"), py::arg("noise_bound"),
        py::arg("graph_storage_type"));

  // MakeMeshRegInvGraph
  m.def("Make3dNormalRegInvGraph", &robin::Make3dNormalRegInvGraph,
        "Make inv graph for 3d reg with normals problems", py::arg("src_3d_points_with_normals"),
        py::arg("dst_3d_points_with_normals"), py::arg("noise_bound"),
        py::arg("graph_storage_type"));

  // K-core decomposition solver
  py::class_<robin::KCoreDecompositionSolver> kcore_solver(m, "KCoreDecompositionSolver");
  py::enum_<robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE>(kcore_solver, "KCORE_SOLVER_MODE")
      .value("PKC_PARALLEL", robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::PKC_PARALLEL)
      .value("PKC_PARALLEL_OPTIMIZED",
             robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::PKC_PARALLEL_OPTIMIZED)
      .value("PKC_SERIAL", robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::PKC_SERIAL)
      .value("BZ_SERIAL", robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::BZ_SERIAL);
  kcore_solver.def(py::init<>())
      .def(py::init<const robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE&>())
      .def("Solve", &robin::KCoreDecompositionSolver::Solve)
      .def("GetCoreNumbers", &robin::KCoreDecompositionSolver::GetCoreNumbers)
      .def("GetMaxKCore", &robin::KCoreDecompositionSolver::GetMaxKCore)
      .def("GetMaxCoreNumber", &robin::KCoreDecompositionSolver::GetMaxCoreNumber)
      .def("GetKCore", &robin::KCoreDecompositionSolver::GetKCore);

  // Max clique solver
  py::class_<robin::MaxCliqueSolver> max_clique_solver(m, "MaxCliqueSolver");
  py::enum_<robin::MaxCliqueSolver::CLIQUE_SOLVER_MODE>(max_clique_solver, "CLIQUE_SOLVER_MODE")
      .value("PMC_EXACT", robin::MaxCliqueSolver::CLIQUE_SOLVER_MODE::PMC_EXACT)
      .value("PMC_HEU", robin::MaxCliqueSolver::CLIQUE_SOLVER_MODE::PMC_HEU);
  py::class_<robin::MaxCliqueSolver::Params>(max_clique_solver, "Params")
      .def(py::init<>())
      .def_readwrite("solver_mode", &robin::MaxCliqueSolver::Params::solver_mode)
      .def_readwrite("time_limit", &robin::MaxCliqueSolver::Params::time_limit);
  max_clique_solver.def(py::init<>())
      .def(py::init<const robin::MaxCliqueSolver::Params&>())
      .def("FindMaxClique", &robin::MaxCliqueSolver::FindMaxClique);
}