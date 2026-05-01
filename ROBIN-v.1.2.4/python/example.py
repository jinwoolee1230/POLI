import spark_robin

if __name__ == "__main__":
    print("Examples showing usage of robin_py")

    # creating a Graph in robin
    g = spark_robin.AdjListGraph()

    for i in range(10):
        g.AddVertex(i)
        g.AddVertex(i+10)
        g.AddEdge(i, i+10)

    # find the corresponding inlier structures
    max_core_indices = spark_robin.FindInlierStructure(
        g, spark_robin.InlierGraphStructure.MAX_CORE
    )
    max_clique_indices = spark_robin.FindInlierStructure(
        g, spark_robin.InlierGraphStructure.MAX_CLIQUE
    )



