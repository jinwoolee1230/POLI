import sys
import scipy.io as sio
import networkx as nx


if __name__ == "__main__":

    if len(sys.argv) - 1 <= 0 :
        print("Please provide filepath as an argument")
    else:
        mtx_file = sys.argv[1]
        print("Analyzing ", mtx_file)
        mtx_matrix = sio.mmread(mtx_file)

        g = nx.Graph(mtx_matrix)
        
        max_kcore = nx.k_core(g)
        print("k core # vertices: ", max_kcore.number_of_nodes()) 

        subgraphs = nx.connected_components(max_kcore)

        for g in subgraphs:
            print("# nodes in one of the connected component: ", len(g))


    
    



