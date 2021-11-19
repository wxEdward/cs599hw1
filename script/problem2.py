#!/usr/bin/env python
# coding: utf-8

# ## Problem 2

# In[1]:


import networkx as nx
import numpy as np
import pandas as pd
import time
import tracemalloc


# ## Load ca-GrQc

# In[16]:


def load_ca_grqc():
    data = pd.read_csv('CA-GrQc.txt', sep="\t", skiprows=3)
    records = data.to_records(index=False)
    edges = list(records)
    G = nx.DiGraph()
    G.add_edges_from(edges[:5000])
    
    return G


# ## Color-coding

# In[8]:


def Chiba_Nishizeki(G):
    nodes_degree_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
    nodes_sorted = [x[0] for x in nodes_degree_sorted]
    num_nodes = len(G.nodes)
    mark_dict = dict(zip(nodes_sorted,np.zeros(num_nodes)))
    triangle_list = []
    
    for i in range(num_nodes-2):
        v = nodes_sorted[i]
        for u in G.neighbors(v):
            mark_dict[u] = 1  
        for u in G.neighbors(v):
            for w in G.neighbors(u):
                if(w!=v and mark_dict[w]==1):
                    tri = {v,u,w}
                    if(len(tri)==3):
                        triangle_list.append(tri)
            mark_dict[u] = 0   
        G.remove_node(v)
        
    return triangle_list


# In[9]:


def color_coding(G,k):
    num_nodes = len(G.nodes)
    nodes_degree_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
    nodes_sorted = [x[0] for x in nodes_degree_sorted]
    
    colors = np.arange(1,3*k+1);
    #random coloring V -> [3k]
    labels = np.random.randint(1,3*k+1, size=num_nodes)
    colors_dict = dict(zip(nodes_sorted,labels))
    
    #List all triangles
    triangles = Chiba_Nishizeki(G)
    
    #List all base triangles, i.e., triangle whose nodes are of different colors
    #Only record the color triples instead of the node triples
    base_triangles = []
    for triangle in triangles:
        u,v,w = triangle
        X_t = {colors_dict[u],colors_dict[v],colors_dict[w]}
        if(len(X_t)==3):
            base_triangles.append(X_t)

    #Eliminate duplication of color triples
    no_dup = []
    for i in base_triangles:
        if i not in no_dup:
            no_dup.append(i)
    
    #Define complex triangles as subsets of colors with size 3i, i is an integer
    #We grow size of subsets of colors by 3 iteratively
    complex_triangles = no_dup
    for i in np.arange(2,k+1):
        new_complex = []
        for t_1 in no_dup:
            for t_2 in complex_triangles:
                t_3 = t_1.union(t_2)
                if(len(t_3)==3*i and t_3 not in new_complex):
                    new_complex.append(t_3)
        complex_triangles = new_complex
    
    #if we finally get the color set with size 3*k, we have k node disjoint triangles
    if(len(complex_triangles)>0):
        return 1
    else:
        return 0

# In[132]:


if __name__ == "__main__":
    G = load_ca_grqc()
    times = []
    memory = []
    output = []
    
    ks = [5,10,15,20]
    for k in ks:
        sub_time = []
        sub_memory = []
        sub_output = []
        for _ in range(5):
            start_time = time.time()
            tracemalloc.start()
            output.append(color_coding(G,5))
            sub_time.append(time.time() - start_time)
            sub_memory.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
        time.append(sub_time)
        memory.append(sub_memory)
        output.append(sub_output)
        print("k=",k,":")
        print("average time cost:",np.mean(sub_time))
        print("total time cost:",np.sum(sub_time))
        print("average memory usage:",np.mean(sub_memory))
        print("total number of success:",np.sum(sub_output))


