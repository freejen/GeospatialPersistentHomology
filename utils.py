import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import networkx as nx
import gudhi


def load_data(county='025-imperial'):
    SHAPE_PATH = "dataset/shapefiles/" + county + ".shp"
    VOTES_PATH = "dataset/final-results/" + county + ".csv"

    shapes_gdf = gpd.read_file(SHAPE_PATH)
    votes_df = pd.read_csv(VOTES_PATH)

    shapes_gdf = shapes_gdf.set_index('pct16').drop('area', axis=1)
    shapes_gdf['centroid'] = shapes_gdf.to_crs(
        'epsg:3785').centroid.to_crs(shapes_gdf.crs)

    votes_df = votes_df.set_index('pct16')[['pres_clinton', 'pres_trump']]
    gdf = shapes_gdf.join(votes_df)
    gdf = gdf.rename(
        columns={'pres_clinton': 'abs_clinton', 'pres_trump': 'abs_trump'})

    gdf['per_clinton'] = gdf['abs_clinton'] / \
        (gdf['abs_clinton'] + gdf['abs_trump'])
    gdf['per_trump'] = gdf['abs_trump'] / \
        (gdf['abs_clinton'] + gdf['abs_trump'])

    return gdf


def plot_simplices(points, simplex_tree, max_epsilon):
    for vert, fval in simplex_tree.get_filtration():
        if fval >= max_epsilon:
            break
        if len(vert) == 1:
            pt = points[vert[0]]
            plt.plot([pt[0]], [pt[1]], marker='o', markersize=3, color='blue')
        elif len(vert) == 2:
            pt1 = points[vert[0]]
            pt2 = points[vert[1]]
            xs = [pt1[0], pt2[0]]
            ys = [pt1[1], pt2[1]]
            plt.plot(xs, ys, color='blue')
        else:
            pts = []
            for v in vert:
                pts.append(points[v])
            pts = np.array(pts)
            p = Polygon(pts, closed=False)
            ax = plt.gca()
            ax.add_patch(p)

    plt.show()


def persistence_1D_with_loops(simplex_tree, adjacency_complex=False):
    '''
    Returns [(birth, death), loop, long_lived]
    '''
    persistance = simplex_tree.persistence()
    persistance1 = [birth_death for (
        dim, birth_death) in persistance if dim == 1]
    persistance1.sort(key=lambda x: x[0])

    homology_birth_times = [birth for (birth, death) in persistance1]
    edges = [x for x in simplex_tree.get_filtration() if len(x[0]) == 2]
    edge_distances = [dist for (_, dist) in edges]

    print("Assumption 1: No 2 homology classes have the same birth time")
    print("\tsatisified:", len(set(homology_birth_times))
          == len(homology_birth_times))
    print()

    print("Assumption 2: No 2 edges have the same length. (Doesn't have to be satisfied.)")
    print("\tsatisified:", len(set(edge_distances)) == len(edge_distances))
    print()

    graph = nx.Graph()
    loops = []
    max_life = max(death - birth for (birth, death) in persistance1)
    long_lived = []
    for birth_time, death_time in persistance1:
        # edges that were added to the filtration at birth_time
        edges_added_now = [edge for edge in edges if edge[1] == birth_time]
        
        # To current graph add all edges that were added to the filtration 
        # before the birth time of the current homology class.
        graph.add_weighted_edges_from([(v1, v2, dist) for (
            (v1, v2), dist) in edges if dist < birth_time])
        
        # Add current edges one-by-one and check if any closes a loop
        for ((v1, v2), dist) in edges_added_now:
            try:
                loop = nx.shortest_path(graph, source=v1, target=v2, weight="weight")
                if(not adjacency_complex or len(loop) > 3):
                    loops.append(loop)
                    long_lived.append(death_time == float('inf') or (death_time - birth_time) / max_life >= 0.75)
                else:
                    raise Exception
            except:
                graph.add_weighted_edges_from([(v1, v2, dist)])
                


    return list(zip(persistance1, loops, long_lived))



def plot_loops(gdf, points, persistance_1D_with_loops):
    ax = gdf.plot("per_clinton", legend=True, cmap='RdBu',
                  figsize=(20, 12), vmin=0, vmax=1)
    loops = [x[1] for x in persistance_1D_with_loops]
    long_lived_list = [x[2] for x in persistance_1D_with_loops]
    for loop, long_lived in zip(loops, long_lived_list):
        x = [points[idx][0] for idx in loop]
        x.append(x[0])
        y = [points[idx][1] for idx in loop]
        y.append(y[0])
        linewidth = 3 if long_lived else 1
        color = 'darkred' if long_lived else 'indianred'
#         color = 'darkgreen' if long_lived else 'lightgreen'

        ax.plot(x, y, color=color, linewidth=linewidth)
    
def get_adjacency_simplex_tree(gdf, pct_id_to_vert):
    delta = 0.95
    simplex_tree = gudhi.SimplexTree()

    edges = []
    triangles = []

    while delta >= 0.5:
        curr_complex = []
        vertices = gdf[gdf['delta'] >= delta].index.tolist()
    
        # Add 0-simplices
        for v in vertices:
            # Don't add duplicates from a complex with higher delta
            vert_index = pct_id_to_vert[v]
            duplicate = False
            for simplex, fval in simplex_tree.get_filtration():
                if len(simplex) != 1:
                    continue
                if simplex[0] == vert_index:
                    duplicate = True
                    break
            if not duplicate:
                curr_complex.append([vert_index])
    
        # Find neighbors for each vertex
        neighbors_dict = {}
        tmp = gdf.loc[vertices, :]
        for index, row in tmp.iterrows():
            neighbors_dict[index] = tmp[tmp['geometry'].touches(row['geometry'])].index.tolist()
    
        # Add 1-simplices
        for key, neighbors in neighbors_dict.items():
            for val in neighbors:
                simplex = [pct_id_to_vert[val], pct_id_to_vert[key]]
                simplex.sort()
                # Don't add duplicate 1-simplices from the same complex
                if simplex in curr_complex:
                    continue
            
                # Don't add duplicates from a complex with higher delta
                duplicate = False
                for added_simplex, fval in simplex_tree.get_filtration():
                    if len(added_simplex) != 2:
                        continue
                    if np.array_equal(simplex, added_simplex):
                        duplicate = True
                        break
                if not duplicate:
                    curr_complex.append(simplex)
                    edges.append(simplex)

        for simplex in curr_complex:
            simplex_tree.insert(simplex, 1-delta)
        
        # Add 2-simplices
        edges.sort()
        for i in range(len(edges)):
            v1 = edges[i][0]
            v2 = edges[i][1]
            for j in range(i+1, len(edges)):
                foundTriangle = False
                if v1 in edges[j] or v2 in edges[j]:
                    # Found edge2 which has v1 or v2
                    edge3 = []
                
                    if v1 in edges[j]:
                        edge3.append(v2)
                        edge3.append(edges[j][1] if v1 == edges[j][0] else edges[j][0])
                    else:
                        edge3.append(v1)
                        edge3.append(edges[j][1] if v2 == edges[j][0] else edges[j][0])
                
                    edge3.sort()
                    for k in range(j+1, len(edges)):
                        if np.array_equal(edge3, edges[k]):
                            triangle = [v1, v2]
                            for v in edges[k]:
                                if v not in triangle:
                                    triangle.append(v)
                                    break
                       
                            triangle.sort()
                            foundTriangle = True
                            if triangle not in triangles:
                                triangles.append(triangle)
                                simplex_tree.insert(triangle, 1-delta)
                    
                            break
                    if foundTriangle:
                        break
   
        delta = round(delta-0.05, 2)
    
    return simplex_tree
