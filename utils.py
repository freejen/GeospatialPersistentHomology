import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import networkx as nx


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


def persistance_1D_with_loops(simplex_tree):
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

    print("Assumption 2: No 2 edges have the same length.")
    print("\tsatisified:", len(set(edge_distances)) == len(edge_distances))
    print()

    edge_by_distance = {dist: edge for (edge, dist) in edges}

    graph = nx.Graph()
    loops = []
    for birth_time in homology_birth_times:
        v1, v2 = edge_by_distance[birth_time]
        graph.add_weighted_edges_from([(v1, v2, dist) for (
            (v1, v2), dist) in edges if dist < birth_time])

        loop = nx.shortest_path(graph, source=v1, target=v2, weight="weight")
        loops.append(loop)

    # Isfiltriram feature koji kratko zive
    max_life = max(death - birth for (birth, death) in persistance1)
    long_lived = [(death - birth) / max_life >=
                  0.75 for (birth, death) in persistance1]
    print(sum(long_lived))

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
