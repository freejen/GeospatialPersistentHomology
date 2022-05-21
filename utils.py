import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


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
