"""
===============================================================================
Topological Data Analysis Visualization Utilities
===============================================================================

This Python file provides tools for visualizing Vietoris–Rips (VR) complexes 
and persistence barcodes from point clouds and weighted graphs. It is designed 
to support Topological Data Analysis (TDA) workflows and educational visualizations.

-------------------------------------------------------------------------------
Main Functionalities
-------------------------------------------------------------------------------

1. plot_barcode
   - Plots persistence barcodes from a list of (dimension, (birth, death)) tuples.
   - Supports multiple homology dimensions (H₀, H₁, ...).
   - Truncates death times at a specified scale `r` for animation consistency.
   - Displays different homology dimensions using color-coded bars.
   - H₀ bars are stacked bottom-up; H₁ bars are stacked top-down.

2. plot_persistence_diagram
   - Plots the persistence diagram as a scatter plot.
   - Supports truncation of death values using `r`.
   - Adds a diagonal reference line.
   - Color-coded points for different homology dimensions (H₀, H₁, ...).

3. generate_polygon_positions
   - Generates 2D coordinates for nodes arranged uniformly in a circle.
   - Useful for laying out graphs (e.g., toy weighted graphs) in a symmetric way.

4. plot_VR
   - Plots the Vietoris–Rips (VR) complex built from a point cloud or distance matrix.
   - Supports both 2D and 3D input.
   - Draws vertices, edges (1-simplices), and triangles (2-simplices).
   - Optionally computes and displays Betti numbers.
   - Accepts a precomputed simplex tree to save recomputation.

5. plot_weighted_graph
   - Visualizes a small weighted graph using a circular layout (uses a helper function generate_polygon_positions)
   - Displays edge weights as text labels along edges.
   - Suitable for simple graph sanity checks or teaching examples.

6. plot_VR_from_graph
   - Computes and visualizes the VR complex from a weighted undirected graph.
   - Uses NetworkX graph object with weights interpreted as distances.
   - Converts graph to a distance matrix and applies GUDHI's RipsComplex.
   - Displays 1- and 2-simplices, edge weights, and node labels.
   - Optionally computes and displays Betti numbers.


-------------------------------------------------------------------------------
Dependencies
-------------------------------------------------------------------------------

- numpy
- matplotlib
- gudhi
- networkx
- IPython.display (Jupyter notebook display utilities)
- ipywidgets (for interactivity)

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation, HTMLWriter
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import gudhi as gd
import networkx as nx

from IPython.display import display, IFrame, HTML
import ipywidgets as widgets


def plot_barcode(diag, r = 5, dims=[0, 1], figsize=(8, 6), bar_height=0.1, pad=0.05, ax=None, colors=None):
    """
    Plot the persistence barcode from a list of (dimension, (birth, death)) tuples.

    Parameters:
    - diag: list of tuples (dim, (birth, death)) representing the persistence diagram
    - r: filtration value at which to truncate bars (default: 5)
    - dims: list of homology dimensions to include (default: [0, 1])
    - figsize: size of the figure if ax is None (default: (8, 6))
    - bar_height: vertical thickness of each bar (default: 0.1)
    - pad: vertical spacing between bars (default: 0.05)
    - ax: matplotlib axis object to plot on (if None, a new figure is created)
    - colors: dictionary mapping dimension to color; defaults to {0: 'tomato', 1: 'cornflowerblue', 2: 'purple'}

    Returns:
    - None
    """
    if colors is None:
        colors = {0: 'tomato', 1: 'cornflowerblue', 2: 'purple'}

    # Filter and organize bars by dimension
    dim_bars = {}
    for dim in dims:
        bars = np.array([pair for d, pair in diag if d == dim])
        bars = np.array([[b, min(d, r)] for b, d in bars if b < r])
        dim_bars[dim] = bars

    total_bars = sum(len(b) for b in dim_bars.values())
    if total_bars == 0:
        return

    # Compute y-positions
    y_range = np.arange(total_bars) * (bar_height + pad)
    y_bottom = 0
    y_top = y_range[-1] + bar_height

    y_positions = []
    bar_data = []

    # Stack H0 from bottom up
    offset = 0
    for bd in dim_bars.get(0, []):
        y = offset * (bar_height + pad)
        y_positions.append(y)
        bar_data.append((0, bd))
        offset += 1

    # Stack H1 from top down
    offset = 0
    for bd in dim_bars.get(1, []):
        y = y_top - (offset + 1) * (bar_height + pad)
        y_positions.append(y)
        bar_data.append((1, bd))
        offset += 1

    # Other dims (optional: add similar logic)
    for dim in dims:
        if dim not in [0, 1]:
            for bd in dim_bars.get(dim, []):
                y = offset * (bar_height + pad)
                y_positions.append(y)
                bar_data.append((dim, bd))
                offset += 1

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot bars
    for (dim, (b, d)), y in zip(bar_data, y_positions):
        ax.plot([b, d], [y, y], color=colors.get(dim, 'black'), linewidth=2)

    # Axes formatting
    ax.set_yticks([])
    ax.set_ylim(-bar_height, y_top)
    ax.set_xlim(0, max(r, np.max([d for (_, (b, d)) in bar_data])) * 1.05)
    ax.set_xlabel("Filtration Value")
    ax.set_title("Persistence Barcode")

    # Legend
    legend_elements = [Patch(facecolor=colors[dim], edgecolor='none', label=f"H{dim}") for dim in dims if dim in colors]
    ax.legend(handles=legend_elements, loc='upper right')

def plot_persistence_diagram(diag, dims=[0, 1], r=5, ax=None, figsize=(6, 6), colors=None):
    """
    Plot the persistence diagram from a list of (dimension, (birth, death)) tuples.

    Parameters:
    - diag: list of tuples (dim, (birth, death)) from GUDHI
    - dims: list of dimensions to include in the plot (default: [0, 1])
    - r: truncation value for death times (useful for visualization only)
    - ax: matplotlib axis object (if None, a new figure is created)
    - figsize: size of the figure if ax is None
    - colors: dictionary mapping dimension to color (optional)

    Returns:
    - None
    """
    if colors is None:
        colors = {0: 'tomato', 1: 'cornflowerblue', 2: 'purple'}

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for dim in dims:
        births_deaths = np.array([pair for d, pair in diag if d == dim])
        if len(births_deaths) == 0:
            continue
        births = births_deaths[:, 0]
        deaths = np.minimum(births_deaths[:, 1], r)  # truncate at r
        ax.scatter(births, deaths, label=f"H{dim}", color=colors.get(dim, 'black'), alpha=0.7, edgecolors='k', s=40)

    # Plot diagonal
    ax.plot([0, r], [0, r], 'k--', linewidth=1, alpha=0.5)

    ax.set_xlim(-0.1, r * 1.05)
    ax.set_ylim(-0.1, r * 1.05)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title("Persistence Diagram")
    ax.legend()
    ax.set_aspect('equal')


def generate_polygon_positions(num_sides, radius=1):
    """
    Generate the positions of the nodes of a regular polygon.
    
    Parameters:
    num_sides (int): Number of sides (and nodes) of the polygon.
    radius (float): Radius of the circumscribed circle. Default is 1.
    
    Returns:
    np.ndarray: Array of shape (num_sides, 2) containing the (x, y) positions of the nodes.
    """
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    positions = np.array([(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles])
    return positions





def plot_VR(X, r, ax=None, Betti_Seq=False, max_dimension=2, frame=True, distance_matrix=False, simplex_tree_given=None):
    """
    INPUT:  
    X : Nx2 or Nx3 array or distance matrix. If distance_matrix=True, X is interpreted as a distance matrix.
    r : VR complex scale parameter.
    ax : matplotlib axis. If None, one will be created.
    Betti_Seq : whether to print Betti numbers.
    max_dimension : maximum dimension of the simplices.
    frame : whether to draw axis frame.
    distance_matrix : True if X is a distance matrix.
    simplex_tree_given : optional precomputed gudhi.SimplexTree
    """

    ############################# STEP 1: Compute simplex_tree if needed #########################
    if distance_matrix:
        DM_X = X
        n = DM_X.shape[0]
        X_plot = generate_polygon_positions(n)  # for display only
        if simplex_tree_given is None:
            rips_complex = gd.RipsComplex(distance_matrix=DM_X, max_edge_length=r)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
        else:
            simplex_tree = simplex_tree_given
        X = X_plot  # only used for plotting
    else:
        if simplex_tree_given is None:
            rips_complex = gd.RipsComplex(points=X, max_edge_length=r)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
        else:
            simplex_tree = simplex_tree_given

    ############################# STEP 2: Filter simplices #########################
    if simplex_tree_given is None:
        edges = list(simplex_tree.get_skeleton(1))
        triangles = list(simplex_tree.get_skeleton(2))
    else:
        edges = [edg for edg in simplex_tree.get_skeleton(1) if edg[1] <= r]
        triangles = [tri for tri in simplex_tree.get_skeleton(2) if tri[1] <= r]

    ############################# STEP 3: Optional Betti numbers #########################
    if Betti_Seq:        
        simplex_tree.persistence(homology_coeff_field=2, min_persistence=0.001)
        betti_numbers = simplex_tree.betti_numbers()
        print("Betti Numbers:", betti_numbers, f'of the VR complex at scale r = {r}')

    ############################# STEP 4: Plot #########################
    is_3d = X.shape[1] == 3

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

    if is_3d:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue', zorder=2, depthshade=False, s=10)
    else:
        ax.scatter(X[:, 0], X[:, 1], c='blue', zorder=2,s =10)
      


    # Plot edges (1-simplices)
    for simplex in edges:
        if len(simplex[0]) == 2:  # Only plot edges (1-simplices)
            i, j = simplex[0]
            if is_3d:
                ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], [X[i, 2], X[j, 2]], 'k-', zorder=1, linewidth=0.5)
            else:
                ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'k-', zorder=1)

    # Plot triangles (2-simplices)
    if max_dimension >= 2:
        for simplex in triangles:
            if len(simplex[0]) == 3:  # Only plot triangles (2-simplices)
                i, j, k = simplex[0]
                triangle = [X[i], X[j], X[k]]
                if is_3d:
                    tri = Poly3DCollection(
                        [triangle], alpha=0.08, facecolor='gold', edgecolor='gray', linewidths=0.1
                    )
                    ax.add_collection3d(tri)
                else:
                    poly = plt.Polygon(triangle, alpha=0.1, facecolor='gold', edgecolor='gray')
                    ax.add_patch(poly)

    ax.set_title(f'VR - complex with $r$ = {r}')
    ax.set_xticks([]), ax.set_yticks([])
    if is_3d:
        ax.set_zticks([])
    if not frame:
        ax.set_frame_on(False)


def plot_weighted_graph(edges, ax=None):
    """
    Manually plots a weighted graph using circular polygon layout.

    Parameters:
    - edges: (weighted) array-like, each element is [i, j, weight]
    - ax: matplotlib axis (optional)
    """
    edges = np.array(edges)
    nodes = np.unique(edges[:, :2].astype(int))
    n = len(nodes)
    positions = generate_polygon_positions(n, radius = 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Plot edges with weights
    for i, j, w in edges:
        i, j = int(i), int(j)
        x_coords = [positions[i][0], positions[j][0]]
        y_coords = [positions[i][1], positions[j][1]]
        ax.plot(x_coords, y_coords, 'k-', linewidth=1)
        
        # Label edge at midpoint
        mid_x = (x_coords[0] + x_coords[1]) / 2
        mid_y = (y_coords[0] + y_coords[1]) / 2
        ax.text(mid_x, mid_y, f"{w:.2f}", fontsize=9, ha='center', va='top')

    # Plot nodes and labels
    ax.scatter(positions[:, 0], positions[:, 1], s=400, zorder=2, edgecolor='k')
    for i, (x, y) in enumerate(positions):
        ax.text(x, y, str(i), fontsize=12, weight='bold', ha='center', va='center')

    ax.set_aspect('equal')
    ax.set_title("Weighted Graph")
    ax.axis('off')


def plot_VR_from_graph(graph, r, ax=None, Betti_Seq=False, max_dimension=2, frame=True, simplex_tree_given=None):
    """
    Visualizes the Vietoris-Rips complex from a weighted undirected graph (2D only).
    
    Parameters:
    - graph: networkx.Graph with 'weight' edge attributes as distances
    - r: VR complex scale parameter
    - ax: matplotlib axis (optional)
    - Betti_Seq: whether to print Betti numbers
    - max_dimension: maximum simplex dimension
    - frame: whether to draw axis frame
    - simplex_tree_given: precomputed gudhi.SimplexTree (optional)
    """
    # Step 1: Convert to distance matrix and polygon layout
    nodes = list(graph.nodes())
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    DM_X = np.full((n, n), fill_value=np.inf)
    np.fill_diagonal(DM_X, 0.0)

    for u, v, data in graph.edges(data=True):
        i, j = index[u], index[v]
        w = data.get('weight', 1.0)
        DM_X[i, j] = DM_X[j, i] = w

    X = generate_polygon_positions(n, radius=1.0)

    # Step 2: Compute simplex tree if needed
    if simplex_tree_given is None:
        rips_complex = gd.RipsComplex(distance_matrix=DM_X, max_edge_length=r)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    else:
        simplex_tree = simplex_tree_given

    # Step 3: Filter simplices
    if simplex_tree_given is None:
        edges = list(simplex_tree.get_skeleton(1))
        triangles = list(simplex_tree.get_skeleton(2))
    else:
        edges = [e for e in simplex_tree.get_skeleton(1) if e[1] <= r]
        triangles = [t for t in simplex_tree.get_skeleton(2) if t[1] <= r]

    # Step 4: Betti numbers
    if Betti_Seq:
        simplex_tree.persistence(homology_coeff_field=2, min_persistence=0.001)
        betti_numbers = simplex_tree.betti_numbers()
        print("Betti Numbers:", betti_numbers, f'of the VR complex at scale r = {r}')

    # Step 5: Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(X[:, 0], X[:, 1], s=400, zorder=2, edgecolor='k')

    for simplex in edges:
        if len(simplex[0]) == 2:
            i, j = simplex[0]
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'k-', linewidth=1)
            # Label edge weights
            for u, v, data in graph.edges(data=True):
                if set([index[u], index[v]]) == set([i, j]):
                    midx = (X[i, 0] + X[j, 0]) / 2
                    midy = (X[i, 1] + X[j, 1]) / 2
                    ax.text(midx, midy, f"{data['weight']:.2f}", fontsize=9, ha='center', va='top')

    if max_dimension >= 2:
        for simplex in triangles:
            if len(simplex[0]) == 3:
                i, j, k = simplex[0]
                triangle = [X[i], X[j], X[k]]
                poly = plt.Polygon(triangle, alpha=0.1, facecolor='gold', edgecolor='gray')
                ax.add_patch(poly)

    for i, (x, y) in enumerate(X):
        ax.text(x, y, str(i), fontsize=12, weight='bold', ha='center', va='center')

    ax.set_title(f'VR - complex with $r$ = {r}')
    ax.set_xticks([]), ax.set_yticks([])
    if not frame:
        ax.set_frame_on(False)



    