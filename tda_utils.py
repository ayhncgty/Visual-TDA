"""
### Explanation of the Python File

This Python file contains various functions for spike train data analysis, Victor-Purpura distance computation, and a Topological Data Analysis (TDA) pipeline for neural data classification.

#### **Main Functionalities**
1. **Victor Purpura Distance Calculation** (`VP` function)
   - Computes the Victor-Purpura (VP) spike train distance matrix for a given collection of spike trains.
   - Uses `elephant.spike_train_dissimilarity.victor_purpura_distance`.

2. **Checking for Repeated Rows in an Array** (`check_repeated_rows`)
   - Identifies duplicate rows in an `N x M` NumPy array.

3. **Generating Unique Random Matrices** (`generate_unique_random_array`)
   - Generates a unique `N x M` matrix with:
     - Non-repeating rows
     - Non-repeating values within each row

4. **TDA Pipeline for Neural Data Classification** (`TDA_Pipeline`)
   - Implements a full pipeline that:
     1. Selects trials from a given dataset.
     2. Computes Victor-Purpura distance matrices for spike trains.
     3. Extracts topological features using persistent homology.
     4. Computes bottleneck distances between barcodes.
     5. Classifies neural activity patterns using Leave-One-Out cross-validation.
   - The output is the **mean LOU classification score** over multiple iterations.

---

### **How to Import This File in Another Python Script or Jupyter Notebook**



"""
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
from sklearn.metrics import pairwise_distances
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

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



def plot_VR(X, r, ax=None, Betti_Seq=False, max_dimension=2, frame=True , distance_matrix = None):
    """
    INPUT:  
    1) X : Numpy array Nx2. Represent the original metric space. We will use the points in X as the vertex set. X could also be a distance matrix NxN. In this case
    set distance_matrix = True
    2) r : The scale of the VR complex.
    
    PARAMETERS:
    1) ax : The axis on which to plot the VR complex. If None, a new figure and axis are created.
    2) Betti_seq = False     ; If true, it will output the betti sequence of the VR - complex
    3) max_dimension = 2     ; Specifies the maximum dimensional simplex in the VR - complex
    OUTPUT: Plots the VR - complex
    4) frame = True          ; Set to false if you don't wish to have the plot framed
    """
    ############################# PRELIMINARY #########################
    # WE COMPUTE THE RIPS COMPLEX OF X FIRST
    
    if not distance_matrix:
        DM_X = pairwise_distances(X)  # Distance Matrix of X
    if distance_matrix:
        DM_X = X # Distance Matrix
        # If X is a distance matrix representing the network, the nodes in the VR - complex do not correspond to the original position.
        # Nevertheless, we want to plot nodes. So, X will be redefined as a point cloud
        n = DM_X.shape[0]
        X = generate_polygon_positions(n) 
        
    rips_complex = gd.RipsComplex(distance_matrix = DM_X, max_edge_length = r)  # This will create a RipsComplex object at scale r

    # The following is a Simplex tree object which is essentially the VR-complex
    simplex_tree = rips_complex.create_simplex_tree(max_dimension = max_dimension) 

    ############################# DISPLAY THE BETTI SEQUENCE (OPTIONAL) #########################
    if Betti_Seq:
        simplex_tree.persistence()
        betti_numbers = simplex_tree.betti_numbers()
        print("Betti Numbers:", betti_numbers, f'of the VR complex at scale r = {r}')  # displays the current betti sequence at scale r
        
    ############################# PLOT THE VR - COMPLEX #########################
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
    ax.scatter(X[:, 0], X[:, 1], c='blue', zorder=2)  # to get the vertices
    
    # Plot edges (1-simplices)
    for simplex in simplex_tree.get_skeleton(1):
        if len(simplex[0]) == 2:  # Only plot edges (1-simplices)
            i, j = simplex[0]
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'k-', zorder=1)
    
    # Plot triangles (2-simplices)
    for simplex in simplex_tree.get_skeleton(2):
        if len(simplex[0]) == 3:  # Only plot triangles (2-simplices)
            i, j, k = simplex[0]
            triangle = plt.Polygon([X[i], X[j], X[k]], alpha=0.3, color='yellow', zorder=0)
            ax.add_patch(triangle)

    ax.set_aspect('equal')
    ax.set_title(f'VR - complex with $r$ = {r}')
    ax.set_xticks([])
    ax.set_yticks([])
    if not frame:
        ax.set_frame_on(False)
    