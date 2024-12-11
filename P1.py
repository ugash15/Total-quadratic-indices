import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
import networkx as nx
import matplotlib.pyplot as plt

# SMILES string for Ibuprofen
smiles = 'C1=CC=C2C(=C1)C(OC2=O)OC(=O)C3=C(N=CC=C3)NC4=CC=CC(=C4)C(F)(F)F'




# Convert the SMILES string to an RDKit molecule object
mol = Chem.MolFromSmiles(smiles)

# Get the adjacency matrix (unmodified, no loops yet)
adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)
adjacency_matrix = np.array(adjacency_matrix, dtype=int)

# Degree of each atom (vertex)
degrees = np.sum(adjacency_matrix, axis=1)

# Create a NetworkX graph for further manipulation (with self-loops)
G = nx.Graph()
for i in range(adjacency_matrix.shape[0]):
    G.add_node(i)
for i in range(adjacency_matrix.shape[0]):
    for j in range(i + 1, adjacency_matrix.shape[1]):
        if adjacency_matrix[i, j] > 0:
            G.add_edge(i, j)

# Compute the First Zagreb Index Matrix (Z₁)
Z1_matrix = np.zeros_like(adjacency_matrix, dtype=int)
for i in range(adjacency_matrix.shape[0]):
    for j in range(i + 1, adjacency_matrix.shape[1]):
        if adjacency_matrix[i, j] > 0:
            Z1_matrix[i, j] = degrees[i] + degrees[j]
            Z1_matrix[j, i] = Z1_matrix[i, j]  # Symmetric matrix

# Compute the Second Zagreb Index Matrix (Z₂)
Z2_matrix = np.zeros_like(adjacency_matrix, dtype=int)
for i in range(adjacency_matrix.shape[0]):
    for j in range(i + 1, adjacency_matrix.shape[1]):
        if adjacency_matrix[i, j] > 0:
            Z2_matrix[i, j] = degrees[i] * degrees[j]
            Z2_matrix[j, i] = Z2_matrix[i, j]  # Symmetric matrix

# Print the Zagreb matrices
print("First Zagreb Index Matrix (Z₁):")
print(Z1_matrix)
print("\nSecond Zagreb Index Matrix (Z₂):")
print(Z2_matrix)

# Calculate the First Zagreb Index (Z₁)
Z1 = np.sum(Z1_matrix)
print(f"\nFirst Zagreb Index (Z₁): {Z1}")

# Calculate the Second Zagreb Index (Z₂)
Z2 = np.sum(Z2_matrix)
print(f"\nSecond Zagreb Index (Z₂): {Z2}")

# Remove self-loops (diagonal elements = 0) for the adjacency matrix without self-loops
np.fill_diagonal(adjacency_matrix, 0)

# Print the adjacency matrix (without self-loops)
print("\nAdjacency Matrix (without self-loops):")
print(adjacency_matrix)

# Degree of each vertex
degrees_no_loops = np.sum(adjacency_matrix, axis=1)

# Function to calculate the Randić Matrix
def randic_matrix(adj_matrix, degrees):
    R = np.zeros_like(adj_matrix, dtype=float)
    for i in range(len(adj_matrix)):
        for j in range(i + 1, len(adj_matrix)):  # Consider only distinct vertices (i != j)
            if adj_matrix[i][j] == 1:
                R[i][j] = 1 / np.sqrt(degrees[i] * degrees[j])
                R[j][i] = 1 / np.sqrt(degrees[i] * degrees[j])
    return R

# Compute the Randić matrix for the Naproxen molecule (without self-loops)
R = randic_matrix(adjacency_matrix, degrees_no_loops)

# Print the Randić Matrix
print("\nRandić Matrix (R):")
print(R)

# Calculate the Randić Index (half the sum of the off-diagonal elements of the Randić matrix)
randić_index = np.sum(R) / 2

# Print the Randić Index
print("\nRandić Index:", randić_index)

# Now, treat all bonds (single and double) as single bonds (set all bond values to 1)
for bond in mol.GetBonds():
    idx1 = bond.GetBeginAtomIdx()
    idx2 = bond.GetEndAtomIdx()
    adjacency_matrix[idx1, idx2] = 1
    adjacency_matrix[idx2, idx1] = 1  # Ensure symmetry

# Print the resulting adjacency matrix (with double bonds treated as single bonds)
print("\nAdjacency Matrix (with double bonds treated as single bonds):")
print(adjacency_matrix)

# Calculate the energy of the graph (sum of absolute values of eigenvalues)
eigenvalues = np.linalg.eigvals(adjacency_matrix)  # Compute eigenvalues of the adjacency matrix
energy = np.sum(np.abs(eigenvalues))  # Sum of the absolute values of the eigenvalues

# Print the energy of the graph
print(f"\nEnergy of the graph (with double bonds as single bonds): {energy}")

# Create a NetworkX graph from the adjacency matrix (with self-loops)
G_with_loops = nx.Graph()

# Add nodes and edges (including self-loops)
for i in range(adjacency_matrix.shape[0]):
    G_with_loops.add_node(i)
for i in range(adjacency_matrix.shape[0]):
    for j in range(i, adjacency_matrix.shape[1]):
        if adjacency_matrix[i, j] > 0:
            G_with_loops.add_edge(i, j, weight=adjacency_matrix[i, j])

# Get positions for atoms using NetworkX's built-in layout
pos = nx.spring_layout(G_with_loops)

# Draw the graph with self-loops
plt.figure(figsize=(10, 8))
nx.draw(G_with_loops, pos, with_labels=True, node_color='#87CEEB', node_size=500, font_size=10)

# Draw edge labels to indicate bond types (single, double, loop)
edge_labels = nx.get_edge_attributes(G_with_loops, 'weight')
nx.draw_networkx_edge_labels(G_with_loops, pos, edge_labels=edge_labels)

plt.title('Molecular Graph of Naproxen with Loops and Bond Types')
plt.show()

# Create a NetworkX graph from the adjacency matrix (with double bonds as single bonds)
G = nx.Graph()

# Add nodes and edges
for i in range(adjacency_matrix.shape[0]):
    G.add_node(i)
for i in range(adjacency_matrix.shape[0]):
    for j in range(i, adjacency_matrix.shape[1]):
        if adjacency_matrix[i, j] > 0:
            G.add_edge(i, j, weight=adjacency_matrix[i, j])

# Get positions for atoms using RDKit's built-in layout
pos = nx.spring_layout(G)

# Draw the graph
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color='#87CEEB', node_size=500, font_size=10)

# Draw edge labels to indicate bond types (all as single bonds)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title('Molecular Graph of Naproxen (Double Bonds as Single Bonds)')
plt.show()
