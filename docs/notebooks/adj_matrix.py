rows = 8
cols = 10

# Initialize the adjacency matrix with zeros
adj_matrix = np.zeros((rows * cols, rows * cols), dtype=int)

# Iterate over each node in the grid
for i in range(rows):
    for j in range(cols):
        node_index = i * cols + j

        if i%2 == 0:
            # Connect to the right neighbor (if exists)
            if j < cols - 1:
                adj_matrix[node_index, node_index + 1] = 1
                adj_matrix[node_index + 1, node_index] = 1 

            # Connect to the bottom neighbor (if exists)
            if i < rows - 1:
                adj_matrix[node_index, node_index + cols] = 1
                adj_matrix[node_index + cols, node_index] = 1

                adj_matrix[node_index, node_index + cols - 1] = 1
                adj_matrix[node_index + cols - 1, node_index] = 1    
                
        else:
            # Connect to the right neighbor (if exists)
            if j < cols - 1:
                adj_matrix[node_index, node_index + 1] = 1
                adj_matrix[node_index + 1, node_index] = 1   

            # Connect to the bottom-right neighbor (if exists)
            if i < rows - 1 and j < cols - 1:
                adj_matrix[node_index, node_index + cols + 1] = 1
                adj_matrix[node_index + cols + 1, node_index] = 1   

                adj_matrix[node_index, node_index + cols] = 1
                adj_matrix[node_index + cols, node_index] = 1   

        if node_index%(2*cols) == 0:
            adj_matrix[node_index, node_index + cols - 1] = 0
            adj_matrix[node_index + cols - 1, node_index] = 0   

            if node_index > 0:
                adj_matrix[node_index - 1, node_index + cols - 1] = 1
                adj_matrix[node_index + cols - 1, node_index - 1] = 1
                print(node_index - cols - 1, node_index - 1)

adj_matrix *= 0.001

plt.imshow(adj_matrix); plt.colorbar()
