import torch
import os

# Directory where your .pt files are located
directory = 'mv_mirrorwic'

# List to hold all averaged embeddings
averaged_embeddings = []

# Iterate through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".pt"):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        # Load the tensor from the file
        data = torch.load(file_path)
        
        # Compute the mean of the embeddings across the first dimension
        # This results in a single embedding of size [768]
        averaged_embedding = torch.mean(data, dim=0)
        
        # Append the averaged embedding to the list
        averaged_embeddings.append(averaged_embedding)

# Stack all averaged embeddings along a new dimension
# Resulting in a tensor of size [514, 768]
all_embeddings = torch.stack(averaged_embeddings)

# Save the tensor to mirrorwic.pt
torch.save(all_embeddings, 'mirrorwic.pt')

