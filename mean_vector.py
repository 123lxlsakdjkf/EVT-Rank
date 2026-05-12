import pickle
import numpy as np
import os
from datetime import datetime

# --- Configuration Path ---
embedding_file = ''
output_dir = ''

# --- Check if file exists ---
if not os.path.exists(embedding_file):
    print(f"Error: The file does not exist, please check the path: {embedding_file}")
    exit(1)

print(f"Loading file: {embedding_file}")
print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Load with pickle ---
print("Loading with pickle...")
with open(embedding_file, 'rb') as f:
    allids, allembeddings = pickle.load(f)

print(f"Loading successful!")
print(f"Data type: {type(allembeddings)}")
print(f"Shape: {allembeddings.shape}")
print(f"Passage count: {len(allids)}")
print(f"Vector dimension: {allembeddings.shape}")

# --- Compute mean ---
print("\nComputing mean of all vectors...")
mean_vector = np.mean(allembeddings, axis=0)
print("Computation completed!")

# --- Output results ---
print(f"\n{'='*50}")
print(f"Global mean vector (Shape: {mean_vector.shape}):")
print(f"{'='*50}")
print(mean_vector)

print(f"\nMean vector (first 10 elements): {mean_vector[:10]}")
print(f"Mean vector (last 10 elements): {mean_vector[-10:]}")

# --- Save mean vector ---
print(f"\nSaving mean vector...")

# Save as NumPy format
npy_path = os.path.join(output_dir, 'mean_vector.npy')
np.save(npy_path, mean_vector)
print(f"Saved as NumPy format: {npy_path}")

# Save as text format
txt_path = os.path.join(output_dir, 'mean_vector.txt')
with open(txt_path, 'w') as f:
    f.write(f"# Mean vector (Dimension: {mean_vector.shape})\n")
    f.write(f"# Computation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"# Source file: {embedding_file}\n")
    f.write(f"# Passage count: {len(allids)}\n\n")
    f.write(" ".join([f"{x:.6f}" for x in mean_vector.tolist()]))
print(f"Saved as text format: {txt_path}")

print(f"\n{'='*50}")
print(f"All files saved to: {output_dir}")
print(f"{'='*50}")
