import os
import tarfile
import pickle
import numpy as np
import torch
from tqdm import tqdm
import index_utils.contriever
import index_utils.utils
import index_utils.normalize_text

# --- Configuration Path ---
data_tar_path = ''          # Dataset
mean_vector_path = ''  # Previously computed mean vector
model_path = ''       # Contriever model path
output_file = ''   # Output file

# --- 1. Extract all questions from data.tar ---
print("Extracting questions from data.tar...")
questions = []

with tarfile.open(data_tar_path, 'r') as tar:
    for member in tar.getmembers():
        if member.name.endswith('.csv'):
            f = tar.extractfile(member)
            if f is None:
                continue
            lines = f.read().decode('utf-8').strip().split('\n')
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 6:
                    question = parts[0].strip()
                    if question:
                        questions.append(question)

print(f"{len(questions)} questions were extracted")

# --- 2. Load Contriever model ---
print("Loading Contriever model...")
model, tokenizer = index_utils.contriever.load_retriever(model_path)
model.eval()
model = model.cuda()
model = model.half()  

# --- 3. Load mean vector ---
print("Loading mean vector...")
mean_vector = np.load(mean_vector_path)
mean_vector_tensor = torch.from_numpy(mean_vector).cuda().half()
print(f"Mean vector shape: {mean_vector_tensor.shape}")

# --- 4. Encode all questions and compute long-tail values ---
print("Encoding questions and computing long-tail values...")
a_values = []
batch_size = 64

with torch.no_grad():
    for i in tqdm(range(0, len(questions), batch_size)):
        batch_questions = questions[i:i+batch_size]
        
        # coding
        encoded = tokenizer.batch_encode_plus(
            batch_questions,
            return_tensors="pt",
            max_length=512,
            padding=True,
            truncation=True,
        )
        encoded = {k: v.cuda() for k, v in encoded.items()}
        
        # Get Embedded Vector
        embeddings = model(**encoded)  # shape: (batch_size, 768)

        # Compute cosine similarity
        # Normalize
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
        mean_norm = mean_vector_tensor / mean_vector_tensor.norm()

        # Cosine similarity s
        s = torch.matmul(embeddings_norm, mean_norm)  # shape: (batch_size,)

        # Compute long-tail value
        s = torch.clamp(s, min=1e-6, max=1.0)
        a = torch.exp(1.0 / s)
        
        a_values.extend(a.cpu().numpy().tolist())

# --- 5. Save results ---
print(f"Saving results to {output_file}...")
with open(output_file, 'w') as f:
    f.write(f"# Long-tail values\n")
    f.write(f"# Total questions: {len(questions)}\n")
    f.write(f"# Mean vector source: {mean_vector_path}\n")
    f.write(f"# Format: Question\t| Long-tail value\n")
    f.write("#" + "="*80 + "\n")
    
    for q, a_val in zip(questions, a_values):
        q_clean = q.replace('\t', ' ').replace('\n', ' ')
        f.write(f"{q_clean}\t| {a_val:.6f}\n")

print(f"Done! A total of {len(questions)} questions were processed")
