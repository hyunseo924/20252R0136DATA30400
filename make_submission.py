"""
Generate Kaggle Submission using trained model (best_model_fixed.pt)
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ------------------------
# Paths
# ------------------------
ROOT = r"F:/work/BIGdata/project_release/Amazon_products"
TEST_PATH = os.path.join(ROOT, "test/test_corpus.txt")
HIERARCHY_PATH = os.path.join(ROOT, "class_hierarchy.txt")
KEYWORDS_PATH = os.path.join(ROOT, "class_related_keywords.txt")
MODEL_SAVE_DIR = r'F:/work/BIGdata/project_release/model'
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model_fixed.pt")

SUBMISSION_DIR = r"./BIGdata/project_release/submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------
# Load Test Data
# ------------------------
print("\nLoading test data...")
test_documents = []
with open(TEST_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            test_documents.append({'id': parts[0], 'text': parts[1]})  # Keep id as string for submission

print(f"Loaded {len(test_documents)} test documents")

# ------------------------
# Load Hierarchy and Class Info
# ------------------------
print("Loading hierarchy and class info...")

# Load class info
class_info = {}
with open(KEYWORDS_PATH, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        parts = line.strip().split(':')
        if len(parts) == 2:
            name = parts[0].strip().replace('_', ' ')
            keywords = parts[1].strip()
            class_info[idx] = {'name': name, 'keywords': keywords}

num_classes = len(class_info)
print(f"Number of classes: {num_classes}")

# Build hierarchy graph
G = nx.DiGraph()
with open(HIERARCHY_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        p, c = map(int, line.strip().split())
        G.add_edge(p, c)

# ------------------------
# Hierarchy Functions
# ------------------------
def get_ancestors(G, node):
    """Get all ancestor nodes"""
    ancestors = set()
    queue = [node]
    while queue:
        current = queue.pop(0)
        for parent in G.predecessors(current):
            if parent not in ancestors:
                ancestors.add(parent)
                queue.append(parent)
    return ancestors

def enforce_hierarchy_constraint(labels, G):
    """Enforce hierarchical consistency"""
    constrained_labels = set(labels)
    for label in labels:
        ancestors = get_ancestors(G, label)
        constrained_labels.update(ancestors)
    return list(constrained_labels)

# ------------------------
# Build Adjacency Matrix
# ------------------------
def build_adjacency_matrix(G, num_classes):
    A = torch.eye(num_classes)
    for u, v in G.edges():
        A[u, v] = 1
        A[v, u] = 1
    
    degree = A.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    
    return A_hat

A_hat = build_adjacency_matrix(G, num_classes).to(device)

# ------------------------
# Model Definition
# ------------------------
class ImprovedGCNClassifier(nn.Module):
    def __init__(self, num_classes, bert_model='bert-base-uncased', hidden_dim=768, gcn_layers=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        self.gcn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(gcn_layers)
        ])
        
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, input_ids, attention_mask, A_hat):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        doc_embed = bert_output.last_hidden_state[:, 0, :]
        
        class_feats = self.class_embeddings
        for gcn_layer in self.gcn_layers:
            class_feats = F.relu(gcn_layer(A_hat @ class_feats))
            class_feats = self.layer_norm(class_feats)
        
        doc_expanded = doc_embed.unsqueeze(1)
        class_expanded = class_feats.unsqueeze(0).expand(doc_embed.size(0), -1, -1)
        
        attn_output, _ = self.attention(doc_expanded, class_expanded, class_expanded)
        attn_output = attn_output.squeeze(1)
        
        combined = torch.cat([doc_embed, attn_output], dim=1)
        logits = self.classifier(combined)
        
        return logits

# ------------------------
# Dataset
# ------------------------
class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

# ------------------------
# Load Model
# ------------------------
print(f"\nLoading model from {MODEL_PATH}...")
model = ImprovedGCNClassifier(num_classes).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded successfully! (Best F1: {checkpoint.get('f1', 'N/A')})")

# ------------------------
# Generate Predictions
# ------------------------
print("\nGenerating predictions...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

test_texts = [doc['text'] for doc in test_documents]
test_dataset = TestDataset(test_texts, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        
        logits = model(ids, mask, A_hat)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        for prob_vec in probs:
            # Get top predictions
            top_indices = np.argsort(prob_vec)[-5:][::-1]  # Top 5 candidates
            
            # Apply threshold
            threshold = 0.3
            predictions = [idx for idx in top_indices if prob_vec[idx] > threshold]
            
            # Ensure 1-3 labels (task requirement)
            if len(predictions) < 1:
                predictions = [top_indices[0]]  # At least 1
            elif len(predictions) > 3:
                predictions = predictions[:3]  # At most 3
            
            # Apply hierarchy constraint
            predictions = enforce_hierarchy_constraint(predictions, G)
            
            # Final limit to 3 most confident
            if len(predictions) > 3:
                top_probs = [(p, prob_vec[p]) for p in predictions]
                top_probs.sort(key=lambda x: x[1], reverse=True)
                predictions = [p for p, _ in top_probs[:3]]
            
            # Ensure at least 1 label
            if len(predictions) == 0:
                predictions = [top_indices[0]]
            
            all_predictions.append(sorted(predictions))

# ------------------------
# Save Submission
# ------------------------
print(f"\nSaving submission to {SUBMISSION_PATH}...")

with open(SUBMISSION_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'labels'])
    
    for doc, preds in zip(test_documents, all_predictions):
        pid = doc['id']
        labels_str = ','.join(map(str, preds))
        writer.writerow([pid, labels_str])

print("[SUCCESS] Submission file saved!")
print(f"  Path: {SUBMISSION_PATH}")
print(f"  Total samples: {len(all_predictions)}")
print(f"  Avg labels per sample: {np.mean([len(p) for p in all_predictions]):.2f}")
print(f"  Label distribution:")
print(f"    - 1 label: {sum(1 for p in all_predictions if len(p) == 1)}")
print(f"    - 2 labels: {sum(1 for p in all_predictions if len(p) == 2)}")
print(f"    - 3 labels: {sum(1 for p in all_predictions if len(p) == 3)}")
print(f"    - 3+ labels: {sum(1 for p in all_predictions if len(p) > 3)}")

print("\n[SUCCESS] Done! Ready for Kaggle submission.")