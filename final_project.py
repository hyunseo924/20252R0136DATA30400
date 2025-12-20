"""
Hierarchical Multi-label Text Classification with Self-Training
- BCELoss with class frequency-based weighting
- Conservative pseudo-labeling with golden label protection
"""

import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

# =============================================================================
# 설정 및 시드 고정
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 경로 설정 (스크립트 위치 기준)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "Amazon_products")
TRAIN_PATH = os.path.join(ROOT, "train", "train_corpus.txt")
TEST_PATH = os.path.join(ROOT, "test", "test_corpus.txt")
HIERARCHY_PATH = os.path.join(ROOT, "class_hierarchy.txt")
KEYWORDS_PATH = os.path.join(ROOT, "class_related_keywords.txt")
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "model")
LLM_DATA_PATH = os.path.join(ROOT, "new_llm_generated_data.pkl")
SUBMISSION_DIR = os.path.join(SCRIPT_DIR, "submission")

# 필요한 디렉토리 생성
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Script directory: {SCRIPT_DIR}")
print(f"Data root: {ROOT}")

# =============================================================================
# 데이터 로딩
# =============================================================================

def load_data_and_graph():
    """학습/테스트 문서, 클래스 정보, 계층 구조 그래프 로드"""
    # 필수 파일 존재 확인
    required_files = {
        'Train corpus': TRAIN_PATH,
        'Test corpus': TEST_PATH,
        'Class hierarchy': HIERARCHY_PATH,
        'Class keywords': KEYWORDS_PATH,
        'LLM data': LLM_DATA_PATH
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"  - {name}: {path}")
    
    if missing_files:
        print("\n[ERROR] Required files not found:")
        print("\n".join(missing_files))
        print(f"\nPlease ensure the following directory structure:")
        print(f"{SCRIPT_DIR}/")
        print(f"  └── Amazon_products/")
        print(f"      ├── train/")
        print(f"      │   └── train_corpus.txt")
        print(f"      ├── test/")
        print(f"      │   └── test_corpus.txt")
        print(f"      ├── class_hierarchy.txt")
        print(f"      ├── class_related_keywords.txt")
        print(f"      └── new_llm_generated_data.pkl")
        raise FileNotFoundError("Required data files are missing")
    
    # 학습 문서 로드
    documents = []
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                documents.append({'id': int(parts[0]), 'text': parts[1]})
    
    # 테스트 문서 로드
    test_documents = []
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                test_documents.append({'id': int(parts[0]), 'text': parts[1]})
    
    # 클래스 정보 로드
    class_info = {}
    with open(KEYWORDS_PATH, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split(':')
            if len(parts) == 2:
                name = parts[0].strip().replace('_', ' ')
                keywords = parts[1].strip()
                class_info[idx] = {'name': name, 'keywords': keywords}
    
    # 계층 구조 그래프 로드
    G = nx.DiGraph()
    with open(HIERARCHY_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            p, c = map(int, line.strip().split())
            G.add_edge(p, c)
    
    num_classes = len(class_info)
    roots = [n for n, d in G.in_degree() if d == 0]
    
    return documents, test_documents, G, class_info, num_classes, roots

all_docs, test_docs, G, class_info, num_classes, roots = load_data_and_graph()
print(f"Loaded {len(all_docs)} training docs, {len(test_docs)} test docs, and {num_classes} classes")

# LLM 생성 데이터 로드
with open(LLM_DATA_PATH, 'rb') as f:
    result = pickle.load(f)
val_data = result['val_data']
train_seed_labels = result['train_seed_labels']
test_seed_labels = result['test_seed_labels']
unlabeled_train_indices = result['unlabeled_train_indices']
unlabeled_test_indices = result['unlabeled_test_indices']
print("LLM generated data loaded")

# =============================================================================
# 계층 구조 제약 함수
# =============================================================================

def get_ancestors(G, node):
    """주어진 노드의 모든 조상 노드 반환"""
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
    """자식 레이블이 있으면 부모 레이블도 추가 (계층 제약)"""
    constrained_labels = set(labels)
    for label in labels:
        ancestors = get_ancestors(G, label)
        constrained_labels.update(ancestors)
    return list(constrained_labels)

# =============================================================================
# Silver Label 생성 (TF-IDF + Label Propagation)
# =============================================================================

def generate_improved_silver_labels(train_docs, test_docs, unlabeled_indices, class_info,
                                    train_seed_labels, test_seed_labels, G, num_classes):
    """TF-IDF와 Label Propagation을 결합한 Silver Label 생성"""
    print("\n=== Generating Improved Silver Labels ===")
    
    # Stage 1: TF-IDF 기반 후보 생성
    print("Stage 1: TF-IDF candidates...")
    class_descriptions = [f"{class_info[i]['name']} {class_info[i]['keywords']}" for i in range(num_classes)]
    
    unlabeled_docs = [train_docs[i] for i in unlabeled_indices]
    doc_texts = [d['text'] for d in unlabeled_docs]
    
    vectorizer = TfidfVectorizer(max_features=8000, stop_words='english', ngram_range=(1, 3), sublinear_tf=True)
    all_texts = doc_texts + class_descriptions
    vectorizer.fit(all_texts)
    
    doc_vectors = vectorizer.transform(doc_texts)
    class_vectors = vectorizer.transform(class_descriptions)
    tfidf_sim = cosine_similarity(doc_vectors, class_vectors)
    
    # Stage 2: Seed Label 전파
    print("Stage 2: Label propagation from seeds...")
    
    seed_texts = []
    seed_labels_list = []
    
    # Train seeds 수집
    for idx, labels in train_seed_labels.items():
        seed_texts.append(train_docs[idx]['text'])
        seed_labels_list.append(labels)
    
    # Test seeds 수집
    for idx, labels in test_seed_labels.items():
        seed_texts.append(test_docs[idx]['text'])
        seed_labels_list.append(labels)
    
    print(f"  Total seeds: {len(seed_texts)} (train: {len(train_seed_labels)}, test: {len(test_seed_labels)})")
    
    seed_vectors = vectorizer.transform(seed_texts)
    doc_to_seed_sim = cosine_similarity(doc_vectors, seed_vectors)
    
    # 유사 seed로부터 label 전파
    propagated_scores = np.zeros((len(unlabeled_indices), num_classes))
    for doc_idx in range(len(unlabeled_indices)):
        top_seeds = np.argsort(doc_to_seed_sim[doc_idx])[-5:]
        
        for seed_idx in top_seeds:
            sim = doc_to_seed_sim[doc_idx, seed_idx]
            if sim > 0.3:
                for label in seed_labels_list[seed_idx]:
                    if label < num_classes:
                        propagated_scores[doc_idx, label] += sim
    
    max_prop = propagated_scores.max()
    if max_prop > 0:
        propagated_scores /= max_prop
    
    # Stage 3: 점수 결합 및 레이블 할당
    print("Stage 3: Combining scores...")
    combined_scores = 0.6 * tfidf_sim + 0.4 * propagated_scores
    
    silver_labels = {}
    for doc_idx, unlabeled_idx in enumerate(tqdm(unlabeled_indices, desc="Assigning")):
        scores = combined_scores[doc_idx]
        
        threshold = max(np.percentile(scores, 85), 0.1)
        candidates = np.where(scores > threshold)[0].tolist()
        
        if len(candidates) < 2:
            candidates = np.argsort(scores)[-2:].tolist()
        elif len(candidates) > 5:
            candidates = np.argsort(scores)[-5:].tolist()
        
        candidates = enforce_hierarchy_constraint(candidates, G)
        
        if len(candidates) > 10:
            top_scores = [(c, scores[c]) for c in candidates]
            top_scores.sort(key=lambda x: x[1], reverse=True)
            candidates = [c for c, _ in top_scores[:10]]
        
        silver_labels[unlabeled_idx] = candidates
    
    print(f"Generated {len(silver_labels)} silver labels")
    return silver_labels

# Silver label 생성 및 학습 레이블 결합
silver_labels = generate_improved_silver_labels(
    train_docs=all_docs,
    test_docs=test_docs,
    unlabeled_indices=unlabeled_train_indices,
    class_info=class_info,
    train_seed_labels=train_seed_labels,
    test_seed_labels=test_seed_labels,
    G=G,
    num_classes=num_classes
)

combined_train_labels = {**train_seed_labels, **silver_labels}
print(f"Total training labels: {len(combined_train_labels)}")

# =============================================================================
# 인접 행렬 구축 (GCN용)
# =============================================================================

def build_adjacency_matrix(G, num_classes):
    """계층 구조 그래프로부터 정규화된 인접 행렬 생성"""
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
print(f"Adjacency matrix shape: {A_hat.shape}")

# =============================================================================
# 데이터 준비 (학습/검증 분할)
# =============================================================================

id_to_doc = {d['id']: d for d in all_docs}
index_to_doc = {i: d for i, d in enumerate(all_docs)}

train_texts = []
train_labels_remapped = {}

# combined_train_labels에서 텍스트와 레이블 추출
for key, labels in combined_train_labels.items():
    if isinstance(key, int):
        if key in id_to_doc:
            doc = id_to_doc[key]
            train_texts.append(doc['text'])
            train_labels_remapped[len(train_texts) - 1] = labels
        elif 0 <= key < len(all_docs):
            doc = all_docs[key]
            train_texts.append(doc['text'])
            train_labels_remapped[len(train_texts) - 1] = labels

# Test seed도 학습에 포함
for key, labels in test_seed_labels.items():
    train_texts.append(test_docs[key]['text'])
    train_labels_remapped[len(train_texts)-1] = labels

print(f"Prepared {len(train_texts)} training samples")

# 검증 데이터 준비
if val_data:
    val_texts = [item['text'] for item in val_data]
    val_labels_remapped = {i: item['labels'] for i, item in enumerate(val_data)}
    print(f"Prepared {len(val_texts)} validation samples")
else:
    from sklearn.model_selection import train_test_split
    train_indices = list(range(len(train_texts)))
    train_idx, val_idx = train_test_split(train_indices, test_size=0.2, random_state=42)
    
    val_texts = [train_texts[i] for i in val_idx]
    val_labels_remapped = {i: train_labels_remapped[val_idx[i]] for i in range(len(val_idx))}
    
    train_texts = [train_texts[i] for i in train_idx]
    new_train_labels = {i: train_labels_remapped[train_idx[i]] for i in range(len(train_idx))}
    train_labels_remapped = new_train_labels
    
    print(f"Split: {len(train_texts)} train, {len(val_texts)} val")

# =============================================================================
# 모델 정의
# =============================================================================

class HierarchicalGCNClassifier(nn.Module):
    """BERT + GCN 기반 계층적 분류 모델"""
    def __init__(self, num_classes, bert_model='bert-base-uncased', hidden_dim=768, gcn_layers=3, freeze_bert_layers=6):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # BERT 하위 레이어 동결 (학습 속도 향상)
        if freeze_bert_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(freeze_bert_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
        
        # GCN 레이어 (residual connection 포함)
        self.gcn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(gcn_layers)
        ])
        self.gcn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(gcn_layers)
        ])
        
        # 학습 가능한 클래스 임베딩
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        
        # Multi-head attention (문서-클래스 상호작용)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True, dropout=0.1)
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask, A_hat):
        # BERT 인코딩
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        doc_embed = bert_output.last_hidden_state[:, 0, :]
        
        # GCN으로 클래스 임베딩 업데이트
        class_feats = self.class_embeddings
        for gcn_layer, gcn_norm in zip(self.gcn_layers, self.gcn_norms):
            residual = class_feats
            class_feats = gcn_layer(A_hat @ class_feats)
            class_feats = F.gelu(class_feats)
            class_feats = gcn_norm(class_feats)
            class_feats = class_feats + residual
        
        class_feats = self.dropout(class_feats)
        
        # 문서-클래스 어텐션
        doc_expanded = doc_embed.unsqueeze(1)
        class_expanded = class_feats.unsqueeze(0).expand(doc_embed.size(0), -1, -1)
        
        attn_output, _ = self.attention(doc_expanded, class_expanded, class_expanded)
        attn_output = attn_output.squeeze(1)
        
        # 결합 및 분류
        combined = torch.cat([doc_embed, attn_output], dim=1)
        logits = self.classifier(combined)
        
        return logits

# =============================================================================
# Loss 함수
# =============================================================================

class WeightedBCELoss(nn.Module):
    """클래스 빈도 기반 가중치 BCE Loss"""
    def __init__(self, class_weights=None, pos_weight_factor=1.5):
        super().__init__()
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
            self.register_buffer('pos_weight', 
                        torch.ones(class_weights.size(0)) * pos_weight_factor)
        else:
            self.pos_weight = None
            
    def forward(self, logits, targets, sample_weights=None, reduction='mean'):
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        # 클래스 빈도 가중치 적용
        if self.class_weights is not None:
            bce_loss = bce_loss * self.class_weights
        
        # 샘플별 가중치 적용
        if sample_weights is not None:
            bce_loss = bce_loss * sample_weights.view(-1, 1)
        
        if reduction == 'none':
            return bce_loss.mean(dim=1)
        elif reduction == 'sum':
            return bce_loss.sum()
        else:
            return bce_loss.mean()

def compute_class_weights(train_labels, num_classes, beta=0.9):
    """Effective Number of Samples 방식의 클래스 가중치 계산"""
    class_counts = np.zeros(num_classes)
    
    for labels in train_labels.values():
        for label in labels:
            if label < num_classes:
                class_counts[label] += 1
    
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-7)
    
    # 정규화 및 클리핑
    weights = weights / weights.sum() * num_classes
    weights = np.clip(weights, 0.1, 10.0)
    
    return torch.FloatTensor(weights)

def create_sample_weights_v2(train_size, unlabeled_indices, round_num):
    """Golden vs Silver 샘플 가중치 생성"""
    sample_weights = {}
    
    golden_weight = 4.0
    
    silver_base = 0.2
    silver_max = 0.6
    silver_growth = min(round_num * 0.15, silver_max - silver_base)
    silver_weight = silver_base + silver_growth
    
    for idx in range(train_size):
        if idx in unlabeled_indices:
            sample_weights[idx] = silver_weight
        else:
            sample_weights[idx] = golden_weight
    
    print(f"  Sample Weights: Golden={golden_weight:.1f}, Silver={silver_weight:.2f}")
    return sample_weights

# =============================================================================
# Dataset
# =============================================================================

class WeightedTextDataset(Dataset):
    """샘플별 가중치를 포함한 텍스트 데이터셋"""
    def __init__(self, texts, labels_dict, weights_dict, tokenizer, num_classes, max_length=128):
        self.labels_dict = labels_dict
        self.weights_dict = weights_dict
        self.num_classes = num_classes
        
        print("Pre-tokenizing data...")
        self.encodings = tokenizer.batch_encode_plus(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        print("Tokenization complete!")

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        
        weight = float(self.weights_dict.get(idx, 1.0))
        
        label_vector = torch.zeros(self.num_classes)
        if idx in self.labels_dict:
            for label in self.labels_dict[idx]:
                if label < self.num_classes:
                    label_vector[label] = 1.0
        
        item['labels'] = label_vector
        item['weight'] = weight
        return item

# =============================================================================
# 학습 및 평가 함수
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, A_hat, device):
    """1 에폭 학습"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        weights = batch['weight'].to(device)
        
        optimizer.zero_grad()
        logits = model(ids, mask, A_hat)
        loss = criterion(logits, labels, weights)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, A_hat, device, threshold=0.45):
    """모델 평가 (Micro/Macro F1)"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(ids, mask, A_hat)
            preds = (torch.sigmoid(logits) > threshold).float()
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Micro-F1
    tp = (all_preds * all_targets).sum()
    fp = (all_preds * (1 - all_targets)).sum()
    fn = ((1 - all_preds) * all_targets).sum()
    micro_f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)
    
    # Macro-F1
    tp_per = (all_preds * all_targets).sum(axis=0)
    fp_per = (all_preds * (1 - all_targets)).sum(axis=0)
    fn_per = ((1 - all_preds) * all_targets).sum(axis=0)
    f1_per = 2 * tp_per / (2 * tp_per + fp_per + fn_per + 1e-10)
    macro_f1 = f1_per.mean()
    
    return micro_f1, macro_f1

class EarlyStoppingV2:
    """Early Stopping (Micro-F1 기준)"""
    def __init__(self, patience=3, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, micro_f1):
        if self.best_score is None:
            self.best_score = micro_f1
            return False
        
        if micro_f1 > self.best_score + self.min_delta:
            self.best_score = micro_f1
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

# =============================================================================
# Pseudo-label 업데이트
# =============================================================================

def update_labels_ultra_conservative(model, loader, train_labels, unlabeled_indices,
                                     G, num_classes, device, A_hat, round_num):
    """매우 보수적인 Pseudo-label 업데이트 (Golden label 보호)"""
    model.eval()
    new_labels = train_labels.copy()
    
    # Round별 threshold 조정
    confidence_threshold = 0.60 + round_num * 0.05
    overlap_threshold = 0.50 + round_num * 0.05
    
    stats = {'updated': 0, 'rejected_conf': 0, 'rejected_overlap': 0}
    
    with torch.no_grad():
        current_idx = 0
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            logits = model(ids, mask, A_hat)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            batch_size = ids.size(0)
            
            for i in range(batch_size):
                global_idx = current_idx + i
                
                # Golden label은 절대 변경 금지
                if global_idx not in unlabeled_indices:
                    continue
                
                prob_vec = probs[i]
                original = train_labels[global_idx]
                
                # 1. High confidence 체크
                high_conf = np.where(prob_vec > confidence_threshold)[0]
                if len(high_conf) < 2:
                    stats['rejected_conf'] += 1
                    continue
                
                # 2. Maximum probability 체크
                max_prob = prob_vec.max()
                if max_prob < 0.80:
                    stats['rejected_conf'] += 1
                    continue
                
                # 3. Top-K must overlap with original
                top5 = np.argsort(prob_vec)[-5:]
                if not set(top5) & set(original):
                    stats['rejected_overlap'] += 1
                    continue
                
                # 4. High overlap ratio
                candidates = high_conf.tolist()
                candidates = enforce_hierarchy_constraint(candidates, G)
                
                if len(candidates) > 10:
                    scores = [(c, prob_vec[c]) for c in candidates]
                    scores.sort(key=lambda x: x[1], reverse=True)
                    candidates = [c for c, _ in scores[:10]]
                
                overlap = len(set(candidates) & set(original)) / max(len(original), 1)
                
                if overlap >= overlap_threshold:
                    new_labels[global_idx] = candidates
                    stats['updated'] += 1
                else:
                    stats['rejected_overlap'] += 1
            
            current_idx += batch_size
    
    # 통계 출력
    total = sum(stats.values())
    print(f"\n  Pseudo-label Update (threshold={confidence_threshold:.2f}):")
    print(f"    ✓ Updated: {stats['updated']} ({100*stats['updated']/total:.1f}%)")
    print(f"    - Rejected (low conf): {stats['rejected_conf']}")
    print(f"    - Rejected (low overlap): {stats['rejected_overlap']}")
    
    return new_labels

# =============================================================================
# Self-Training 메인 함수
# =============================================================================

def train_simplified_selftraining(
    model, train_texts, train_labels, unlabeled_indices,
    val_texts, val_labels, tokenizer, num_classes, G, A_hat, device,
    rounds=2, epochs_per_round=8, batch_size=32
):
    """Simplified Self-Training with Weighted BCE"""
    model_path = os.path.join(MODEL_SAVE_DIR, 'best_model_final.pt')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 클래스 가중치 계산
    class_weights = compute_class_weights(train_labels, num_classes, beta=0.9)
    print(f"Class weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}")
    
    # Loss 함수
    criterion = WeightedBCELoss(class_weights, pos_weight_factor=1.5).to(device)
    print(f"Loss pos_weight device: {criterion.pos_weight.device}")
    
    best_micro = 0
    best_info = {}
    
    for round_num in range(rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_num + 1}/{rounds}")
        print(f"{'='*60}")
        
        # 샘플 가중치 생성
        sample_weights = create_sample_weights_v2(
            len(train_texts), unlabeled_indices, round_num
        )
        
        # Dataset 생성
        train_ds = WeightedTextDataset(
            train_texts, train_labels, sample_weights, tokenizer, num_classes
        )
        val_ds = WeightedTextDataset(
            val_texts, val_labels, {}, tokenizer, num_classes
        )
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        
        # Optimizer
        bert_params = list(model.bert.parameters())
        other_params = [p for n, p in model.named_parameters() if 'bert' not in n]
        
        optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': 5e-6},
            {'params': other_params, 'lr': 1e-4}
        ], weight_decay=0.01)
        
        # Scheduler
        total_steps = len(train_loader) * epochs_per_round
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-7
        )
        
        # Early stopping
        early_stop = EarlyStoppingV2(patience=3, min_delta=0.002)
        
        # Training loop
        for epoch in range(epochs_per_round):
            loss = train_epoch(model, train_loader, optimizer, criterion, A_hat, device)
            micro, macro = evaluate(model, val_loader, A_hat, device, threshold=0.45)
            
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            
            print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Micro={micro:.4f}, "
                  f"Macro={macro:.4f}, LR={lr:.2e}")
            
            # 최고 성능 모델 저장
            if micro > best_micro:
                best_micro = micro
                best_info = {
                    'round': round_num + 1,
                    'epoch': epoch + 1,
                    'micro': micro,
                    'macro': macro
                }
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'micro_f1': micro,
                    'macro_f1': macro,
                    'round': round_num,
                    'epoch': epoch
                }, model_path)
                print(f"    ✓ BEST saved! Micro={micro:.4f}")
            
            # Early stopping
            if early_stop(micro):
                print(f"  Early stop at epoch {epoch+1}")
                break
        
        # Pseudo-label 업데이트 (마지막 round 제외)
        if round_num < rounds - 1:
            print(f"\n  Updating pseudo-labels...")
            inference_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
            
            train_labels = update_labels_ultra_conservative(
                model, inference_loader, train_labels, unlabeled_indices,
                G, num_classes, device, A_hat, round_num
            )
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"  Best Micro-F1: {best_micro:.4f}")
    print(f"  Round {best_info['round']}, Epoch {best_info['epoch']}")
    print(f"{'='*60}")
    
    return best_micro

# =============================================================================
# 테스트 예측 및 제출 파일 생성
# =============================================================================

class TestDataset(Dataset):
    """테스트 데이터셋"""
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

def generate_predictions(model, test_docs, test_seed_labels, tokenizer, A_hat, device, threshold=0.45):
    """테스트 데이터 예측 생성"""
    test_texts = [doc['text'] for doc in test_docs]
    test_dataset = TestDataset(test_texts, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print("\nGenerating predictions...")
    all_predictions = []
    current_idx = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            logits = model(ids, mask, A_hat)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            batch_size = ids.size(0)
            
            for i in range(batch_size):
                global_idx = current_idx + i
                
                # LLM label이 있으면 우선 사용
                if global_idx in test_seed_labels:
                    pred_labels = test_seed_labels[global_idx]
                else:
                    # 모델 예측
                    prob_vec = probs[i]
                    high_conf = np.where(prob_vec > threshold)[0].tolist()
                    
                    if len(high_conf) < 2:
                        high_conf = np.argsort(prob_vec)[-2:].tolist()
                    
                    pred_labels = enforce_hierarchy_constraint(high_conf, G)
                    
                    if len(pred_labels) > 10:
                        scores = [(c, prob_vec[c]) for c in pred_labels]
                        scores.sort(key=lambda x: x[1], reverse=True)
                        pred_labels = [c for c, _ in scores[:10]]
                
                all_predictions.append(pred_labels)
            
            current_idx += batch_size
    
    return all_predictions

def save_submission(all_predictions, submission_path):
    """제출 파일 저장"""
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    
    with open(submission_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['TEST_ID', 'CATEGORY'])
        
        for idx, pred_labels in enumerate(all_predictions):
            pred_str = ' '.join(map(str, pred_labels))
            writer.writerow([idx, pred_str])
    
    print(f"\n[SUCCESS] Submission saved to {submission_path}")
    print(f" - Total predictions: {len(all_predictions)}")
    print(f" - Avg labels per prediction: {np.mean([len(p) for p in all_predictions]):.2f}")

# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    # 토크나이저 및 모델 초기화
    print("\n=== Model Training ===")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = HierarchicalGCNClassifier(num_classes, gcn_layers=3, freeze_bert_layers=6).to(device)
    
    # Self-training 실행
    best_f1 = train_simplified_selftraining(
        model=model,
        train_texts=train_texts,
        train_labels=train_labels_remapped,
        unlabeled_indices=unlabeled_train_indices,
        val_texts=val_texts,
        val_labels=val_labels_remapped,
        tokenizer=tokenizer,
        num_classes=num_classes,
        G=G,
        A_hat=A_hat,
        device=device,
        rounds=2,
        epochs_per_round=8,
        batch_size=32
    )
    
    # 최고 성능 모델 로드
    print("\n=== Test Prediction ===")
    model_path = os.path.join(MODEL_SAVE_DIR, 'best_model_final.pt')
    print(f"Loading model from {model_path}...")
    
    model = HierarchicalGCNClassifier(num_classes, gcn_layers=3, freeze_bert_layers=0).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 테스트 예측 생성
    all_predictions = generate_predictions(
        model, test_docs, test_seed_labels, tokenizer, A_hat, device, threshold=0.45
    )
    
    # 제출 파일 저장
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    save_submission(all_predictions, submission_path)
    
    print("\n=== All Done! ===")
