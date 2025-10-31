# Embedding Model Comparison for Turkish Medical Texts

## Which embedding model works best for Turkish medical texts? I tested 3 popular models with the MedTurkQuaD dataset. The fastest model isn't always the best — here's the proof.

---



##  TL;DR (Quick Summary)

> I compared 3 popular embedding models (Multi-MiniLM, BGE-M3, all-mpnet) using a Turkish medical Q&A dataset. **The results are surprising:**
> 
> -  **BGE-M3:** Best retrieval (MRR: 0.0338) but slowest (50.59s)
> -  **Multi-MiniLM:** Fastest (15.81s) and champion in Turkish morphology (0.9284)
> -  **all-mpnet:** Great for English but fails in Turkish (MRR: 0.0084)
> 
> **Key takeaway:** A "multilingual" label isn't enough. Domain-specific testing is essential!

---

##  The Story: Why I Needed This Test

Last month, I was developing a medical Q&A system. I tried the most popular embedding models on HuggingFace. The results... were disastrous.

For the question "What is an abscess?", the system returned "lung cancer" as the answer. I switched models, got slightly better results, but still not satisfactory.

That's when I realized: **Benchmark tables are valid for English. There was no data for the Turkish + Medical combination.**

In this article, I'll show you which model actually works through a **systematic comparison**.

---

##  Why This Comparison Matters

### Common Problems When Choosing an Embedding Model

 **"Let me pick the most popular model"** → Popularity ≠ Suitable for your use case  
 **"It says multilingual, supports Turkish"** → In theory yes, in practice sometimes no  
 **"Ranked #1 on benchmarks"** → In which language? Which domain?  
 **"Bigger model is better"** → Slower, more expensive, not always better

### What Makes This Test Different

 **Same dataset** → Fair comparison  
 **Same metrics** → Objective evaluation  
 **Reproducible code** → You can try it yourself  
 **Turkish + Domain-specific** → Real-world scenario

---

## 🔬 Test Setup

### Competing Models

| Model | Dimensions | Features | Expectation |
|-------|-----------|----------|-------------|
| **Multi-MiniLM-L12-v2** | 384 | Lightweight, multilingual | Fast but sufficient? |
| **BGE-M3** | 1024 | Next-gen, powerful | Best but how slow? |
| **all-mpnet-base-v2** | 768 | English SOTA | What about Turkish? |

### Test Arena: MedTurkQuaD Dataset

**What?** Turkish medical Q&A dataset  
**Why difficult?** Two-layered challenge:
1.  **Turkish morphology** (suffixes, inflections)
2.  **Medical terminology** (domain-specific)

**Example Challenge:**
```
Question: "An abscess is usually a type of inflammation caused by what?"

 Correct: "pyogenic bacteria"
 Misleading Negative: "uncontrolled cells in lung tissue..."

→ Both answers contain medical terms!
→ Model must capture subtle differences
```

### Reproducibility Guarantee
```python
# Same results on every run
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

**Why 42?** The answer to life, the universe, and everything  (and the AI community's standard seed)

---

##  Test Process: Step by Step

### Step 1: Data Preparation - Negative Sampling
```python
def process_qa_data(qa_data):
    all_queries, all_positives, all_negatives = [], [], []
    
    # Questions and correct answers
    for doc in qa_data.get('data', []):
        for paragraph in doc.get('paragraphs', []):
            for qa_pair in paragraph.get('qas', []):
                all_queries.append(qa_pair['question'])
                all_positives.append(qa_pair['answers'][0]['text'])
    
    # Random negative for each positive
    num_pairs = len(all_positives)
    for i in range(num_pairs):
        idx = i
        while idx == i:  # Don't pick the same answer
            idx = random.choice(range(num_pairs))
        all_negatives.append(all_positives[idx])
    
    return all_queries, all_positives, all_negatives
```

**Why this method?**
- In the real world, correct answers get lost among wrong ones
- Tests the model's discrimination ability
- Classic benchmark method for retrieval systems

### Step 2: Embedding Generation and Time Measurement
```python
for model_name, model in models_to_test.items():
    start_time = time.time()
    
    # Encode
    query_vectors = model.encode(queries, convert_to_numpy=True, show_progress_bar=True)
    doc_vectors = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
    
    duration = time.time() - start_time
    print(f" {model_name}: {duration:.2f} seconds")
```

**Output:**
```
 Multi-MiniLM-L12-v2: 15.81 seconds
 BGE-M3: 50.59 seconds
 all-mpnet-base-v2: 25.00 seconds
```

### Step 3: Similarity Search with FAISS

**Critical Detail:** L2 Normalization
```python
dim = query_vectors.shape[1]
index = faiss.IndexFlatIP(dim)  # Inner Product Index

#  Normalization = Cosine Similarity
faiss.normalize_L2(doc_vectors)
faiss.normalize_L2(query_vectors)

index.add(doc_vectors)
D, I = index.search(query_vectors, k=len(documents))
```

**Why normalize?**

| Case | Formula | What it measures? |
|------|---------|-------------------|
| No normalization | `IP(A,B) = \|A\| × \|B\| × cos(θ)` | Magnitude + Angle |
| With normalization | `IP(A,B) = cos(θ)` | Only Angle (semantic) |

---

##  Evaluation: 4 Different Metrics

###  MRR (Mean Reciprocal Rank)

**What does it measure?** On average, what rank is the correct answer?
```python
def compute_mrr(search_results, true_indices):
    rr_sum = 0
    for i in range(len(true_indices)):
        ranks = np.where(search_results[i] == true_indices[i])[0]
        if len(ranks) > 0:
            rr_sum += 1 / (ranks[0] + 1)
    return rr_sum / len(true_indices)
```

**Interpretation:**
- MRR = 1.0 → Correct answer at rank 1 for every question (perfect!)
- MRR = 0.5 → On average at rank 2
- MRR = 0.033 → On average at ~rank 30 (low)

###  Recall@K

**What does it measure?** Is the correct answer in the top K results?

| Metric | Description |
|--------|-------------|
| Recall@1 | Is the first result correct? (strictest test) |
| Recall@3 | Is it in the top 3? |
| Recall@10 | Is it in the top 10? |

**Why important?**
- Recall@1 → If you're showing only one result to the user
- Recall@10 → If you're showing a list

###  Morphology Score

**What does it measure?** Sensitivity to Turkish suffixes

**Test pairs:**
```python
morph_pairs = [
    ("geliyorum", "gelmekteyim"),      # I'm coming (different forms)
    ("gidecek", "gider"),              # Will go / goes
    ("yaptım", "yapıyorum"),           # I did / I'm doing
    ("okuyor", "okumakta"),            # Reading (different forms)
    ("koşacağım", "koşarım"),          # I will run / I run
    ("araba", "arabalar"),             # Car / cars
    ("evdeyim", "evde olmak")          # I'm at home (different forms)
]
```

**Calculation:**
```python
# Calculate cosine similarity for each pair
similarities = []
for pair in morph_pairs:
    vec1 = model.encode(pair[0])
    vec2 = model.encode(pair[1])
    sim = cosine_similarity([vec1], [vec2])[0][0]
    similarities.append(sim)

morph_score = np.mean(similarities)
```

**Interpretation:**
- Score > 0.9 → Excellent Turkish understanding
- Score 0.7-0.9 → Good
- Score < 0.7 → Weak (treats each suffix as different word)

###  Silhouette Score

**What does it measure?** How organized is the embedding space?
```python
kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
labels = kmeans.fit_predict(doc_vectors)
sil_score = silhouette_score(doc_vectors, labels)
```

**Interpretation:**
- Close to +1 → Clusters are well separated
- Close to 0 → Clusters overlap
- Close to -1 → Incorrectly clustered

---

##  Results: Champions and Surprises

###  Complete Results Table
```
┌─────────────────────┬──────┬──────────┬────────────┬─────────────┬────────┬──────────┬──────────┬──────────┬───────────┐
│ Model               │ Dim  │ Time (s) │ Silhouette │ Morph Score │  MRR   │ Recall@1 │ Recall@3 │ Recall@5 │ Recall@10 │
├─────────────────────┼──────┼──────────┼────────────┼─────────────┼────────┼──────────┼──────────┼──────────┼───────────┤
│ BGE-M3              │ 1024 │  50.59   │   0.0366   │   0.8113    │ 0.0338 │  1.12%   │  3.24%   │  4.91%   │   7.66%   │
│ Multi-MiniLM-L12-v2 │  384 │  15.81   │   0.0758   │   0.9284    │ 0.0200 │  0.70%   │  1.93%   │  2.72%   │   4.34%   │
│ all-mpnet-base-v2   │  768 │  25.00   │   0.1185   │   0.7460    │ 0.0084 │  0.30%   │  0.78%   │  1.29%   │   1.85%   │
└─────────────────────┴──────┴──────────┴────────────┴─────────────┴────────┴──────────┴──────────┴──────────┴───────────┘
```

###  Visual Analysis

#### 1. Performance Metrics (2×2 Grid)

![Performance Report](performans_metrikleri_raporu.png)

**What we see:**
- **MRR chart:** All bars are short (low values) → Domain is very challenging
- **Recall@1 chart:** BGE-M3 clearly ahead but still low
- **Morph Score chart:** Multi-MiniLM champion 🏆
- **Silhouette chart:** all-mpnet first but this is misleading

#### 2. Speed vs Quality Trade-off (Scatter Plot)

![Speed vs Quality](performans_vs_hiz.png)

**Analysis:**
- **Top left = Ideal zone** (fast + quality)
- **BGE-M3:** Top right (slow but quality)
- **Multi-MiniLM:** Bottom left (fast but medium MRR)
- **all-mpnet:** Lost in the middle (neither fast nor quality)

**Decision guide:**
- Real-time system → Multi-MiniLM
- Offline batch → BGE-M3

#### 3. Radar Chart: Model Profiles

![Radar Profile](modellerin_radar_profili.png)

**Character analysis:**

 **BGE-M3:** "Slow but Effective"
- High MRR, low speed
- Ideal for batch processing in large projects

 **Multi-MiniLM:** "Fast and Turkish-Specialized"
- High speed and morph score
- Perfect for real-time applications

 **all-mpnet:** "Organized but Wrong"
- Only good silhouette
- Don't use for Turkish

---

##  Surprising Findings and Analysis

###  Finding 1: Why Are MRR Values So Low?

**Expectation:** MRR > 0.5 (correct answer in top 2)  
**Reality:** MRR = 0.008-0.033 (correct answer at rank 30-120)

**3 Reasons:**

1. **Domain Gap**
   - Models trained on Wikipedia, books, news
   - Medical terminology is less than 1% of training data
   - Terms like "pyogenic bacteria" rarely seen

2. **Negative Sampling Difficulty**
   - Randomly selected "wrong" answers are actually related
   - Both contain medical terms → Model confuses them
   - Very similar to real-world scenario (good test!)

3. **Lack of Fine-tuning**
   - General-purpose models weak in specific domains
   - 5-10x improvement expected with fine-tuning

> ** Practical lesson:** Don't panic if you see MRR < 0.1. Normal for domain-specific datasets. Fine-tuning is essential!

###  Finding 2: Morphology Champion ≠ Retrieval Champion

| Model | Morph Score | MRR | Relationship |
|-------|-------------|-----|--------------|
| Multi-MiniLM | 🥇 0.9284 | 🥈 0.0200 | Inverse correlation! |
| BGE-M3 | 🥈 0.8113 | 🥇 0.0338 | |

**Why?**

**Required for morphology:**
- Surface-level similarity ("geliyorum" ≈ "gelmekteyim")
- Grammar rules
- Syntax patterns

**Required for retrieval:**
- Deep semantic understanding
- Context awareness
- Domain knowledge

**Analogy:**
> Morphology = Recognizing word **forms**  
> Retrieval = Understanding word **meanings**

### 🇬🇧 Finding 3: English Model's Turkish Fiasco

**all-mpnet-base-v2 report card:**
-  MRR: 0.0084 (last place)
-  Morph: 0.7460 (last place)
-  Recall@1: 0.30% (last place)
-  Silhouette: 0.1185 (1st place) 🤔

**Why high silhouette but low others?**

Silhouette measures "organization", not "correctness". The model organized vectors nicely but **organized them wrongly**.

**Analogy:**
> You organized books by color (well organized)  
> But people searching by topic can't find them (wrongly organized)

**Lesson:** Don't trust a single metric!

###  Finding 4: Dramatic Speed Difference

| Model | Time | vs Multi-MiniLM |
|-------|------|-----------------|
| Multi-MiniLM | 15.81s | 1.0x (baseline) |
| all-mpnet | 25.00s | 1.6x slower |
| BGE-M3 | 50.59s | **3.2x slower** |

**Real-world impact:**

Processing 1000 queries:
- Multi-MiniLM: ~4.4 hours
- all-mpnet: ~7 hours
- BGE-M3: ~14 hours

**In real-time systems:**
- 50ms vs 160ms per user makes a difference
- 100 concurrent users = server struggles

---

##  Decision Guide: Which Model Should I Choose?

###  Scenario-Based Recommendations

#### Scenario 1: Customer Support Chatbot (Real-time)

**Requirements:**
-  Speed critical (users won't wait)
-  Turkish morphology important (users write differently)
-  Sufficient accuracy (doesn't need to be perfect)

**Choice:**  **Multi-MiniLM-L12-v2**

**Why:**
- 3.2x faster (vs BGE-M3)
- Morphology champion (0.9284)
- Sufficient MRR (0.0200)
- Small vectors = low RAM

**Example implementation:**
```python
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Encode all KB answers (offline)
kb_answers = ["answer1", "answer2", ...]
answer_vectors = model.encode(kb_answers)

# Create FAISS index
index = faiss.IndexFlatIP(384)
faiss.normalize_L2(answer_vectors)
index.add(answer_vectors)

# When user question arrives (online)
def get_answer(user_question):
    q_vec = model.encode([user_question])
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, k=3)
    return [kb_answers[i] for i in I[0]]
```

#### Scenario 2: Medical Document Search Engine (Offline)

**Requirements:**
-  Quality critical (wrong result = critical error)
-  Speed secondary (batch processing)
-  Very specific domain

**Choice:**  **BGE-M3 + Fine-tuning**

**Why:**
- Best MRR (0.0338)
- Large model = more capacity
- Speed irrelevant in batch processing

**Fine-tuning example:**
```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load model
model = SentenceTransformer('BAAI/bge-m3')

# Prepare medical Q&A pairs
train_examples = [
    InputExample(texts=['What is an abscess?', 'inflammation caused by pyogenic bacteria']),
    InputExample(texts=['High blood pressure...', 'hypertension...']),
    # ... at least 1000 examples
]

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Train with contrastive loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=100
)

# Save
model.save('bge-m3-medical-turkish')
```

#### Scenario 3: E-commerce Product Search

**Requirements:**
- 🇹🇷 Turkish variations (tişört/tshirt, çorap/sock)
-  Medium speed
-  Lots of products

**Choice:**  **Multi-MiniLM-L12-v2**

**Why:**
- Morphology champion (users write differently)
- Fast
- Small vectors = millions of products can be indexed

#### Scenario 4: Multilingual Platform (TR + EN + DE)

**Requirements:**
-  Cross-lingual search
-  Single model for multiple languages

**Choice:**  **BGE-M3**

**Why:**
- 100+ language support
- Good cross-lingual alignment
- Single embedding space

---

##  Running the Code Guide

###  Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install sentence-transformers faiss-cpu scikit-learn pandas torch matplotlib seaborn

# If you have GPU
pip install faiss-gpu  # instead of faiss-cpu
```

###  Quick Start
```bash
# 1. Clone the code
git clone [repository-url]
cd embedding-comparison

# 2. Prepare data.json (or use fallback sample data)
# 3. Run
python compare_embedding_v2.py

# 4. Results
#  Table in terminal
#  3 visualizations saved as PNG
```

###  Dataset Format

**MedTurkQuaD JSON structure:**
```json
{
  "data": [
    {
      "title": "Medical Topic",
      "paragraphs": [
        {
          "context": "Medical text context...",
          "qas": [
            {
              "question": "What causes abscess?",
              "answers": [
                {
                  "text": "pyogenic bacteria",
                  "answer_start": 42
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

###  Customization Options

#### Add New Model
```python
models_to_test = {
    'Multi-MiniLM-L12-v2': SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
    'BGE-M3': SentenceTransformer('BAAI/bge-m3'),
    'all-mpnet-base-v2': SentenceTransformer('sentence-transformers/all-mpnet-base-v2'),
    'YOUR-MODEL': SentenceTransformer('your-model-name')  # Add here
}
```

#### Adjust Turkish Morphology Tests
```python
# Add more challenging pairs
morph_pairs = [
    ("geliyorum", "gelmekteyim"),
    ("custom_word1", "custom_word2"),  # Add your own
]
```

#### Change Evaluation Metrics
```python
# Adjust recall@k values
recall_at = [1, 3, 5, 10, 20]  # Add @20 if needed
```

---

##  Understanding the Visualizations

### 1. Performance Metrics Report (4 subplots)

**Purpose:** Compare all models across 4 key metrics

**How to read:**
- Taller bars = better (except Silhouette, see below)
- Look for consistent patterns across metrics
- Single high bar doesn't mean best overall

### 2. Performance vs Speed Scatter Plot

**Purpose:** Trade-off analysis

**Quadrants:**
- **Top-left:** Fast and accurate (ideal but rare)
- **Top-right:** Slow but accurate (batch processing)
- **Bottom-left:** Fast but less accurate (real-time with compromise)
- **Bottom-right:** Slow and inaccurate (avoid!)

### 3. Radar Chart: Model Profiles

**Purpose:** Holistic view of strengths/weaknesses

**Reading tips:**
- Larger area = better overall (but check which dimensions!)
- Look for spikes = strong specialization
- Balanced polygon = well-rounded model

---

##  Advanced Topics

### Fine-tuning for Your Domain

**When to fine-tune:**
- MRR < 0.1 on your data
- Your domain very different from general text
- You have 1000+ labeled examples

**Simple fine-tuning recipe:**
```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. Prepare training data
train_examples = []
for query, positive, negative in your_data:
    train_examples.append(InputExample(texts=[query, positive, negative]))

# 2. Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 3. Define loss
train_loss = losses.TripletLoss(model)

# 4. Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='fine-tuned-model'
)
```

### Hybrid Search: Combining Multiple Models
```python
def hybrid_search(query, alpha=0.7):
    # Fast model for initial filtering
    fast_results = multi_minilm.search(query, k=100)
    
    # Slow model for re-ranking top results
    reranked = bge_m3.rerank(query, fast_results)
    
    return reranked[:10]
```

### Monitoring Model Performance
```python
import mlflow

# Log metrics during evaluation
mlflow.log_metric("mrr", mrr_score)
mlflow.log_metric("recall_at_1", recall_1)
mlflow.log_artifact("performance_plot.png")
```

---

##  Troubleshooting

### Common Issues

#### 1. Out of Memory Error

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Reduce batch size in encoding
model.encode(texts, batch_size=8)  # default is 32

# Or use CPU
model = SentenceTransformer('model-name', device='cpu')
```

#### 2. FAISS Installation Issues

**Windows:**
```bash
# Use conda instead of pip
conda install -c conda-forge faiss-cpu
```

**macOS (M1/M2):**
```bash
conda install -c conda-forge faiss-cpu
```

#### 3. Slow Encoding

**Check GPU usage:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(model.device)  # Should be 'cuda'
```

**Force GPU:**
```python
model = SentenceTransformer('model-name', device='cuda')
```

#### 4. Different Results on Each Run

**Ensure reproducibility:**
```python
import random, numpy as np, torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
```

---

##  Further Reading

### Academic Papers

- **MTEB:** [Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316)
- **BGE-M3:** [BGE M3-Embedding](https://arxiv.org/abs/2402.03216)
- **Sentence-BERT:** [Sentence Embeddings using Siamese Networks](https://arxiv.org/abs/1908.10084)

### Practical Guides

- [HuggingFace Sentence Transformers Docs](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Fine-tuning Guide](https://www.sbert.net/docs/training/overview.html)

### Related Projects

- **MTEB Leaderboard:** [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- **Sentence Transformers:** [https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)

---

##  Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more Turkish embedding models
- [ ] Test on other Turkish domains (legal, finance)
- [ ] Implement cross-lingual evaluation
- [ ] Add interactive dashboard
- [ ] Benchmark on GPU vs CPU

**How to contribute:**
1. Fork the repository
2. Create feature branch (`git checkout -b feature/NewModel`)
3. Commit changes (`git commit -m 'Add new model'`)
4. Push to branch (`git push origin feature/NewModel`)
5. Open Pull Request

---





**💬 Have questions? Start a discussion!**

**🐛 Found a bug? Open an issue!**
