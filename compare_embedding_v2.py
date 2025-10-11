# ==========================================================
# GEREKLİ KÜTÜPHANELERİN İÇE AKTARILMASI
# (Bu kütüphanelerin önceden yüklenmiş olması gerekir)
# pip install sentence-transformers faiss-cpu scikit-learn pandas torch matplotlib seaborn
# ==========================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import random
import pandas as pd
import torch
import sys

# ==========================================================
# 0. TEMEL AYARLAR VE SABİTLER
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Hesaplamalar için kullanılacak cihaz: {device}")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ==========================================================
# 1. VERİ YÜKLEME VE İŞLEME FONKSİYONLARI
# ==========================================================
def load_data(file_path):
    """Belirtilen yoldan JSON verisini yükler veya örnek veri oluşturur."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"'{file_path}' dosyası başarıyla yüklendi.")
        return data
    except FileNotFoundError:
        print(f"Uyarı: '{file_path}' dosyası bulunamadı. Örnek veri seti oluşturuluyor.")
        return {
            "data": [{"paragraphs": [{"qas": [
                {"question": "Türkiye'nin başkenti neresidir?", "answers": [{"text": "Ankara"}]},
                {"question": "İstanbul hangi kıtalar üzerindedir?", "answers": [{"text": "Asya ve Avrupa"}]},
                {"question": "Ege'nin incisi olarak bilinen şehir hangisidir?", "answers": [{"text": "İzmir"}]},
                {"question": "Kapadokya hangi ilimizdedir?", "answers": [{"text": "Nevşehir"}]},
                {"question": "Anıtkabir nerededir?", "answers": [{"text": "Ankara"}]},
                {"question": "En kalabalık şehrimiz hangisidir?", "answers": [{"text": "İstanbul"}]}
            ]}]}]
        }
    except json.JSONDecodeError:
        print(f"Hata: Dosya içeriği geçerli bir JSON formatında değil: {file_path}")
        return None


def process_qa_data(qa_data):
    """QA verisini, retrieval test formatına dönüştürür."""
    all_queries, all_positives, all_negatives = [], [], []
    if not qa_data or 'data' not in qa_data:
        return [], [], []
    for doc in qa_data.get('data', []):
        for paragraph in doc.get('paragraphs', []):
            for qa_pair in paragraph.get('qas', []):
                if qa_pair.get('question') and qa_pair.get('answers'):
                    all_queries.append(qa_pair['question'])
                    all_positives.append(qa_pair['answers'][0]['text'])
    num_pairs = len(all_positives)
    if num_pairs < 2:
        return all_queries, all_positives, []
    for i in range(num_pairs):
        idx = i
        while idx == i:
            idx = random.choice(range(num_pairs))
        all_negatives.append(all_positives[idx])
    return all_queries, all_positives, all_negatives


# ==========================================================
# 2. DEĞERLENDİRME METRİKLERİ
# ==========================================================
def compute_recall_at_k(search_results, true_indices, ks=[1, 3, 5, 10]):
    """Recall@K metriklerini hesaplar."""
    recalls = {}
    num_queries = len(true_indices)
    for k in ks:
        correct_predictions = 0
        if search_results.shape[1] < k: continue
        for i in range(num_queries):
            if true_indices[i] in search_results[i, :k]:
                correct_predictions += 1
        recalls[f"recall@{k}"] = correct_predictions / num_queries if num_queries > 0 else 0
    return recalls


def compute_mrr(search_results, true_indices):
    """Mean Reciprocal Rank (MRR) metriğini hesaplar."""
    rr_sum = 0
    num_queries = len(true_indices)
    for i in range(num_queries):
        ranks = np.where(search_results[i] == true_indices[i])[0]
        if len(ranks) > 0:
            rr_sum += 1 / (ranks[0] + 1)
    return rr_sum / num_queries if num_queries > 0 else 0


# ==========================================================
# 3. MORFOLOJİK BENZERLİK TEST SETİ
# ==========================================================
morph_pairs = [
    ("geliyorum", "gelmekteyim"), ("gidecek", "gider"), ("yaptım", "yapıyorum"),
    ("okuyor", "okumakta"), ("koşacağım", "koşarım"), ("araba", "arabalar"), ("evdeyim", "evde olmak")
]


def morphology_test(model):
    """Modelin morfolojik benzerlik yeteneğini test eder."""
    all_words = [item for pair in morph_pairs for item in pair]
    vecs = model.encode(all_words, convert_to_numpy=True, device=device, show_progress_bar=False)
    similarities = [1 - pairwise_distances([vecs[i]], [vecs[i + 1]], metric="cosine")[0][0] for i in
                    range(0, len(vecs), 2)]
    return float(np.mean(similarities)) if similarities else 0.0


# ==========================================================
# 4. ANA VERİ YÜKLEME VE HAZIRLAMA ADIMI
# ==========================================================
file_name = "data.json"
qa_data = load_data(file_name)
if not qa_data: sys.exit()
queries, positives, negatives = process_qa_data(qa_data)
if not queries or not positives or not negatives:
    print("Test verisi oluşturulamadı. Yeterli Soru-Cevap çifti olmayabilir.")
    sys.exit()
documents = positives + negatives
positive_indices = list(range(len(positives)))

# ==========================================================
# 5. TEST EDİLECEK MODELLERİN TANIMLANMASI
# ==========================================================
all_model_names = {
    "Multi-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BGE-M3": "BAAI/bge-m3",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
}
models_to_test = {}
print("--- Modeller Yükleniyor ---")
for name, path in all_model_names.items():
    try:
        print(f"Yükleniyor: {name} ({path})")
        models_to_test[name] = SentenceTransformer(path, device=device)
    except Exception as e:
        print(f"HATA: {name} modeli yüklenemedi. Atlanıyor. Hata: {e}")

if not models_to_test:
    print("Test edilecek hiçbir model başarıyla yüklenemedi.")
    sys.exit()
results = []

# ==========================================================
# 6. MODELLERİ DEĞERLENDİRME DÖNGÜSÜ
# ==========================================================
print("\n--- Model Değerlendirme Süreci Başlatılıyor ---")
for model_name, model in models_to_test.items():
    print(f"\n=== Model Test Ediliyor: {model_name} ===")
    start_time = time.time()
    query_vectors = model.encode(queries, convert_to_numpy=True, show_progress_bar=True)
    doc_vectors = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
    duration = time.time() - start_time
    print(f"Embedding süresi: {duration:.2f} saniye")

    dim = query_vectors.shape[1]
    k_search = len(documents)
    if k_search == 0: continue

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(doc_vectors)
    faiss.normalize_L2(query_vectors)
    index.add(doc_vectors)
    D, I = index.search(query_vectors, k=k_search)

    recalls = compute_recall_at_k(I, positive_indices)
    mrr = compute_mrr(I, positive_indices)

    sil_score = np.nan
    unique_vectors = np.unique(doc_vectors, axis=0)
    if len(unique_vectors) > 1:
        n_clusters = min(2, len(unique_vectors))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(doc_vectors)
        if len(np.unique(kmeans.labels_)) > 1:
            sil_score = silhouette_score(doc_vectors, kmeans.labels_)

    morph_score = morphology_test(model)

    results.append({
        "model": model_name, "dim": dim, "time_sec": round(duration, 2),
        "silhouette": round(sil_score, 4) if not np.isnan(sil_score) else np.nan,
        "morph_score": round(morph_score, 4), "mrr": round(mrr, 4), **recalls
    })
    print(f"MRR: {mrr:.4f}, Recall@1: {recalls.get('recall@1', 0.0):.4f}")

# ==========================================================
# 7. SONUÇLARI TABLO OLARAK GÖSTERME
# ==========================================================
if results:
    df_results = pd.DataFrame(results).sort_values(by="mrr", ascending=False)
    print("\n\n--- TÜM MODELLERİN KARŞILAŞTIRMALI SONUCLARI ---")
    print(df_results.to_string(index=False))

    # ==========================================================
    # 8. GELİŞMİŞ GÖRSELLEŞTİRME
    # ==========================================================
    if results:
        df_results = pd.DataFrame(results).sort_values(by="mrr", ascending=False)
        print("\n\n--- TÜM MODELLERİN KARŞILAŞTIRMALI SONUCLARI ---")
        print(df_results.to_string(index=False))

        print("\n--- Gelişmiş Karşılaştırma Grafikleri Oluşturuluyor ---")
        sns.set_theme(style="whitegrid")

        # --- 1. Performans Metrikleri (4'ü 1 arada) ---
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Model Performans Metrikleri', fontsize=20, y=1.03)

        metrics_to_plot = {
            "mrr": ("MRR Karşılaştırması", axes[0, 0]),
            "recall@1": ("Recall@1 Karşılaştırması", axes[0, 1]),
            "morph_score": ("Morfoloji Skoru", axes[1, 0]),
            "silhouette": ("Silhouette Skoru", axes[1, 1])
        }

        for metric, (title, ax) in metrics_to_plot.items():
            df_sorted = df_results.dropna(subset=[metric]).sort_values(by=metric, ascending=False)
            sns.barplot(x=metric, y="model", data=df_sorted, palette="viridis", ax=ax, hue="model", legend=False)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            # Barların üzerine değerleri yazdır
            for container in ax.containers:
                ax.bar_label(container, fmt='%.4f', padding=3, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig("performans_metrikleri_raporu.png", dpi=150)
        print("'performans_metrikleri_raporu.png' olarak kaydedildi.")

        # --- 2. Performans vs. Hız Ödünleşim Grafiği (Scatter Plot) ---
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df_results,
            x="time_sec",
            y="mrr",
            size="dim",  # Nokta boyutu embedding boyutunu temsil etsin
            hue="model",  # Her model farklı renkte olsun
            sizes=(100, 1000),
            alpha=0.8,
            palette="muted"
        )
        # Noktaların yanına model isimlerini yazdır
        for i in range(df_results.shape[0]):
            plt.text(df_results['time_sec'][i] + 0.5, df_results['mrr'][i], df_results['model'][i],
                     horizontalalignment='left', size='medium', color='black', weight='semibold')

        plt.title("Performans (MRR) vs. Hız (Saniye) Trade-off", fontsize=16)
        plt.xlabel("Embedding Süresi (Saniye) - Düşükse Daha Hızlı", fontsize=12)
        plt.ylabel("MRR Skoru - Yüksekse Daha İyi", fontsize=12)
        plt.legend(title="Model Boyutu (dim)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("performans_vs_hiz.png", dpi=150)
        print("'performans_vs_hiz.png' olarak kaydedildi.")

        # --- 3. Radar Grafiği ---
        from sklearn.preprocessing import MinMaxScaler

        radar_metrics = ['mrr', 'recall@1', 'morph_score', 'silhouette']
        # Hızı temsil etmek için sürenin tersini alalım (1/time), böylece yüksek değer daha iyi olur
        df_radar = df_results.copy()
        df_radar['speed'] = 1 / df_radar['time_sec']
        radar_metrics.append('speed')

        # Metrikleri 0-1 arasına ölçekle
        scaler = MinMaxScaler()
        df_radar[radar_metrics] = scaler.fit_transform(df_radar[radar_metrics].fillna(0))

        labels = radar_metrics
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Kapatmak için ilk açıyı sona ekle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        for index, row in df_radar.iterrows():
            values = row[labels].values.tolist()
            values += values[:1]  # Kapatmak için ilk değeri sona ekle
            ax.plot(angles, values, label=row['model'], linewidth=2)
            ax.fill(angles, values, alpha=0.1)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        plt.title("Modellerin Genel Yetenek Profili (Radar Grafiği)", size=20, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.savefig("modellerin_radar_profili.png", dpi=150)
        print("'modellerin_radar_profili.png' olarak kaydedildi.")

    else:
        print("Hiçbir sonuç üretilmedi.")