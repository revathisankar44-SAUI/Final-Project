import os
import csv
import gc
import time
import psutil
import warnings
import random
import hashlib
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow as pa
import pyarrow.parquet as pq
from collections import Counter
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')

# ── LOCAL PATHS ──
project_path = './amazon_project'
os.makedirs(project_path, exist_ok=True)

PARQUET_PATH = f'{project_path}/reviews_clean_sample.parquet'
FLAGGED_PATH = f'{project_path}/reviews_flagged_sample.csv'
CSV_PATH     = f'{project_path}/amazon_reviews.csv'

# ── PIPELINE LIMITS (OUT-OF-CORE ENFORCED) ──
SAMPLE_SIZE      = 50_000   # Stratified sample size for BERTopic/EDA
MAX_DOWNLOAD_ROWS= 690_000  # Total rows to process (10% limit)
BERT_TRAIN       = 10_000   # DistilBERT training subset cap
BERT_EVAL        = 2_000    # DistilBERT eval subset cap
LDA_SIZE         = 20_000   # LDA topic-modeling sample cap
BERTOPIC_SIZE    = 10_000   # BERTopic fit cap

def ram():
    return psutil.virtual_memory().percent

def ram_guard(label='', threshold=85):
    """Warn loudly if RAM exceeds threshold."""
    pct = ram()
    if pct > threshold:
        print(f"⚠️  RAM CRITICAL at {label}: {pct}% — running gc.collect()")
        gc.collect()
    return pct

def clean_text(text):
    """Clean review text by lowercasing, removing URLs, HTML tags, and non-alphabetic chars."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return ' '.join(text.split())


def main():
    print("=========================================================")
    print("   AMAZON REVIEWS SUSPICIOUS DETECTION LOCAL PIPELINE")
    print("=========================================================")
    print(f"Project Path: {project_path}")
    print(f"RAM Usage: {ram()}%")
    
    # ── STEP 1: Verify CSV data source ──
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"❌ Amazon reviews dataset not found at {CSV_PATH}.\n"
            "Please place your CSV file at that path before running the pipeline."
        )
    print(f"✅ Using Amazon reviews CSV dataset: {CSV_PATH}")
    data_source = 'local_csv'

    # ── STEP 2: Phase 1 Out-of-Core Filtering ──
    print("\n--- PHASE 1: COMPILING SIGNALS ---")
    
    print("Pass 1: Counting reviewer burst dates globally...")
    burst_counter = Counter()
    total_rows = 0
    for chunk in pd.read_csv(CSV_PATH, usecols=['customer_id', 'review_date'], chunksize=50000, on_bad_lines='skip', dtype=str):
        chunk = chunk.dropna()
        pairs = zip(chunk['customer_id'], chunk['review_date'].str.strip())
        burst_counter.update(pairs)
        total_rows += len(chunk)
    print(f"✅ Pass 1 done. Counted {len(burst_counter):,} unique user-date pairs across {total_rows:,} rows. RAM: {ram()}%")

    print("\nPass 2: Computing VADER mismatch and splitting clean/flagged in chunks...")
    # Pre-compile flat keys dictionary for vectorized burst lookup
    burst_dict = {f"{cust}_{dt}": cnt for (cust, dt), cnt in burst_counter.items()}
    analyzer = SentimentIntensityAnalyzer()
    WEIGHTS = {
        'flag_vine':0.5, 'flag_unverified':0.5, 'flag_too_short':1.0,
        'flag_generic':1.5, 'flag_unhelpful':1.5, 'flag_burst':2.0, 'flag_mismatch':1.5
    }
    THRESHOLD = 2.0
    flag_cols = list(WEIGHTS.keys())
    BOT_PHRASES = (
        'five stars|perfect product|excellent product|as described|'
        'highly recommend this product|exactly as advertised|'
        'great seller|fast shipping great product|'
        'received this product for free|received this product in exchange|'
        'i received this product|disclaimer|complimentary|'
        'honest and unbiased|in exchange for my honest'
    )
    MISMATCH_T = {1.0:1.2, 2.0:0.9, 3.0:0.7, 4.0:0.9, 5.0:1.2}

    if os.path.exists(FLAGGED_PATH): os.remove(FLAGGED_PATH)
    if os.path.exists(PARQUET_PATH): os.remove(PARQUET_PATH)

    pq_writer = None
    total_clean = 0
    total_flagged = 0
    chunk_idx = 0
    first_flagged = True

    mismatch_counts = {s: 0 for s in MISMATCH_T}
    mismatch_flags = {s: 0 for s in MISMATCH_T}

    for chunk in pd.read_csv(CSV_PATH, chunksize=50000, on_bad_lines='skip', dtype=str):
        chunk_idx += 1
        chunk['star_rating']   = pd.to_numeric(chunk['star_rating'],   errors='coerce')
        chunk['helpful_votes'] = pd.to_numeric(chunk['helpful_votes'], errors='coerce').fillna(0).astype(np.int32)
        chunk['total_votes']   = pd.to_numeric(chunk['total_votes'],   errors='coerce').fillna(0).astype(np.int32)
        chunk['review_body']   = chunk['review_body'].fillna('')
        chunk['review_length'] = chunk['review_body'].str.len().astype(np.int32)
        
        # Signals with safe string casting to avoid type errors
        chunk['flag_vine'] = chunk['vine'].astype(str).str.strip().str.upper() == 'Y'
        chunk['flag_unverified'] = chunk['verified_purchase'].astype(str).str.strip().str.upper() == 'N'
        chunk['flag_too_short'] = chunk['review_length'] < 15
        chunk['flag_generic'] = (
            chunk['review_body'].str.lower().str.contains(BOT_PHRASES, na=False, regex=True) &
            (chunk['review_length'] < 60)
        )
        has_votes = chunk['total_votes'] >= 5
        chunk['helpful_ratio'] = np.where(has_votes, chunk['helpful_votes']/chunk['total_votes'], np.nan).astype(np.float32)
        chunk['flag_unhelpful'] = (chunk['helpful_ratio'] < 0.3).fillna(False)
        # Vectorized reviewer burst lookup (50x faster than axis=1 apply)
        keys = chunk['customer_id'].astype(str) + "_" + chunk['review_date'].astype(str).str.strip()
        chunk['flag_burst'] = keys.map(burst_dict).fillna(0) >= 5
        chunk['star_normalised'] = ((chunk['star_rating'] - 3) / 2).astype(np.float32)
        chunk['text_polarity'] = chunk['review_body'].apply(
            lambda x: analyzer.polarity_scores(str(x))['compound']
        ).astype(np.float32)
        chunk['_mt'] = chunk['star_rating'].map(MISMATCH_T).fillna(0.8)
        chunk['flag_mismatch'] = (abs(chunk['text_polarity'] - chunk['star_normalised']) > chunk['_mt']).fillna(False)
        chunk.drop(columns=['_mt'], inplace=True)
        
        for s in MISMATCH_T:
            m = chunk['star_rating'] == s
            mismatch_counts[s] += m.sum()
            mismatch_flags[s] += chunk.loc[m, 'flag_mismatch'].sum()
            
        chunk['suspicion_score'] = sum(chunk[c].astype(np.float32)*w for c,w in WEIGHTS.items()).astype(np.float32)
        
        clean_chunk = chunk[chunk['suspicion_score'] < THRESHOLD].copy()
        flagged_chunk = chunk[chunk['suspicion_score'] >= THRESHOLD].copy()
        
        if not flagged_chunk.empty:
            flagged_chunk.to_csv(FLAGGED_PATH, mode='a', index=False, header=first_flagged)
            first_flagged = False
            total_flagged += len(flagged_chunk)
            
        if not clean_chunk.empty:
            clean_chunk['review_date'] = pd.to_datetime(clean_chunk['review_date'], errors='coerce')
            table = pa.Table.from_pandas(clean_chunk, preserve_index=False)
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(PARQUET_PATH, table.schema, compression='snappy')
            pq_writer.write_table(table)
            total_clean += len(clean_chunk)
            
        if chunk_idx % 3 == 0:
            print(f"   Chunk {chunk_idx}: processed {chunk_idx*50000:,} rows | Clean: {total_clean:,} | Flagged: {total_flagged:,} | RAM: {ram()}%")
            
        del chunk, clean_chunk, flagged_chunk; gc.collect()

    if pq_writer is not None:
        pq_writer.close()

    print(f"✅ Pass 2 complete! Clean count: {total_clean:,} | Flagged count: {total_flagged:,}")

    # ── STEP 3: Phase 2 Sentiment Classification (Out-of-Core SGD) ──
    print("\n--- PHASE 2: SENTIMENT CLASSIFICATION ---")
    print("Initializing HashingVectorizer and SGDClassifiers...")
    tfidf = HashingVectorizer(n_features=2**18, alternate_sign=False, stop_words='english')
    
    lr = SGDClassifier(loss='log_loss', random_state=42)
    rf = SGDClassifier(loss='hinge', random_state=42)
    lr5 = SGDClassifier(loss='log_loss', random_state=42)
    rf5 = SGDClassifier(loss='hinge', random_state=42)

    classes_3 = np.array([0, 1, 2])
    classes_5 = np.array([1, 2, 3, 4, 5])

    test_y3 = []
    test_y5 = []
    lr_preds_3 = []
    rf_preds_3 = []
    lr_preds_5 = []
    rf_preds_5 = []

    parquet_file = pq.ParquetFile(PARQUET_PATH)
    chunk_size = 20000
    batch_idx = 0

    print("Training classifiers chunk-by-chunk...")
    for record_batch in parquet_file.iter_batches(batch_size=chunk_size, columns=['review_id', 'review_body', 'star_rating']):
        batch_idx += 1
        df_chunk = record_batch.to_pandas()
        df_chunk = df_chunk.dropna(subset=['star_rating'])
        df_chunk['star_rating'] = df_chunk['star_rating'].astype(int)
        df_chunk = df_chunk[df_chunk['review_body'].str.strip().str.len() > 0]
        if df_chunk.empty:
            continue
            
        df_chunk['sentiment'] = df_chunk['star_rating'].apply(lambda s: 0 if s<=2 else (1 if s==3 else 2))
        is_train = df_chunk['review_id'].apply(lambda rid: int(hashlib.md5(str(rid).encode()).hexdigest(), 16) % 100 < 80)
        
        train_df = df_chunk[is_train]
        test_df = df_chunk[~is_train]
        
        if not train_df.empty:
            X_train_vec = tfidf.transform(train_df['review_body'])
            lr.partial_fit(X_train_vec, train_df['sentiment'], classes=classes_3)
            rf.partial_fit(X_train_vec, train_df['sentiment'], classes=classes_3)
            lr5.partial_fit(X_train_vec, train_df['star_rating'], classes=classes_5)
            rf5.partial_fit(X_train_vec, train_df['star_rating'], classes=classes_5)
            
        if not test_df.empty:
            X_test_vec = tfidf.transform(test_df['review_body'])
            lr_preds_3.extend(lr.predict(X_test_vec))
            rf_preds_3.extend(rf.predict(X_test_vec))
            lr_preds_5.extend(lr5.predict(X_test_vec))
            rf_preds_5.extend(rf5.predict(X_test_vec))
            test_y3.extend(test_df['sentiment'].tolist())
            test_y5.extend(test_df['star_rating'].tolist())
            
        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx}: processed {batch_idx*chunk_size:,} rows | RAM: {ram()}%")

    yte3 = np.array(test_y3)
    yte5 = np.array(test_y5)
    lr_p3 = np.array(lr_preds_3)
    rf_p3 = np.array(rf_preds_3)
    lr_p5 = np.array(lr_preds_5)
    rf_p5 = np.array(rf_preds_5)

    lr_a3 = accuracy_score(yte3, lr_p3)
    lr_f3 = f1_score(yte3, lr_p3, average='weighted')
    lr_cm = confusion_matrix(yte3, lr_p3)
    lr_a5 = accuracy_score(yte5, lr_p5)

    rf_a3 = accuracy_score(yte3, rf_p3)
    rf_f3 = f1_score(yte3, rf_p3, average='weighted')
    rf_cm = confusion_matrix(yte3, rf_p3)
    rf_a5 = accuracy_score(yte5, rf_p5)

    print(f"\n✅ Logistic Regression (SGD): Acc={lr_a3:.4f} ({lr_a3*100:.1f}%) | F1={lr_f3:.4f}")
    print(f"✅ Linear SVM (SGD): Acc={rf_a3:.4f} ({rf_a3*100:.1f}%) | F1={rf_f3:.4f}")

    # ── STEP 4: DistilBERT Training & Streaming Inference ──
    print("\n--- DISTILBERT INFERENCE CHECK ---")
    import torch
    HAS_GPU = torch.cuda.is_available()
    bert_a3 = None
    bert_f3 = None
    bert_cm = None
    bert_time = None
    bert_p3 = None
    bert_tl = None

    if HAS_GPU:
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        try:
            from transformers import (DistilBertTokenizerFast, DistilBertForSequenceClassification,
                                      TrainingArguments, Trainer)
            from datasets import Dataset
            from torch.utils.data import DataLoader
            
            # Load subset
            df_bert = pd.read_parquet(PARQUET_PATH, columns=['review_id', 'review_body', 'star_rating'])
            df_bert['sentiment'] = df_bert['star_rating'].apply(lambda s: 0 if s<=2 else (1 if s==3 else 2))
            is_train = df_bert['review_id'].apply(lambda rid: int(hashlib.md5(str(rid).encode()).hexdigest(), 16) % 100 < 80)
            
            BT = min(BERT_TRAIN, len(df_bert[is_train]))
            BE = min(BERT_EVAL, len(df_bert[~is_train]))
            
            Xtr = df_bert[is_train]['review_body'].sample(n=BT, random_state=42).tolist()
            ytr3 = df_bert[is_train]['sentiment'].sample(n=BT, random_state=42).tolist()
            Xte = df_bert[~is_train]['review_body'].sample(n=BE, random_state=42).tolist()
            yte3_bert = df_bert[~is_train]['sentiment'].sample(n=BE, random_state=42).tolist()
            bert_tl = yte3_bert

            tok = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            tds = Dataset.from_dict({'text': Xtr, 'label': ytr3})
            eds = Dataset.from_dict({'text': Xte, 'label': yte3_bert})
            fn = lambda b: tok(b['text'], padding='max_length', truncation=True, max_length=128)
            tds = tds.map(fn, batched=True)
            eds = eds.map(fn, batched=True)
            tds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
            eds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

            model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
            args = TrainingArguments(
                output_dir=f'{project_path}/bert_ckpt', num_train_epochs=1,
                per_device_train_batch_size=32, per_device_eval_batch_size=64,
                eval_strategy='epoch', save_strategy='epoch', report_to='none', seed=42, fp16=True
            )
            trainer = Trainer(model=model, args=args, train_dataset=tds, eval_dataset=eds)
            print("Training DistilBERT locally...")
            t0_bert = time.time()
            trainer.train()
            train_time = time.time() - t0_bert

            # Custom PyTorch inference loader
            model.eval()
            model.to('cuda')
            
            class PyTorchDataset(torch.utils.data.Dataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels = labels
                def __getitem__(self, idx):
                    item = {key: val[idx] for key, val in self.encodings.items()}
                    item['labels'] = torch.tensor(self.labels[idx])
                    return item
                def __len__(self):
                    return len(self.labels)

            test_encodings = tok(Xte, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            test_dataset = PyTorchDataset(test_encodings, yte3_bert)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            print("Streaming DistilBERT evaluation loop (preventing CPU/GPU memory stack)...")
            bert_preds = []
            with torch.no_grad():
                for idx, batch in enumerate(test_loader):
                    input_ids = batch['input_ids'].to('cuda')
                    attention_mask = batch['attention_mask'].to('cuda')
                    outputs = model(input_ids, attention_mask=attention_mask)
                    preds = outputs.logits.argmax(-1).cpu().numpy()
                    bert_preds.extend(preds)
                    if idx % 10 == 0:
                        torch.cuda.empty_cache()

            bert_time = train_time + (time.time() - t0_bert)
            bert_p3 = np.array(bert_preds)
            bert_a3 = accuracy_score(yte3_bert, bert_p3)
            bert_f3 = f1_score(yte3_bert, bert_p3, average='weighted')
            bert_cm = confusion_matrix(yte3_bert, bert_p3)
            
            print(f"✅ DistilBERT done | Acc: {bert_a3:.4f}")
            del model, trainer, test_loader, test_dataset, test_encodings
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as err:
            print(f"⚠️ Transformers training/inference skipped or failed: {err}")
    else:
        print("ℹ️ DistilBERT skipped (No CUDA/GPU found)")

    # ── STEP 5: Phase 3 Topic Modeling ──
    print("\n--- PHASE 3: TOPIC MODELING ---")
    
    # Stratified clean sample for topic modeling
    print("Preparing topic modeling dataset...")
    topic_df = pd.read_parquet(PARQUET_PATH, columns=['review_body', 'star_rating']).dropna(subset=['star_rating'])
    topic_df['sentiment'] = topic_df['star_rating'].apply(lambda s: 0 if s<=2 else (1 if s==3 else 2))
    topic_df = topic_df.groupby('sentiment').apply(lambda x: x.sample(n=min(len(x), 17000), random_state=42)).reset_index(drop=True)
    topic_df['clean_text'] = topic_df['review_body'].apply(clean_text)

    # LDA
    print("Fitting LDA model...")
    actual_lda = min(LDA_SIZE, len(topic_df))
    lda_sample = topic_df.sample(n=actual_lda, random_state=42)
    cv = CountVectorizer(max_features=8000, min_df=5, max_df=0.9, stop_words='english')
    dtm = cv.fit_transform(lda_sample['clean_text'])
    lda = LatentDirichletAllocation(n_components=8, random_state=42, max_iter=10)
    lda.fit(dtm)
    print("LDA topic words:")
    feature_names = cv.get_feature_names_out()
    lda_topics = []
    for idx, comp in enumerate(lda.components_):
        top_words = [feature_names[j] for j in comp.argsort()[-8:][::-1]]
        lda_topics.append(top_words)
        print(f"  Topic {idx}: {', '.join(top_words)}")
    del dtm, cv, lda; gc.collect()

    # BERTopic
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        
        print("\nFitting BERTopic on 50k stratified sample...")
        bt_sample = topic_df.sample(n=min(BERTOPIC_SIZE, len(topic_df)), random_state=42)
        docs = bt_sample['clean_text'].tolist()
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        bt_model = BERTopic(embedding_model=embedding_model, min_topic_size=50, nr_topics='auto', verbose=True)
        bt_topics, _ = bt_model.fit_transform(docs)
        
        # Transform globally in chunks
        print("Transforming all clean reviews in chunks of 10,000...")
        bertopic_csv = f"{project_path}/bertopic_assignments.csv"
        if os.path.exists(bertopic_csv): os.remove(bertopic_csv)
        
        parquet_file = pq.ParquetFile(PARQUET_PATH)
        first = True
        for record_batch in parquet_file.iter_batches(batch_size=10000, columns=['review_id', 'review_body']):
            df_chunk = record_batch.to_pandas()
            docs_chunk = df_chunk['review_body'].apply(clean_text).tolist()
            topics, _ = bt_model.transform(docs_chunk)
            df_out = pd.DataFrame({'review_id': df_chunk['review_id'], 'bt_topic': topics})
            df_out.to_csv(bertopic_csv, mode='a', index=False, header=first)
            first = False
            
        print(f"✅ Topics transform done! Saved topic IDs to: {bertopic_csv}")
    except Exception as err:
        print(f"⚠️ BERTopic setup or fit failed: {err}")

    # ── STEP 6: Phase 4 Review Quality Score ──
    print("\n--- PHASE 4: REVIEW QUALITY SCORE ---")
    print("Loading review length and suspicion score metadata...")
    meta = pd.read_parquet(PARQUET_PATH, columns=['review_length', 'suspicion_score', 'helpful_ratio'])
    max_susp = meta['suspicion_score'].max()
    median_ratio = meta.loc[meta['helpful_ratio'].notna(), 'helpful_ratio'].median()

    # Length rank percentile mapping
    lengths_clipped = meta['review_length'].clip(upper=2000)
    unique_lengths = np.sort(lengths_clipped.unique())
    unique_ranks = pd.Series(unique_lengths).rank(pct=True).values
    length_rank_map = dict(zip(unique_lengths, unique_ranks))
    del meta, lengths_clipped; gc.collect()

    QUALITY_PATH = PARQUET_PATH.replace('.parquet', '_quality.parquet')
    pq_quality_writer = None
    parquet_file = pq.ParquetFile(PARQUET_PATH)
    
    print("Calculating quality scores in chunks...")
    QUALITY_WEIGHTS = {
        'length_score': 0.25,
        'sentiment_strength': 0.30,
        'helpfulness_score': 0.25,
        'authenticity_score': 0.20
    }

    for record_batch in parquet_file.iter_batches(batch_size=50000):
        chunk = record_batch.to_pandas()
        chunk['length_score'] = chunk['review_length'].clip(upper=2000).map(length_rank_map).fillna(0.5).astype(np.float32)
        chunk['sentiment_strength'] = chunk['text_polarity'].abs().astype(np.float32)
        chunk['helpfulness_score'] = chunk['helpful_ratio'].fillna(median_ratio).astype(np.float32)
        chunk['authenticity_score'] = (1 - (chunk['suspicion_score'] / max_susp if max_susp > 0 else 0)).astype(np.float32)
        
        chunk['quality_score'] = sum(chunk[col].fillna(0).clip(0, 1) * w for col, w in QUALITY_WEIGHTS.items()).astype(np.float32)
        
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if pq_quality_writer is None:
            pq_quality_writer = pq.ParquetWriter(QUALITY_PATH, table.schema, compression='snappy')
        pq_quality_writer.write_table(table)
        del chunk; gc.collect()

    if pq_quality_writer is not None:
        pq_quality_writer.close()

    os.remove(PARQUET_PATH)
    os.rename(QUALITY_PATH, PARQUET_PATH)
    print("✅ Quality score calculation complete!")

    # ── STEP 7: Phase 5 Visualisations & Actionable Insights ──
    print("\n--- PHASE 5: BUSINESS INSIGHTS ---")
    # Column projection to avoid loading heavy 'review_body' text column for plotting
    df_sample = pd.read_parquet(PARQUET_PATH, columns=['star_rating', 'quality_score', 'sentiment_strength', 'product_category', 'authenticity_score', 'helpful_votes']).sample(n=min(50000, total_clean), random_state=42)
    df_sample['sentiment_label'] = df_sample['star_rating'].apply(
        lambda s: 'Negative' if s<=2 else ('Neutral' if s==3 else 'Positive') if pd.notna(s) else 'Unknown'
    )
    
    # Save a basic dashboard layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df_sample['quality_score'].hist(bins=30, ax=axes[0], color='#1976d2')
    axes[0].set_title("Review Quality Score Distribution")
    axes[0].set_xlabel("Score")
    
    sent_counts = df_sample['sentiment_label'].value_counts()
    sent_counts.plot(kind='bar', ax=axes[1], color=['#2e7d32','#fdd835','#e53935'])
    axes[1].set_title("Sentiment Distribution (Sample)")
    
    plt.tight_layout()
    plt.savefig(f"{project_path}/local_dashboard.png")
    print(f"📊 Dashboard chart saved to: {project_path}/local_dashboard.png")

    print("\n" + "=" * 50)
    print("       LOCAL ANALYSIS BUSINESS INSIGHTS")
    print("=" * 50)
    flagged_cnt = sum(1 for _ in open(FLAGGED_PATH, encoding='utf-8')) - 1
    total_cnt = total_clean + flagged_cnt
    print(f"1️⃣ REVIEW AUTHENTICITY: {flagged_cnt:,} reviews flagged ({flagged_cnt/total_cnt*100:.2f}%)")
    print(f"2️⃣ SENTIMENT PREDICTION: Linear SVM (Acc={rf_a3:.2%}) & Logistic Regression (Acc={lr_a3:.2%})")
    print(f"3️⃣ REVIEW QUALITY: Q5 Mean helpful votes: {df_sample[df_sample['quality_score']>=df_sample['quality_score'].quantile(0.8)]['helpful_votes'].mean():.2f}")
    print("=========================================================")

if __name__ == '__main__':
    main()
