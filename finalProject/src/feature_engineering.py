import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pathlib import Path

def extract_phobert_features(texts, model_name="vinai/phobert-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
    
    features = []
    batch_size = 32

    texts = [str(t) if not pd.isna(t) else "" for t in texts]
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        features.extend(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    
    return np.array(features)

def extract_features(df):
    df_features = df.copy()
    
    df_features['vid_desc_clean'] = df_features['vid_desc_clean'].fillna('')
    df_features['vid_hashtags_normalized'] = df_features['vid_hashtags_normalized'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Extract TF-IDF features
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=1000)
    text_features = tfidf.fit_transform(
        df_features['vid_desc_clean'] + ' ' + 
        df_features['vid_hashtags_normalized'].apply(lambda x: ' '.join(x))
    )
    
    # Extract PhoBERT features
    print("Extracting PhoBERT embeddings...")
    phobert_features = extract_phobert_features(df_features['vid_desc_clean'])
    
    # Use precomputed viral score
    viral_scores = df_features['viral_score'].values
    
    # Create metadata dictionary
    metadata = {
        'video_ids': df_features['vid_id'].values,
        'post_times': df_features['vid_postTime'].values,
        'hashtag_counts': df_features['hashtag_count'].values,
        'durations': df_features['vid_duration_sec'].values,
        'followers': df_features['user_nfollower'].values,
        'views': df_features['vid_nview'].values,
        'likes': df_features['vid_nlike'].values,
        'comments': df_features['vid_ncomment'].values,
        'shares': df_features['vid_nshare'].values,
        'saves': df_features['vid_nsave'].values
    }
    
    return {
        'tfidf_features': text_features.toarray(),
        'phobert_features': phobert_features,
        'viral_scores': viral_scores,
        'tfidf_vectorizer': tfidf,
        'metadata': metadata
    }

def save_features(features, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dense features using numpy
    np.savez_compressed(
        output_dir / 'dense_features.npz',
        tfidf_features=features['tfidf_features'],
        phobert_features=features['phobert_features'],
        viral_scores=features['viral_scores']
    )
    
    # Save metadata using numpy
    np.savez_compressed(
        output_dir / 'metadata.npz',
        **features['metadata']
    )
    
    # Save the TF-IDF vectorizer using pickle
    with open(output_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(features['tfidf_vectorizer'], f)

def main():
    infile = "finalProject/data/preprocessed_data.csv"
    output_dir = "finalProject/data/features"
    
    print(f"Loading preprocessed data from {infile}...")
    df = pd.read_csv(infile)
    
    print("Extracting features...")
    features = extract_features(df)
    
    print(f"Saving features to {output_dir}...")
    save_features(features, output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
