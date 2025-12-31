import pandas as pd
import sklearn
import scipy
import keras
from rapidfuzz import process, fuzz
import pickle
import numpy


def load_model():
    encoder=keras.models.load_model("save_dir/encoder.keras")
    with open("save_dir/encoders.pkl","rb") as f:
        encoders=pickle.load(f)
    job_embeddings=numpy.load("save_dir/job_embeddings.npy")
    df =pd.read_pickle("save_dir/jobs_df.pkl")
    print("loaded")
    return encoder, encoders, job_embeddings, df

def encode_user_sills_fuzzy(user_skills, skills_encoder, threshold=70):

    known_skills = list(skills_encoder.classes_)
    matched_skills = []
    for skill in user_skills:
        if skill in known_skills:
            matched_skills.append(skill)
        else:
            match, score, _ = process.extractOne(
                skill, known_skills, scorer=fuzz.WRatio
            )
            if score >= threshold:
                matched_skills.append(match)
            else:
                print(f"No good matched skill for {skill}")
    return skills_encoder.transform([matched_skills])

def encode_user_input(user_description, user_skills, user_category, user_title, encoders):
    """
    Encode a user's job preferences into the same feature space as our training data.
    
    This is crucial: we must transform user input EXACTLY the same way we
    transformed the training data, using the SAME fitted encoders.
    
    Args:
        user_description: Text describing desired job (string)
        user_skills: List of skills the user has (list of strings)
        user_category: Preferred job category (string)
        encoders: Dictionary of fitted encoders from preprocessing
    
    Returns:
        X_user: Encoded feature vector for the user (sparse matrix)
    """
    # Encode description using the SAME TF-IDF vectorizer
    # transform() uses vocabulary learned during fit()
    X_desc = encoders['description'].transform([user_description])
    
    # Encode skills using the SAME MultiLabelBinarizer
    # Unknown skills are simply ignored (not in the learned vocabulary)
    X_skills = encode_user_sills_fuzzy(user_skills, encoders["skills"])
    
    # Encode category using the SAME OneHotEncoder
    # Must be a DataFrame with the same column name
    cat_df=pd.DataFrame({'category':[user_category]})
    X_cat = encoders['category'].transform(cat_df)

    
    x_title = encoders['title'].transform([user_title])
    
    # Combine in the SAME order as training
    X_user = scipy.sparse.hstack([X_desc, X_skills, X_cat, x_title])
    
    return X_user


def recommend_jobs(user_description, user_skills, user_category, user_title, 
                   encoder, encoders, job_embeddings, df, top_n=5):
    """
    Recommend jobs based on user input.
    
    Process:
    1. Encode user input into feature vector
    2. Pass through encoder to get user's embedding
    3. Compute cosine similarity with all job embeddings
    4. Return top N most similar jobs
    
    Args:
        user_description: What kind of job the user wants
        user_skills: Skills the user has
        user_category: Preferred category
        encoder: Trained encoder model
        encoders: Dictionary of fitted encoders
        job_embeddings: Pre-computed embeddings for all jobs
        df: Original dataframe with job details
        top_n: Number of recommendations to return
    
    Returns:
        recommendations: DataFrame with top job matches and similarity scores
    """
    
    # Step 1: Encode user input
    X_user = encode_user_input(user_description, user_skills, user_category, user_title, encoders)
    
    # Step 2: Generate user embedding
    X_user_dense = X_user.toarray().astype('float32')
    user_embedding = encoder.predict(X_user_dense, verbose=0)
    
    # Step 3: Compute similarity with all jobs
    # Cosine similarity: 1 = identical, 0 = orthogonal, -1 = opposite
    similarities = sklearn.metrics.pairwise.cosine_similarity(user_embedding, job_embeddings)[0]
    
    # Step 4: Get top N indices
    top_indices = similarities.argsort()[::-1][:top_n]
    
    # Build recommendations DataFrame
    recommendations = df.iloc[top_indices][['job_title', 'category', 'job_description']].copy()
    recommendations['similarity_score'] = similarities[top_indices]
    recommendations['job_description'] = recommendations['job_description'].str[:200] + '...'
    
    return recommendations