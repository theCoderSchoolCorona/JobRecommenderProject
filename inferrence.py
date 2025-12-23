import pandas as pd
import sklearn
import ast
import scipy
import keras


def generate_job_embeddings(encoder, X):
    """
    Generate embeddings for all jobs in the dataset.
    
    Args:
        encoder: Trained encoder model
        X: Sparse feature matrix
    
    Returns:
        embeddings: Dense matrix of job embeddings (n_jobs × embedding_dim)
    """
    print("\n" + "=" * 60)
    print("STEP 3: INFERENCE SETUP")
    print("=" * 60)
    
    print("\n[3.1] Generating embeddings for all jobs...")
    X_dense = X.toarray().astype('float32')
    embeddings = encoder.predict(X_dense, verbose=0)
    
    return embeddings


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
    X_skills = encoders['skills'].transform([user_skills])
    
    # Encode category using the SAME OneHotEncoder
    # Must be a DataFrame with the same column name
    X_cat = encoders['category'].transform(user_category)

    
    x_title = encoders['title'].transform([user_title])
    
    # Combine in the SAME order as training
    X_user = encoders([X_desc, X_skills, X_cat, x_title])
    
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
    similarities = sklearn.metrics.pairwise.cosine_similarity(job_embeddings, user_embedding)[0]
    
    # Step 4: Get top N indices
    top_indices = similarities.argsort()[::-1][:top_n]
    
    # Build recommendations DataFrame
    recommendations = df.iloc[top_indices][['job_title', 'category', 'job_description']].copy()
    recommendations['similarity_score'] = similarities[top_indices]
    recommendations['job_description'] = recommendations['job_description'].str[:200] + '...'
    
    return recommendations