import os
from screen import Screen
Screen()
from model import train_autoencoder, generate_job_embeddings, save_model, data_preprocess
from inf import load_model
from inf import recommend_jobs

if not os.path.exists("save_dir"):
    df,x,encoders = data_preprocess()
    autoencoder, encoder, history =train_autoencoder(x,epochs=50,batch_size=32,validation_split=0.2)
    job_embeddings=generate_job_embeddings(encoder,x)
    save_model(encoder, encoders, df, job_embeddings)
else:
    encoder,encoders,job_embeddings,df= load_model()

Screen(encoder, encoders, job_embeddings,df)

print(recommend_jobs(Screen.Entry4.get(),Screen.Entry3.get(),Screen.Entry2.get(),Screen.Entry1.get(), encoder, encoders, job_embeddings, df, top_n=5))