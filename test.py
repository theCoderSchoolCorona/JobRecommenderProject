import pandas as pd
import sklearn
import ast

df = pd.read_parquet("hf://datasets/batuhanmtl/job-skill-set/data/train-00000-of-00001.parquet")
df.to_csv('job-skills.csv')



job_title_encoder = sklearn.feature_extraction.text.TfidfVectorizer()
y = job_title_encoder.fit_transform(df['job_title'])
print(y)

job_description_encoder = sklearn.feature_extraction.text.TfidfVectorizer()
df["job_description_encoded"] = job_description_encoder.fit_transform(df['job_description'])

job_skill_set_encoder = sklearn.preprocessing.MultiLabelBinarizer()
df["job_skill_set_encoded"] = job_skill_set_encoder.fit_transform(df['job_skill_set'])
