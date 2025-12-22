import pandas as pd

df = pd.read_parquet("hf://datasets/batuhanmtl/job-skill-set/data/train-00000-of-00001.parquet")
df.to_csv('job-skills.csv')

