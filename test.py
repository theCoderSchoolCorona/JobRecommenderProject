import pandas as pd
import sklearn
import ast
import scipy

df = pd.read_parquet("hf://datasets/batuhanmtl/job-skill-set/data/train-00000-of-00001.parquet")
df.to_csv('job-skills.csv')


job_description_encoder = sklearn.feature_extraction.text.TfidfVectorizer()
jobDescription = job_description_encoder.fit_transform(df['job_description'])

job_skill_set_encoder = sklearn.preprocessing.MultiLabelBinarizer()
jobSkillset = job_skill_set_encoder.fit_transform(df['job_skill_set'].apply(ast.literal_eval))

job_category_encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False)
jobCategory = job_category_encoder.fit_transform(df[['category']])

job_title_encoder = sklearn.feature_extraction.text.TfidfVectorizer()
y = job_title_encoder.fit_transform(df['job_title'])
print(y)

x = scipy.sparse.hstack([jobDescription, jobSkillset, jobCategory])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state = 11)
print(f"=====x_train here:========== \n{x_train[0]}")
print(f"=====y_train here:========== \n{y_train[0]}")
print(x)

