import re
import nltk
import warnings
import numpy as np
import contractions
import pandas as pd
from nltk.corpus import stopwords
import surprise
import seaborn as sns
from IPython.display import display
import matplotlib.pyplot as plt
from surprise import accuracy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from surprise import SVD, KNNBasic
from collections import defaultdict
from surprise import Reader, Dataset
from sklearn.metrics import davies_bouldin_score
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

"""
df = pd.read_csv('data.csv', sep='|')
df.columns = ['user','item','rating','review']

def clean_text(text):
    text = contractions.fix(text) 
    return text

for index in range(len(df)):
    df.loc[index,"review"] = clean_text(df.loc[index,"review"])

df.review = df.review.str.replace(';','', regex=True)
df.review = df.review.str.replace('(',' ', regex=True)
df.review = df.review.str.replace(')',' ', regex=True)
df.review = df.review.str.replace('!',' ', regex=True)
df.review = df.review.str.replace(':',' ', regex=True)
df.review = df.review.str.replace('?',' ', regex=True)
df.review = df.review.str.replace('&',' ', regex=True)
df.review = df.review.str.replace('-',' ', regex=True)
df.review = df.review.str.replace('=',' ', regex=True)
df.review = df.review.str.replace('"',' ', regex=True)
df.review = df.review.str.replace(' +',' ', regex=True)
df.review = df.review.str.casefold()
"""

df = pd.read_csv('clean_data_manual.csv', sep='|', index_col=0)
print(df,"\n")
print(df.info(),"\n")

df_1 = df[['item','rating','review']]
df_1 = df_1.groupby("item").agg(list).reset_index()
print(df_1,"\n")

for i in range(len(df_1)):
    a = " ".join(df_1.loc[i,'review'])
    df_1.at[i,'review'] = a.split(' ')
print(df_1,"\n")

stop_words = set(stopwords.words('english'))
for j in range(len(df_1)):
    filtered = [w for w in df_1.loc[j,'review'] if not w.lower() in stop_words]
    df_1.at[j,'review'] = filtered
print(df_1,"\n")

warnings.filterwarnings('ignore')      

tfidf = TfidfVectorizer()
vec = tfidf.fit_transform(df_1['review'].apply(' '.join).to_list())
vectors = pd.DataFrame(vec.todense(), columns=tfidf.get_feature_names())
print(vectors,"\n")

vectors['sum'] = vectors.sum(axis = 1)
df_1['review'] = vectors['sum']

for i in range(len(df_1)):
  df_1.loc[i,'rating'] = sum(df_1.loc[i,'rating'])
print(df_1,"\n")

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df_1[['rating','review']])

"""
sse = []
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k,random_state=0)
    kmeans.fit(scaled_df)
    sse.append(kmeans.inertia_)

sns.set_style("whitegrid")
g = sns.lineplot(x=range(1,21),y=sse)

g.set(xlabel = "Jumlah Cluster",
      ylabel = "Sum Squared Error",
      title = "Elbow Method")
plt.show()
"""

kmeans = KMeans(n_clusters=4, init='random',n_init=5,random_state=0)
kmeans = kmeans.fit(scaled_df)
"""
centroid = kmeans.cluster_centers_
plt.scatter(df_1['rating'],df_1['review'],s=150,c=kmeans.labels_,cmap='gist_rainbow')
plt.scatter(centroid[:,0],centroid[:,1],color='black')
plt.show()
"""

df_2 = df[['user','item','rating']]
df_2 = df_2.groupby('item').agg(list).reset_index()
print(df_2,"\n")
df_2['cluster'] = kmeans.labels_

print("---------------------------------------------------------\n")
dbi = davies_bouldin_score(scaled_df, kmeans.labels_)
print('Davies Boudlin Index = ',dbi,"\n")
print("---------------------------------------------------------\n")

df_2 = df_2.sort_values(by="cluster")
group = df_2.groupby(df_2['cluster'],sort=True)  

for k in df_2['cluster'].unique():
  
  z = group.get_group(k)
  print(z,"\n")
  cv = []
  for a in[SVD(), KNNBasic(sim_options = { "name": "cosine"})]:
      z = z.explode(['user','rating'])
      z = z.groupby('user').agg(list).reset_index()
      z = z.explode(['item','rating'])
      z = z[['user','item','rating']].reset_index(drop=True)
      reader = Reader(rating_scale=(1.0, 10.0))
      data = Dataset.load_from_df(z,reader)
      result = cross_validate(a, data, measures=['RMSE','MAE'], cv=5, verbose=True)
  print("---------------------------------------------------------------------------------\n")
  
"""
clus = int(input("Pilih cluster = "))
df_2= group.get_group(clus)
df_2 = df_2.drop(['cluster'], axis=1)
df_2 = df_2.explode(['user','rating'])
df_2 = df_2.groupby('user').agg(list).reset_index()
df_2 = df_2.explode(['item','rating'])
print(df_2,"\n")
df_2.insert(0,'userID',df_2.index+1)
print(df_2.reset_index(drop=True),'\n')
print(df_2[['userID','item','rating']].reset_index(drop=True),'\n')

unique_ids = df_2['item'].unique()
userid = int(input("Pilih userID = "))
print('\n',df_2[df_2['userID']==userid],'\n')
itemid = df_2.loc[df_2['userID']==userid, 'item']
to_predict = np.setdiff1d(unique_ids,itemid)

df_2 = df_2[['userID','item','rating']].reset_index(drop=True)
reader = Reader(rating_scale=(1.0, 10.0))
data = Dataset.load_from_df(df_2,reader)

algo1 = SVD()
algo1.fit(data.build_full_trainset())
algo2 = KNNBasic(sim_options = { "name": "cosine"})
algo2.fit(data.build_full_trainset())

recs_svd = []
recs_knn = []
for iid in to_predict:
    recs_svd.append((iid, algo1.predict(uid=userid,iid=iid).est))
    recs_knn.append((iid, algo2.predict(uid=userid,iid=iid).est))
    
res_svd = pd.DataFrame(recs_svd, columns=['item', 'predictions']).sort_values('predictions', ascending=False).head(1)
res_knn = pd.DataFrame(recs_knn, columns=['item', 'predictions']).sort_values('predictions', ascending=False).head(1)
print('\nRekomendasi SVD\n',res_svd,'\n')
print('Rekomendasi KNN\n',res_knn,'\n')   
"""