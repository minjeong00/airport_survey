import pandas as pd
from kmodes.kmodes import KModes
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df_ori = pd.read_excel("C:/Users/82104/Desktop/fproject/V10 Survey Passengers raw data_2101_edit0804_Kihun Kim.xlsx",
                       header=1)

df1 = df_ori[['Q9_1', 'Q9_2', 'Q9_3', 'Q9_4', 'Q9_5']]
# , 'Q10_1', 'Q10_2', 'Q4_1.1', 'Q4_2.1', 'Q2_1.3', 'Q2_2.2',
#               'Q3_1.2', 'Q3_2.2', 'Q3_3', 'Q3_4', 'Q3_1.4', 'Q3_1.5', 'Q2_1.7'
df1 = df1.replace(
        {'Not at all Confident': 1, 'Not Very Confident': 2, 'Neutral': 3, 'Somewhat Confident': 4, 'Very Confident': 5})
df1 = df1.replace(
        {'Not at all Concerned': 1, 'Not at all concerned':1, 'Fairly Unconcerned': 2, 'Neutral': 3,
         'Somewhat Concerned': 4, 'Somewhat concerned':4, 'Very Concerned': 5})
df1 = df1.replace(
        {'Much too little': 1, 'Somewhat too little': 2, 'Appropriate': 3, 'Somewhat too much': 4, 'Far too much': 5})

df1 = df1.dropna(axis=0)
df1 = df1.drop([0])

km = KModes(n_clusters=3, init='Huang', n_init=500, verbose=1)
clusters = km.fit_predict(df1)

my_pca = PCA(n_components=2)
transformed_comps = my_pca.fit_transform(df1)
df_comps = pd.DataFrame(data=transformed_comps, columns=['PC1', 'PC2'])

# km = KModes(n_clusters=3, init='Huang', n_init=5, verbose=1)
# clusters = km.fit_predict(df_comps)

# kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10)
# clusters = kmeans.fit_predict(df_comps)

df_comps = df_comps.join(pd.Series(clusters, name='cluster_label'))

markers = ['^', 's', 'o']
for i, marker in enumerate(markers):
    x_data = df_comps[df_comps['cluster_label'] == i]['PC1']
    y_data = df_comps[df_comps['cluster_label'] == i]['PC2']
    plt.scatter(x_data, y_data, marker=marker)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters')
plt.show()