import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score


from sklearn.preprocessing import StandardScaler


# Standardisation 
# p122 of Python Machine learning: machine learning and deep learning with python. second edition
# link to cardiff library:
# https://whel-primo.hosted.exlibrisgroup.com/primo-explore/fulldisplay?docid=44CAR_ALMA51106757330002420&context=L&vid=44WHELF_CAR_VU1&lang=en_US&search_scope=CSCOP_EVERYTHING&adaptor=Local%20Search%20Engine&tab=searchall@cardiff&query=any,contains,python%20machine%20learning&sortby=rank&offset=0


def standardisation(df, columns):
    cluster_dataset=df.loc[:, columns].copy()
    scaler = StandardScaler()
    X = cluster_dataset.fillna(0).values.copy()
    X = scaler.fit_transform(X)
    columns = cluster_dataset.columns
    index = cluster_dataset.index
    return pd.DataFrame(index=index, columns=columns, data=X)

def cluster(values, max_cluster):
    algorithm = "auto"
    silhouette_scores = []
    calinski_harabaz_scores = []
    min_silhouette_avg = 0.
    cluster_centers=[]
    cluster_labels=[]
    for n_clusters in range(2, max_cluster, 1):
        print(n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=10, algorithm=algorithm, n_init=30).fit(values)
        labels_temp= kmeans.labels_
        silhouette_avg = metrics.silhouette_score(values, labels_temp)
        
        silhouette_scores.append(silhouette_avg)
        
        calinski_harabaz_score= metrics.calinski_harabaz_score(values, labels_temp)
        calinski_harabaz_scores.append(calinski_harabaz_score)
        
        nb_values_over = 0
        print("cluster:", n_clusters , "silhouette score:" ,silhouette_avg, "calinski_harabaz:",calinski_harabaz_score )
        if min_silhouette_avg<silhouette_avg:
            min_silhouette_avg=silhouette_avg
            cluster_centers= kmeans.cluster_centers_
            cluster_labels= labels_temp
        
        silhouette_samples_values = metrics.silhouette_samples(values, labels_temp)
        for cluster in range(n_clusters):
            cluster_values = silhouette_samples_values[cluster_labels==cluster]
            nb_values_over = nb_values_over+ len(cluster_values[np.where(cluster_values>silhouette_avg)])
        print("Number of values over average:", nb_values_over, "({:04.1f}%)".format(nb_values_over/len(values)*100)) 
        
    return cluster_centers.shape[0], cluster_centers, cluster_labels, silhouette_scores, calinski_harabaz_scores