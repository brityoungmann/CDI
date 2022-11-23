from sklearn.decomposition import PCA# , SparsePCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Union
from varclushi import VarClusHi
import numpy as np
import pandas as pd
THRESHOLD = 3
PATH1 = "data/"

def cluster(df_orig, cols):
    print(len(df_orig.columns))
    df = df_orig[df_orig.columns.difference(cols)]
    print("num of columns to cluster: ",len(df.columns))

    demo1_vc = VarClusHi(df, maxeigval2=1, maxclus=None)
    demo1_vc.varclus()

    print("num of clusters: ", len(demo1_vc.info))
    # print(demo1_vc.info)
    # print("######################################")
    clusters = demo1_vc.rsquare
    groups = clusters.groupby("Cluster")
    d = {}
    Cluster = []
    center = []
    namesc = []
    for n, g in groups:
        Cluster.append(n)
        index = g['RS_Ratio'].idxmax()
        rep = clusters.iloc[[index]]["Variable"]
        center.append(rep.iloc[0])
        names = g["Variable"].tolist()
        namesc.append(';'.join(names))

    d["Cluster"] = Cluster
    d["center"] = center
    d["Variables"] = namesc
    df = pd.DataFrame(data=d)
    info = demo1_vc.info.copy(deep=True)

    l = info['Cluster'].to_list()
    l = [int(i) for i in l]
    info['Cluster'] = l

    full = info.merge(df, on="Cluster", how='left')
    full = full[full["Eigval1"] > THRESHOLD]
    print("num of meaningful clusters: ", len(full))
    #full.to_csv(PATH1+"clusters_flights_cities.csv")
    return full

def varclus_cluster(df_orig, cols):
    print(len(df_orig.columns))
    df = df_orig[df_orig.columns.difference(cols)]
    print("num of columns to cluster: ",len(df.columns))

    print('varclus clustering')
    imputed_data = impute_table(df)

    # https://medium.com/analytics-vidhya/variable-clustering-from-scratch-using-sas-and-python-4e21c7505cab
    new_table = pd.DataFrame(imputed_data, columns=df.columns)
    print("finish imputing missing values...")
    output = VarClusHi(new_table)
    output.varclus()
    print("finish generating clusters...")
    # output.info
    # output.rsquare

    print("num of clusters: ", len(output.info))
    clusters = output.info.sort_values(by='Eigval1', ascending=False)
    cluster_count = pick_top_cluster_count(clusters['Eigval1'])
    print("num of meaningful clusters: ", cluster_count)
    picked_clusters = clusters[:cluster_count].copy(deep=True)
    # print(picked_clusters)
    # print(picked_clusters.T)

    for cluster_number, cluster_info_row in picked_clusters.iterrows():
        cluster_variables_table = output.rsquare[output.rsquare['Cluster'] == cluster_number]

        center_variable_index = cluster_variables_table['RS_Ratio'].idxmax()
        center_variable = cluster_variables_table.loc[center_variable_index]
        center_variable_name = center_variable['Variable']

        variable_list = list(cluster_variables_table['Variable'])
        variables_string = ';'.join(variable_list)

        picked_clusters.loc[cluster_number, 'center'] = center_variable_name
        picked_clusters.loc[cluster_number, 'Variables'] = variables_string
    # picked_clusters['Variables'] = [';'.join(output.rsquare[output.rsquare['Cluster'] == i]['Variable'].tolist()) for i, row in picked_clusters.iterrows()]

    return picked_clusters.sort_index()

def pca_cluster(df_orig, cols):
    print(len(df_orig.columns))
    # TODO: anna is unclear why filtering out these columns needs to be done, and/or why this needs to be in the clustering algorithm
    df = df_orig[df_orig.columns.difference(cols)]
    print("num of columns to cluster: ",len(df.columns))

    print('pca-based clustering')
    imputed_data = impute_table(df)
    scaled_data = scale_values(imputed_data)

    pca = PCA() # pca = SparsePCA()
    remapped_data = pca.fit_transform(scaled_data)

    print("num of clusters: ", len(pca.explained_variance_))
    cluster_count = pick_top_cluster_count(pca.explained_variance_)
    print("num of meaningful clusters: ", cluster_count)
    picked_clusters = pd.DataFrame(pca.components_[:cluster_count], index=pca.explained_variance_[:cluster_count], columns=df.columns)
    # print(picked_clusters)
    # print(picked_clusters.T)

    output = pd.DataFrame()
    output['Eigval1'] = picked_clusters.index
    output['Cluster'] = output.index
    # determine for each component, which attributes we're using to summarize the component
    output['center'] = [row.idxmax() for i, row in picked_clusters.iterrows()]
    output['Variables'] = [";".join(pick_top_variables(row)) for i, row in picked_clusters.iterrows()]
    return output, pca.components_

# TODO: migrate the following functions into Utils.py
MIN_EXPLAINED_VARIANCE_THRESHOLD = 0.85
def impute_table(table: pd.DataFrame, use_kmeans: bool=False) -> np.array:
    ## imputation of nan values ##
    if not use_kmeans:
        # basic imputation
        imputed_data = SimpleImputer().fit_transform(table) # missing_values=np.nan, strategy='mean' by default or 'median' or 'most_frequent'
    else:
        # k-means clustering-based imputation
        # https://stackoverflow.com/questions/35611465/python-scikit-learn-clustering-with-missing-data#35613047
        input_array = table.to_numpy()
        # print(input_array.shape, input_array)
        def kmeans_missing(X, n_clusters=10, max_iter=10):
            """Perform K-Means clustering on data with missing values.
            Args:
              X: An [n_samples, n_features] array of data to cluster.
              n_clusters: Number of clusters to form.
              max_iter: Maximum number of EM iterations to perform.
            Returns:
              labels: An [n_samples] vector of integer labels.
              centroids: An [n_clusters, n_features] array of cluster centroids.
              X_hat: Copy of X with the missing values filled in.
            """

            # Initialize missing values to their column means
            missing = ~np.isfinite(X)
            mu = np.nanmean(X, 0, keepdims=1)
            X_hat = np.where(missing, mu, X)

            for i in range(max_iter):
                if i > 0:
                    # initialize KMeans with the previous set of centroids. this is much
                    # faster and makes it easier to check convergence (since labels
                    # won't be permuted on every iteration), but might be more prone to
                    # getting stuck in local minima.
                    kmean = KMeans(n_clusters, init=prev_centroids)
                else:
                    # do multiple random initializations in parallel
                    kmean = KMeans(n_clusters)

                # perform clustering on the filled-in data
                labels = kmean.fit_predict(X_hat)
                centroids = kmean.cluster_centers_

                # fill in the missing values based on their cluster centroids
                X_hat[missing] = centroids[labels][missing]

                # when the labels have stopped changing then we have converged
                if i > 0 and np.all(labels == prev_labels):
                    break

                prev_labels = labels
                prev_centroids = kmean.cluster_centers_

            return labels, centroids, X_hat
        labels, centroids, imputed_data = kmeans_missing(input_array)
        # print(labels.shape, labels)
        # print(centroids.shape, centroids)

    # print(imputed_data.shape, imputed_data)

    return imputed_data
def scale_values(data: np.array) -> np.array:
    ## scale values ##
    scaled_data = StandardScaler().fit_transform(data)
    # print(scaled_data.shape, scaled_data)
    return scaled_data
def pick_top_cluster_count(variance_list: Union[pd.Series,list]) -> int:
    # determine which components we're going to be keeping
    total_explained_variance = sum(abs(variance_list))
    # component_count = 0
    for i, c in enumerate(variance_list):
        if sum(variance_list[0:i+1]) >= (MIN_EXPLAINED_VARIANCE_THRESHOLD * total_explained_variance):
            # component_count = i+1
            return i+1
def pick_top_variables(variable_significance_list: pd.Series) -> list:
    variable_variances = variable_significance_list**2
    total_explained_variance = sum(variable_variances)
    sorted_values = variable_variances.sort_values(ascending=False)
    ret = []
    for i, c in enumerate(sorted_values):
        if sum(sorted_values[0:i+1]) >= (MIN_EXPLAINED_VARIANCE_THRESHOLD * total_explained_variance):
            desired_variables = list(sorted_values[0:i+1].index)
            not_desired_variables = list(variable_significance_list[variable_significance_list < 0].index)
            variable_list = [a for a in desired_variables if a not in not_desired_variables]
            return variable_list