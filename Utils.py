import networkx as nx
import pandas as pd
from scipy.stats import spearmanr
import info_theo as info
import Clustering
import TopicGenerator
import pandas_profiling
PATH = "data/"
import random
THRESHOLD = 0.9

def covid():
    df1 = pd.read_csv(PATH + "country_wise_latest.csv")
    df2 = pd.read_csv(PATH + "countries_columns.csv")

    # original columns
    cols = list(df1.columns)
    df = df2.merge(df1, on='Country', how='left')

    print(len(df), len(df1))
    print("original columns: ", len(df1.columns))
    print("with country columns: ", len(df.columns))

    print("num of rows: ", len(df))

    df.to_csv(PATH + "covid_countries.csv")
    return df, cols


def flights():
    df1 = pd.read_csv(PATH+"airportCityState.csv")
    df2 = pd.read_csv(PATH + "flights.csv")

    df3 = pd.read_csv(PATH + "cities_columns.csv")
    df4 = pd.read_csv(PATH + "airlines_columns.csv")


    #original columns
    cols = list(df2.columns)
    # df2 = df2.sample(frac=0.2)

    df = df2.merge(df1, on='ORIGIN_AIRPORT', how='left')
    cities = set(df["City"].tolist())
    print(len(cities))
    cities = random.sample(list(cities), 50)
    print(len(df))
    df = df[df['City'].isin(cities)]
    print(len(df))



    df = df.merge(df4, on='AIRLINE', how='left')
    print("with airline columns: ", len(df.columns))
    df = df.merge(df3, on='City', how='left')
    print("with city columns: ", len(df.columns))

    # df = df.groupby("City").mean()
    # cols = [c for c in cols if c in df.columns]


    print("num or original columns: ", len(df2.columns))
    print("num or columns: ", len(df.columns))
    print("num of rows: ", len(df))

    df.to_csv(PATH+"flights_50_cities_airlines.csv")
    return df, cols

def preprocess(df,cols):
    to_drop = []
    nrows = len(df)
    verbose = False

    print("num of attributes before: ", len(df.columns))

    df = df.infer_objects()
    for col in df.columns:
        if "Unnamed" in col:
            to_drop.append(col)
            continue
        df[col] = df[col].fillna(-1)
        missing = df[col].tolist().count(-1)
        #missing values
        if missing > THRESHOLD*nrows:
            if not col in cols:
                to_drop.append(col)
                continue


        dataTypeObj = df.dtypes[col]
        if dataTypeObj == "object":
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes


        valsCol = set(df[col].tolist())
        #constant value
        if len(valsCol) < 2:
            if not col in cols:
                to_drop.append(col)
                continue

    print("finish simple filtering...")
    black = get_high_entropy_atts(df)
    for e in black:
        if not e in cols:
            to_drop.append(e)

    # print("to drop: ", to_drop)
    for col in to_drop:
        if not col in cols:
            if col in df:
                del df[col]

    print("num of attributes after: ", len(df.columns))
    return df

def get_high_entropy_atts(data, start=0.3, end=0.4, steps=0.01, cut=0.0001, alpha=0.01, debug=False):
    samplesizez = []
    dic = dict()
    black = []
    white = []
    basesample = data.sample(frac=0.3, replace=False)
    features=data.columns.values
    size = len(data.index)
    while start <= end:
        start = start + steps
        sample = basesample.sample(frac=start, replace=False)
        samplesizez.insert(0, (len(sample.index)))
        inf=info.Info(sample)
        for col in features:
            if col in dic:
                list = dic[col]
                list.insert(0, inf.entropy(col, size))
                dic[col] = list
            else:
                dic[col] = [inf.entropy(col, size)]

    for col in features:
        if not any(dic[col]):
            continue
        rho, pval1 = spearmanr(samplesizez, dic[col])

        if pval1 <= alpha:
            black.insert(0, col)
        else:
            white.insert(0, col)
    # self.features=np.array(white)
    return black

def getClustersNames(clusters_info):
    #clusters_info = TopicGenerator.getClustersTopicsBERT(clusters_info)
    clusters_info = TopicGenerator.getClustersTopicsGPT3(clusters_info)
    return clusters_info

def save_graph(G,name):
    f= open(PATH+name,"w")
    for u,v in G.edges():
        f.write(u+","+v+"\n")
    f.close()

def removeFD(df, cols,name):
    all_cols = df.columns
    # print("Top Absolute Correlations")
    # get_top_abs_correlations(df, THRESHOLD,name)
    df = pd.read_csv(PATH+name+"_corr.csv")
    keys = df['metricFirst'].tolist()
    fds = df['singleLine']
    G = nx.Graph()
    for index, row in df.iterrows():
        a = row['metricFirst']
        b = row['singleLine']
        G.add_edge(a,b)
    cc = nx.connected_components(G)
    to_drop = []
    for c in cc:
        c = list(c)
        rep = None
        for att in c:
            if att in cols:
                rep = att
                break
        if rep == None:
            rep = random.choice(c)

        for att in c:
            if not att == rep:
                to_drop.append(att)

    to_drop = [i for i in to_drop if i in all_cols]
    print(to_drop)
    print("dropping FDs: ", len(to_drop))
    return to_drop



def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, t,name):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    #print(au_corr.tolist())
    df = au_corr.to_frame()
    df = df.rename(columns={0: 'corr'})
    print(len(df))
    df = df[df["corr"] > t]
    print(len(df))
    print(len(df.columns))
    df.to_csv(PATH+name+"_corr.csv")
    return




def loadDAG(file):
    edges = readFile(file)
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    return G


def readFile(file):
    f = open(file,"r")
    ans = []
    nodes = set()
    for line in f:

        u = line.split(",")[0].strip()
        v = line.split(",")[1].strip()
        nodes.add(u)
        nodes.add(v)
        ans.append((u,v))
    f.close()
    print("num of edges: ", len(ans), "nodes: ", len(nodes))
    return ans




def getStatistics(gt,baseline):
    intersection = set(
        [
            tuple(sorted(elem))
            for elem in gt
        ]
    ) & set(
        [
            tuple(sorted(elem))
            for elem in baseline
        ]
    )
    missed = list(set(gt) - set(baseline))
    print(missed)
    print(len(gt), len(baseline), len(intersection))
    precision = float(len(intersection))/len(gt)
    recall = float(len(intersection)) / len(baseline)

    # precision = 0.79
    # recall = 0.59
    F1 = (2 * precision * recall) / (precision + recall)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", F1)
    return precision,recall


def getData(clusters_info, df, cols):
    atts = clusters_info["center"].tolist()
    names = clusters_info["Topics"].tolist()
    dic = {}
    for i in range(len(clusters_info)):
        dic[atts[i]] = names[i]
    cols_all = cols + atts
    df_new = df[cols_all]
    df_new = df_new.rename(columns=dic, errors="raise")
    return df_new

'''
check if: If (𝑂 ⊥⊥ 𝑅𝐸 = 1|𝐸) and (𝑂 ⊥⊥ 𝑅𝐸 = 1|𝐸,𝑇 )
'''
# def checkSelectionBias(x,y,e):
#     R = []
#     for i in e:
#         if i == -1:
#             R.append(0)
#         else:
#             R.append(1)
#     Y = [int(i) for i in y]
#     E = [int(i) for i in e]
#     val = drv.information_mutual_conditional(Y,R,E)
#     if val < EPSILON:
#         X = [int(i) for i in x]
#         ET = getRandomVar(E,X)
#         val = drv.information_mutual_conditional(Y,R,ET)
#         if val < EPSILON:
#             return False
#     return True

def main():
    df,cols = flights()
    df = preprocess(df,cols)
    df.to_csv(PATH+"flights_cities_pro.csv")
    clusters_info = Clustering.cluster(df,cols)
    clusters_info = getClustersNames(clusters_info)
    df_new = getData(clusters_info, df, cols)
    df_new.to_csv(PATH+"flights_cities_final.csv")



if __name__ == '__main__':
    main()