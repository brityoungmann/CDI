# import torch as t
# import cdt
# import pandas as pd
# import networkx as nx
# from networkx import Graph
#
# from cdt.independence.graph import FSGNN
# from sklearn.datasets import load_boston
# from cdt.independence.stats import NormMI
# from cdt.independence.stats.model import IndependenceModel
# import matplotlib.pyplot as plt
#
#
#
# def predict_undirected_graph(data, predictor):
#     graph = Graph()
#     for idx_i, i in enumerate(data.columns):
#         for idx_j, j in enumerate(data.columns[idx_i+1:]):
#             score = predictor.predict(data[i].values, data[j].values)
#             if abs(score) > 0.001:
#                 graph.add_edge(i, j, weight=score)
#
#     return graph
#
#
# boston = load_boston()
# # print(pd.DataFrame(boston['data']).head())
# df_features = pd.DataFrame(boston['data'])
# df_target = pd.DataFrame(boston['target'])
# # print(df_target)
# # obj = FSGNN()
# # # output = obj.predict_features(df_features, df_target)
# # ugraph = obj.predict(df_features)
#
#
# obj1 = NormMI()
# g = predict_undirected_graph(df_features, obj1)
# nx.draw(g,with_labels = True)
# plt.show()
#
import causallearn

from causallearn.search.ConstraintBased.FCI import fci
G = fci(data, independence_test_method="chisq")