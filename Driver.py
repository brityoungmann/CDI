import pandas as pd
import matplotlib.pyplot as plt
import Utils
import Clustering
import DAGBuilder
import networkx as nx
from sklearn.impute import SimpleImputer
import numpy as np
import CI
PATH = "data/"

def main():

    #flightsPipeline()
    covidPipeline()

def covidPipeline():
    # get full data joined with the extracted columns and the names of the original columns
    df, origin_cols = Utils.covid()
    #
    #
    # # # preprocessing of columns and pruning
    # # df = Utils.preprocess(df, origin_cols)
    # # df.to_csv(PATH+"covid_countries_pre.csv")
    # df = pd.read_csv(PATH+"covid_countries_pre.csv")
    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # df.replace(-1, np.nan)
    # print("before FD:", len(df.columns))
    # to_drop = Utils.removeFD(df, origin_cols)
    # df = df.drop(to_drop, axis=1)
    # print("after FD:",len(df.columns))
    # df.to_csv(PATH+"covid_countries_pre_fd.csv")

    df = pd.read_csv(PATH + "covid_countries_pre_fd.csv")
    # # clustering the extracted attributes


    cols = ["Country", "Deaths / 100 Cases"]
    # to_drop = [c for c in cols if "Unnamed" in c or "Record" in c or "title" in c or "year" in c]
    #clusters_info = Clustering.varclus_cluster(df, cols)
    #clusters_info.to_csv(PATH+"clusters_covid.csv")

    # clusters_info = pd.read_csv(PATH +"clusters_covid.csv")
    # # # # assigning a topic to each cluster
    # clusters_info = Utils.getClustersNames(clusters_info)
    # clusters_info.to_csv(PATH+"clusters_covid_topics_new.csv")

    # clusters_info = pd.read_csv(PATH+"clusters_covid_topics_new.csv")
    # # # # #
    # # # get the new dataset with the merged attributes and their new names
    # df_new = Utils.getData(clusters_info, df, cols)
    # df_new.to_csv(PATH + "covid_countries_final_v2.csv")
    df_new = pd.read_csv(PATH + "covid_countries_final_v2.csv")



    # import itertools
    # f= open(PATH+"GT_covid.csv","w")
    # nodes = list(df_new.columns)
    # for pair in itertools.permutations(nodes, r=2):
    #     f.write(pair[0]+", "+pair[1]+"\n")
    # f.close()
    # G = Utils.loadDAG(PATH + "GT_covid.csv")
    # print("num of nodes: ", len(G.nodes), "edges: ", len(G.edges))
    # print(nx.is_directed_acyclic_graph(G))
    # cycles = nx.simple_cycles(G)
    # for c in cycles:
    #     print(c)
    # G = DAGBuilder.buildDAGGPT3(df_new)
    # Utils.save_graph(G,"GPT3_Covid.csv")
    # G = Utils.loadDAG(PATH + "GPT3_covid.csv")
    # print("num of nodes: ", len(G.nodes), "edges: ", len(G.edges))
    # print(nx.is_directed_acyclic_graph(G))
    #
    #
    # G = Utils.loadDAG(PATH+"GPT3_Covid.csv")
    # G = DAGBuilder.pruneEdges(G,df_new)
    # print("num of nodes: ", len(G.nodes), "edges: ", len(G.edges))
    # print(nx.is_directed_acyclic_graph(G))
    # Utils.save_graph(G, "CATER_Covid.csv")
    # pos = nx.circular_layout(G)
    # nx.draw(G, pos=pos, with_labels=True, font_weight='bold')
    # plt.show()

    # GT = Utils.readFile(PATH + "GT_covid.csv")
    # df = Utils.readFile(PATH + "CATER_Covid.csv")
    # #df = Utils.readFile(PATH + "GPT3_Covid.csv")
    # #df = Utils.readFile(PATH + "covid2_fci_edgelist.csv")
    # #df = Utils.readFile(PATH + "covid2_pc_edgelist.csv")
    # #df = Utils.readFile(PATH + "covid2_lingam_edgelist.csv")
    # #df = Utils.readFile(PATH + "covid2_ges_edgelist.csv")
    # prec, recall = Utils.getStatistics(GT, df)

    # precision and recall - not existing edges

    # GT = Utils.loadDAG(PATH + "GT_covid.csv")
    # edgesGT = list(nx.non_edges(GT))
    # # G = Utils.loadDAG(PATH+"CATER_Covid.csv")
    # # edges = list(nx.non_edges(G))
    # # G = Utils.loadDAG(PATH + "GPT3_Covid.csv")
    # # edges = list(nx.non_edges(G))
    # # G = Utils.loadDAG(PATH + "covid2_fci_edgelist.csv")
    # # edges = list(nx.non_edges(G))
    # # G = Utils.loadDAG(PATH + "covid2_pc_edgelist.csv")
    # # edges = list(nx.non_edges(G))
    # # G = Utils.loadDAG(PATH + "covid2_lingam_edgelist.csv")
    # # for n in GT.nodes:
    # #     if not n in G.nodes:
    # #         G.add_node(n)
    # # edges = list(nx.non_edges(G))
    # G = Utils.loadDAG(PATH + "covid2_ges_edgelist.csv")
    # edges = list(nx.non_edges(G))
    # prec, recall = Utils.getStatistics(edgesGT, edges)

    #direct and total effect analysis
    G = Utils.loadDAG(PATH + "GPT3_Covid.csv")

    mediators = []
    for path in nx.all_simple_paths(G, source="Country", target="Deaths / 100 Covid-19 Cases", cutoff=3):
        #print(path)
        if len(path) == 3:
            mediators.append(path[1])
    print("mediators: ", mediators)
    # mediators = ['AIRLINE','Airline company statistics']
    CI.mediation(df_new, "Country", "Deaths / 100 Covid-19 Cases", mediators)


def flightsPipeline():
    # get full data joined with the extracted columns and the names of the original columns
    #df, origin_cols = Utils.flights()

    # df = pd.read_csv(PATH+"flights_50_cities_airlines.csv")
    # to_delete = ['YEAR','DAY','FLIGHT_NUMBER','TAIL_NUMBER','SCHEDULED_DEPARTURE','DEPARTURE_TIME',
    #             'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME',
    #             'AIR_TIME','WHEELS_ON','TAXI_IN','SCHEDULED_ARRIVAL'
    #                 ,'ARRIVAL_TIME','ARRIVAL_DELAY','DIVERTED','CANCELLED','CANCELLATION_REASON',
    #                'AIR_SYSTEM_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY','DESTINATION_AIRPORT']
    # df.drop(to_delete, axis=1, inplace=True)
    origin_cols = ['MONTH','DAY_OF_WEEK','AIRLINE',
                   'ORIGIN_AIRPORT',
                   'DEPARTURE_DELAY','DISTANCE','SECURITY_DELAY']
    # # preprocessing of columns and pruning
    # df = Utils.preprocess(df, origin_cols)
    # df.to_csv(PATH+"flights_50_cities_airlines_pre.csv")
    df = pd.read_csv(PATH+"flights_50_cities_airlines_pre.csv")
    df.replace(-1, np.nan)

    # print("before FD:", len(df.columns))
    # to_drop = Utils.removeFD(df, origin_cols, "flights")
    # df = df.drop(to_drop, axis=1)
    # print("after FD:",len(df.columns))
    # df.to_csv(PATH+"flights_50_cities_airlines_pre.csv")

    # # clustering the extracted attributes

    # df_sample = df.sample(frac=0.1)
    # cols = list(df_sample.columns)
    # to_drop = [c for c in cols if "Record" in c or "Unnamed" in c]
    # clusters_info = Clustering.varclus_cluster(df, ["DEPARTURE_DELAY", "ORIGIN_AIRPORT"])
    #
    # clusters_info.to_csv(PATH+"clusters_50_flights_new.csv")

    # clusters_info = pd.read_csv(PATH +"clusters_50_flights_new.csv")
    # print(clusters_info)
    # # # assigning a topic to each cluster
    # clusters_info = Utils.getClustersNames(clusters_info)
    # clusters_info.to_csv(PATH+"clusters_50_flights_topics.csv")


    # clusters_info = pd.read_csv(PATH+"clusters_50_flights_topics.csv")
    # # # #
    # # # get the new dataset with the merged attributes and their new names
    # df_new = Utils.getData(clusters_info, df, ['ORIGIN_AIRPORT','DEPARTURE_DELAY'])
    # df_new.to_csv(PATH + "flights_50_cities_airlines_final_v2.csv")
    df_new = pd.read_csv(PATH + "flights_50_cities_airlines_final_v2.csv")
    df_new = df_new.replace(-1, np.nan)

    # # print(G.nodes)
    # import itertools
    # f= open(PATH+"GT_flights.csv","w")
    # nodes = list(df_new.columns)
    # for pair in itertools.permutations(nodes, r=2):
    #     f.write(pair[0]+", "+pair[1]+"\n")
    # f.close()
    # G = Utils.loadDAG(PATH + "GT_flights.csv")
    # print("num of nodes: ", len(G.nodes), "edges: ", len(G.edges))
    # print(nx.is_directed_acyclic_graph(G))
    # cycles = nx.simple_cycles(G)
    # for c in cycles:
    #     print(c)
    # #

    # G = DAGBuilder.buildDAGGPT3(df_new)
    # Utils.save_graph(G, "GPT3_Flights.csv")
    # G = DAGBuilder.pruneEdges(G,df_new)
    # Utils.save_graph(G, "CATER_Flights.csv")

    # pos = nx.circular_layout(G)
    # nx.draw(G, pos=pos, with_labels=True, font_weight='bold')
    # plt.show()

    #precision and recall - existing edges
    # GT = Utils.readFile(PATH+"GT_flights.csv")
    # df = Utils.readFile(PATH+"CATER_Flights.csv")
    # #df = Utils.readFile(PATH + "GPT3_Flights.csv")
    # # #df = Utils.readFile(PATH + "flights2_fci_edgelist.csv")
    # # #df = Utils.readFile(PATH + "flights2_pc_edgelist.csv")
    # # #df = Utils.readFile(PATH + "flights2_lingam_edgelist.csv")
    # # df = Utils.readFile(PATH + "flights2_ges_edgelist.csv")
    # prec,recall = Utils.getStatistics(GT,df)

    # precision and recall - not existing edges

    # G = Utils.loadDAG(PATH+"GT_flights.csv")
    # edgesGT = list(nx.non_edges(G))
    # G = Utils.loadDAG(PATH+"CATER_Flights.csv")
    # edges = list(nx.non_edges(G))
    # # G = Utils.loadDAG(PATH + "GPT3_Flights.csv")
    # # edges = list(nx.non_edges(G))
    # # G = Utils.loadDAG(PATH + "flights2_fci_edgelist.csv")
    # # edges = list(nx.non_edges(G))
    # # G = Utils.loadDAG(PATH + "flights2_pc_edgelist.csv")
    # # edges = list(nx.non_edges(G))
    # # G = Utils.loadDAG(PATH + "flights2_lingam_edgelist.csv")
    # # edges = list(nx.non_edges(G))
    # # G = Utils.loadDAG(PATH + "flights2_ges_edgelist.csv")
    # # edges = list(nx.non_edges(G))
    # # # df = Utils.readFile(PATH + "graph_ges_edgelist.csv")
    # prec,recall = Utils.getStatistics(edgesGT,edges)

    #direct and total effect analysis
    G = Utils.loadDAG(PATH + "CATER_Flights.csv")
    # parents = G.in_edges("ORIGIN_AIRPORT")
    # parents = [i[0] for i in parents]
    # print("in edges: ", parents)
    mediators = []
    for path in nx.all_simple_paths(G, source="ORIGIN_AIRPORT", target="DEPARTURE_DELAY"):
        if len(path) == 3:
            mediators.append(path[1])
    print("mediators: ", mediators)
    # mediators = ['AIRLINE','Airline company statistics']
    CI.mediation(df_new, "ORIGIN_AIRPORT", "DEPARTURE_DELAY", mediators)



if __name__ == '__main__':
    main()