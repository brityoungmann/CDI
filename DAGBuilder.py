import info_theo as info
import networkx as nx
import os
import openai
import matplotlib.pyplot as plt
import itertools
openai.api_key = os.getenv("OPENAI_API_KEY")
THRESHOLD = 0.00005
PATH = "data/"

def flightsDAG():
    G = nx.DiGraph()
    f = open(PATH+"FlightsDAG.txt","r")
    for line in f:
        l = line.split(";")
        c = l[0].strip()
        e = l[1].strip()
        G.add_edge(c,e)
    f.close()
    return G



def buildDAGGPT3(df, verbose = False):
    G = nx.DiGraph()
    for c in df.columns:
        G.add_node(c)
    # print(G.nodes)
    nodes = list(G.nodes)
    for pair in itertools.permutations(nodes, r =2):
        if "Unnamed" in pair[0] or "Unnamed" in pair[1]:
            continue
        ans_parsed,ans = getCausEffect(pair[0], pair[1], verbose)
        if ans_parsed == "YES":
            G.add_edge(pair[0], pair[1])
            print(pair[0],";", pair[1])
            # print(ans)
            # print("*****************************")
    return G

'''
For each pair of variables (A, B) having an edge between them,
 and for each variable (or subset) C with an edge connected to either of them, 
 eliminate the edge between A and B if A⫫B|C
'''
def pruneEdges(G, df):
    inf = info.Info(df)
    #a list of edges to be removed
    to_remove = []
    for u, v in G.edges():
        init_cmi = inf.CMI(u,v)
        if init_cmi < THRESHOLD:
            continue
        potential_u = G.in_edges(u)
        potential_u = [i[0] for i in potential_u]
        potential_v = G.in_edges(v)
        potential_v = [i[0] for i in potential_v]
        #a set of nodes connected to either u or v
        potential = list(set(potential_u + potential_v))
        for p in itertools.chain.from_iterable(itertools.combinations(potential, r) for r in range(1,3)):
            if u in p or v in p:
                continue
            ad_cmi = inf.CMI(u, v, list(p))
            print(init_cmi, ad_cmi)
            if ad_cmi < init_cmi:
                #the CMI decreases and it approximatly 0
                if ad_cmi < THRESHOLD and abs(ad_cmi) < THRESHOLD:
                    to_remove.append((u,v))
                    print("remove: ",u,v, init_cmi, ad_cmi)
                    break


    print("num of edges before pruning: ", len(G.edges))
    for pair in to_remove:
        G.remove_edge(pair[0], pair[1])
    print("num of edges after pruning: ", len(G.edges))
    return G


def getCausEffect(c,e, verbose = False):
    prompt = "I am a highly intelligent question answering bot." \
             "If you ask me a cause-effect related question, I will give you the answer." \
             "Even if there is weak evidence for a cause-effect relation, I will answer Yes. " \
             "Otherwise, I will answer No. "\
             "Q: Is temperature cause flight delay?\n"\
             "A: There is no definitive answer to this question as weather conditions can vary greatly and" \
             " can cause a number of different problems that could lead to a flight delay. " \
             "However, if the temperature is extremely cold or hot,"\
             " it could potentially cause issues with the plane's engines, which could lead to a delay.\n\n"\
             "Q: Is high temperature cause crime?\n"\
             "A: There is no definitive answer to this question as the causes of crime are" \
             " complex and varied. However, some research has suggested that hot weather may be a " \
             "contributing factor to crime rates, "\
             "as it can lead to increased aggression and impulsive behavior.\n\n"\
             "Q: Is temperature cause Covid-19 death?\n"\
             "A: There is no evidence that temperature is a factor in Covid-19 death rates." \
             "\n\nQ: How does a telescope work?\n"\
             "A: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\n"\
             "Q: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\n"\
             "\n\nQ: Is Human Development Index cause Covid-19 death rate?\n"\
             "A: There is no scientific evidence to suggest that the Human Development Index " \
             "is a cause of Covid-19 death rates. However, some factors that are associated with" \
             " a country's HDI, such as poverty and poor health care,"\
             " may contribute to higher death rates from the virus."

    template = "Is " +c +" a potential cause of " +e+ "? A:"
    prompt = prompt +template
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt= prompt,
      temperature=0,
      max_tokens=1000,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=["\n"]
    )

    ans = response['choices'][0]['text']

    ans_parsed = parseAns(ans)
    #print(ans_parsed,ans)
    if verbose:
        print(c,e,str(ans))
        print("*******************************")
    return ans_parsed, ans

def parseAns(ans):
    if len(ans)<2:
        return "NO"
    if "However, some factors " in ans:
        return "YES"
    if "There is no definitive answer" in ans:
        return "YES"
    if "There is no scientific evidence" in ans:
        return "NO"
    if "No, " in ans:
        return "NO"
    if "There is no evidence" in ans:
        return "No"
    if "Yes, " in ans:
        return "YES"
    if ans.startswith("No "):
        return "NO "
    if "However, " in ans:
        return "YES"
    return "YES"


if __name__ == '__main__':
    print("hi")