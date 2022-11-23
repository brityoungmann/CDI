import nltk
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
# Lemmatize the documents.
from nltk.stem.wordnet import WordNetLemmatizer
# nltk.download()
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import re
import transformers
from transformers import pipeline
import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

PATH = "data/"

NUM = 3



def getClustersTopicsBERT(clusters_info):
    docs = clusters_info["Variables"].tolist()
    docs = [i.replace(";", ", ") for i in docs]
    docs = [re.findall('[A-Z][^A-Z]*', i) for i in docs]
    docs = [" ".join(i) for i in docs]
    docs = [i.replace(" F,", "Fahrenheit,") for i in docs]
    docs = [i.replace(" C,", "Celsius,") for i in docs]
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    # labels = ["economy", "location", "land measurements", "weather", "politics", "population measurements",
    #           "temperature", "Precipitation", "Monetary Rankings", "Income", "geography", "area measurements",
    #           "Demography", "Population", "Company's financial", "Business efficiency", "Location"]
    labels = getCasueNETtopics()
    hypothesis_template = 'Topic: {}.'
    names = []
    for d in docs:
        prediction = classifier(d, labels, hypothesis_template=hypothesis_template, multi_class=True)
        print(prediction['labels'][0])
        names.append(prediction['labels'][0])
    clusters_info["Topics"] = names
    return clusters_info

def getClustersTopicsGPT3(clusters_info):
    docs = clusters_info["Variables"].tolist()
    docs = [i.replace(";", ", ") for i in docs]
    docs = [re.findall('[A-Z][^A-Z]*', i) for i in docs]
    docs = [" ".join(i) for i in docs]
    docs = [i.replace(" F,", "Fahrenheit,") for i in docs]
    docs = [i.replace(" C,", "Celsius,") for i in docs]

    names = []
    for d in docs:
        prediction = getTopicGPT3(d)
        names.append(prediction)
    clusters_info["Topics"] = names
    return clusters_info


def getTopicGPT3(d):

    prompt = "What is the topic of "+d+"? A:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )

    ans = response['choices'][0]['text']
    ans = ans.replace(d,"")
    ans = ans.replace("The topic of the ", "")
    ans = ans.replace("The topic of ","")
    # ans = ans.replace("is the", "")
    # ans = ans.replace("is ","")
    # ans = ans.replace(".","")
    ans = ans.strip()
    print(ans)
    return ans

def getCasueNETtopics():
    labels = []
    with open(PATH+"conceptsCauseNET.txt", "r",encoding='utf-8') as f:
        for line in f:
            labels.append(line.strip())
    return labels


def getClustersTopics(clusters_info):
    docs = clusters_info["Variables"].tolist()
    docs = [i.replace(";", ", ") for i in docs]
    docs = [re.findall('[A-Z][^A-Z]*', i) for i in docs]
    docs = [" ".join(i) for i in docs]
    docs = [i.replace(" F,", "Fahrenheit,") for i in docs]
    docs = [i.replace(" C,", "Celsius,") for i in docs]
    model, corpus, topics = generateLDAmodel(docs)
    # print(topics)
    topics_dic = getTopicDic(topics)
    # print(topics_dic)
    names = []
    for i in range(len(corpus)):
        topic = model.get_document_topics(corpus[i])
        n = topics_dic[topic[0][0]]
        names.append(n)


def getTopicDic(topics):
    dic = {}
    for i in range(len(topics)):
        k = topics[i][0]
        v = topics[i][1]
        v = v.split("+")
        v = [vv.split("*") for vv in v]
        #v = [[float(i) if i.isnumeric() else i for i in inner_list] for inner_list in v]
        for lst in v:
            lst[0] = float(lst[0])
        v = sorted(v, key=lambda x: x[0], reverse=True)[0:NUM]
        words = ""
        for j in range(len(v)):
            words = words + v[j][1] +";"
        words = words.replace('"','')
        dic[k] = words
    return dic

def generateLDAmodel(docs):
    # docs = open("atts_names.txt","r").readlines()
    # print(docs)
    # # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    # Compute bigrams.

    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=2)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Remove rare and common tokens.
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=1, no_above=0.9)
    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    # print('Number of unique tokens: %d' % len(dictionary))
    # print('Number of documents: %d' % len(corpus))

    # Set training parameters.
    num_topics = len(docs)
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # Make an index to word dictionary.
    # temp = dictionary[0]  # This is only to "load" the dictionary.
    #id2word = dictionary.id2token
    id2word = dictionary.token2id
    id2word = {v: k for k, v in id2word.items()}

    # Train LDA model.
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )


    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    # avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    # print('Average topic coherence: %.4f.' % avg_topic_coherence)


    # model.print_topics()
    topics = model.show_topics()
    # for t in topics:
    #     print(t)

    # print(model.get_document_topics(corpus[3]))
    return model, corpus, topics