

import os
import openai
import networkx as nx
import matplotlib.pyplot as plt
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    atts = ["Covid-19 death rate", "Country", "GDP", "HDI", "Location", "Population",
            "Human Development Index", "Language", "Covid-19 confirmed cases", "Covid-19 new cases",
            "Government type", "Population density"]

    G = getGraph(atts)
    getCausEffect("HDI", "Covid-19 death rate")

def getGraph(atts):
    G = nx.DiGraph()
    return G


def getCausEffect(a1,a2):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt="I am a highly intelligent question answering bot."
             " If you ask me a cause-effect related question, I will give you the answer. "
             ".\n\nQ: What is human life expectancy in the United States?\n"
             "A: Human life expectancy in the United States is 78 years.\n\n"
             "Q: Is temperature cause flight delay?\n"
             "A: There is no definitive answer to this question as weather conditions can vary greatly and can cause a number of different problems that could lead to a flight delay. However, if the temperature is extremely cold or hot,"
             " it could potentially cause issues with the plane's engines, which could lead to a delay.\n\n"
             "Q: Is high temperature cause crime?\n"
             "A: There is no definitive answer to this question as the causes of crime are complex and varied. However, some research has suggested that hot weather may be a contributing factor to crime rates, "
             "as it can lead to increased aggression and impulsive behavior.\n\n"
             "Q: Is temperature cause Covid-19 death?\n"
             "A: There is no evidence that temperature is a factor in Covid-19 death rates.\n\nQ: How does a telescope work?\n"
             "A: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\n"
             "Q: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\n"
             "\n\nQ: Is Human Development Index cause Covid-19 death rate?\nA:",
      temperature=0.7,
      max_tokens=1000,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=["\n"]
    )

    ans = response['choices'][0]['text']
    # print(str(ans))
    return ans

if __name__ == '__main__':
    main()
