import pandas as pd

import spacy


nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

pd.set_option('display.max_colwidth', 200)


def get_entities(sent):
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  
    prv_tok_text = ""  

    prefix = ""
    modifier = ""


    for tok in nlp(sent):
        if tok.dep_ != "punct":
            if tok.dep_ == "compound":
                prefix = tok.text
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text

    return [ent1.strip(), ent2.strip()]


def get_relation(sent):
    doc = nlp(sent)

    matcher = Matcher(nlp.vocab)

    pattern = [{'DEP': 'ROOT'},
               {'DEP': 'prep', 'OP': "?"},
               {'DEP': 'agent', 'OP': "?"},
               {'POS': 'ADJ', 'OP': "?"}]

    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]]

    return span.text


candidate_sentences = pd.read_csv("/Users/jai/Desktop/test-data.csv")
candidate_sentences.shape
entity_pairs = []
relations = []

for i in tqdm(candidate_sentences["Search Data"]):
    entity_pairs.append(get_entities(i))
    for i in range(len(target_unique)-1):
        for j in range(i+1, len(target_unique)):
            word1 = nlp(target_unique[i])
            word2 = nlp(target_unique[j])
        if word1.similarity(word2) > 0.70:
            print(target_unique[i], "similar to ", target_unique[j], " because of similarity ", word1.similarity(word2))
            source.append(target_unique[i])
            target.append(target_unique[j])
            relations.append("similar to")

for i in tqdm(candidate_sentences["Search Data"]):
    relations.append(get_relation(i))
    for x in target:
        if x not in target_unique:
            target_unique.append(x)
            for j in range(i+1, len(target_unique)):
                for i in range(len(target_unique)-1):


                    target_unique = []

for x in target:
    if x not in target_unique:
        target_unique.append(x)
        for j in range(i+1, len(target_unique)):
            for i in range(len(target_unique)-1):
            ##for j in range(i+1, len(target_unique)):
                word1 = nlp(target_unique[i])
                word2 = nlp(target_unique[j])
                if word1.similarity(word2) > 0.70:
                    print(target_unique[i], "similar to ", target_unique[j], " because of similarity ", word1.similarity(word2))
                    source.append(target_unique[i])
                    target.append(target_unique[j])
                    relations.append("similar to")
                word1 = nlp(target_unique[i])
                word2 = nlp(target_unique[j])
                if word1.similarity(word2) > 0.70:
                    print(target_unique[i], "similar to ", target_unique[j], " because of similarity ", word1.similarity(word2))
                    source.append(target_unique[i])
                    target.append(target_unique[j])
                    relations.append("similar to")

for i in range(len(target_unique)-1):
    for j in range(i+1, len(target_unique)):
        word1 = nlp(target_unique[i])
        word2 = nlp(target_unique[j])
        if word1.similarity(word2) > 0.70:
            print(target_unique[i], "similar to ", target_unique[j], " because of similarity ", word1.similarity(word2))
            source.append(target_unique[i])
            target.append(target_unique[j])
            relations.append("similar to")

kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})

G = nx.from_pandas_edgelist(kg_df, "source", "target",
                            edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12, 12))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
plt.show()




