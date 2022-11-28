import csv
import time
import re
import pickle
from collections import defaultdict
import string
import urllib.parse
import pandas as pd
import os.path


def remove_newline(string1):
    return re.sub('\x0a', '', string1)


def remove_tab(string1):
    return re.sub('\x09', '', string1)


def transform_entities_to_match_graph_concept(string1):
    string1 = string1.lower()
    string1 = string1.translate(str.maketrans({c: '' for c in string.punctuation+'’'}))
    string1 = re.sub(' ', '_', string1)
    return re.sub(' ', '', string1)


def parse_url(string1):
    url = ''.join(string1 .partition("http")[1:])
    url = urllib.parse.unquote(url).split('+')
    url = list(map(remove_tab, url))
    url = [x for x in url if x.startswith("http")]
    return url


def load_data_in_datab(graph_db_length, dict_pickle_file_name):
    start = time.time()
    entities = defaultdict()
    if os.path.isfile(dict_pickle_file_name):
        with open(dict_pickle_file_name, 'rb') as f:
            entities = pickle.load(f)
        if len(entities.keys()) != graph_db_length:
            print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
        print(f".pkl file loaded in {time.time() - start} seconds")
        return entities
    else:
        with open("knowledgegraphdata.csv", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",")
            next(csv_reader) 
            for i, row in enumerate(csv_reader):
                record = row[0].split("\t")
                try:
                    del record[3]
                    del record[4]
                except IndexError:
                    pass
                try:
                    if len(record) == 0 or float(record[3]) <= 0.95:
                        continue
                except IndexError:
                    pass
                entities[i] = record
        print(f"Finished loading into dict in {time.time() - start} seconds, length is {len(entities.keys())}")
        return entities


def parse_url_in_entities_dict(url_pkl_file_name, graph_db_length, entities=None):
    start = time.time()
    if os.path.isfile(url_pkl_file_name):
        with open(url_pkl_file_name, 'rb') as f:
            entities_1 = pickle.load(f)
        if len(entities_1.keys()) != graph_db_length:
            print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
        print(f".pkl file loaded in {time.time() - start} seconds")
        return entities_1
    entities_1 = dict(map(lambda x: (x[0], parse_url(x[1][-1])), entities.items()))
    print(f"Finished cleaning URLs in {time.time() - start} seconds, length is {len(entities_1.keys())}")
    return entities_1


def to_pkl_file(dict_file_name, entities):
    start = time.time()
    with open(dict_file_name, "wb") as f:
        pickle.dump(entities, f)
    print(f"Saved object to {dict_file_name} in {time.time() - start} seconds")
    f.close()


def validate_observation_structure(value):
    try:
        entity_concept = value[-3].split()
        if not entity_concept[0].startswith("concept"):
            return False
    except IndexError:
        return False
    return True


def delete_blank_entries_in_observation(value):
    try:
        value.remove('')
        del value[-1]
    except ValueError:
        pass
    return value


def entities_to_df(graph_df_pickle_file_name, column_labels, entities=None):
    start = time.time()
    if os.path.isfile(graph_df_pickle_file_name):
        with open(graph_df_pickle_file_name, 'rb') as f:
            knowledge_graph_df = pickle.load(f)
        print(f".pkl file loaded in {time.time() - start} seconds")
        return knowledge_graph_df
    else:
        knowledge_graph_df = pd.DataFrame.from_dict(entities, orient='index')
        knowledge_graph_df = knowledge_graph_df.drop([10], axis=1)
        knowledge_graph_df.columns = column_labels
        knowledge_graph_df = knowledge_graph_df.reset_index()
        knowledge_graph_df["Entity Literal Strings"] = knowledge_graph_df["Entity Literal Strings"].apply(
            lambda x: re.sub(r'"', '\x09', x) if x is not None else None)
        knowledge_graph_df["Value Literal Strings"] = knowledge_graph_df["Value Literal Strings"].apply(
            lambda x: re.sub(r'"', '\x09', x) if x is not None else None)
        knowledge_graph_df["Entity"] = knowledge_graph_df["Entity"].apply(
            lambda x: re.sub(r'(_)\1+', '_', str(x)) if x is not None else None)
        knowledge_graph_df["Value"] = knowledge_graph_df["Value"].apply(
            lambda x: re.sub(r'(_)\1+', '_', str(x)) if x is not None else None)
    return knowledge_graph_df


