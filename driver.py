import json
from urlverification import URLVerification
from graphconstructor import KnowledgeGraph
import pickle
import datacleaning as dc
import os
import spacy
from entityextraction import EntityExtraction
import time

if __name__ == '__main__':
    LENGTH_OF_DB = 89754
    DICT_PICKLE_FILE = "knowgraphdictionary.pkl"
    URL_PICKLE_FILE = "knowgraphurl.pkl"
    VALID_URLS_PICKLE_FILE = "validurls.pkl"
    DEFUNCT_URLS_PICKLE_FILE = "defuncturls.pkl"
    FULL_SEEN_URLS_PICKLE_FILE = "fullseenurls.pkl"
    GRAPH_CONTENT_DF_PICKLE_FILE = "knowgraphdf.pkl"
    KNOWLEDGE_GRAPH_PICKLE_FILE = "fullknowgraph.pkl"
    SEGMENTATIONS_PICKLE_FILE = "segmentation.pkl"
    TEST_URLS_PICKLE_FILE = "testurls.pkl"
    column_labels = ['Entity', 'Relation', 'Value', 'Probability', 'Entity Literal Strings', 'Value Literal Strings',
                     'Best Entity Literal String', 'Best Value Literal String', 'Entity Categories', 'Value Categories']
    top_level_domains = [".com", ".edu", ".net", ".gov"]
    entities = dc.load_data_into_dict(LENGTH_OF_GRAPHDB, DICT_PICKLE_FILE_NAME)
    dc.to_pkl_file(DICT_PICKLE_FILE_NAME, entities)
    urls_for_samples = dc.parse_urls_in_entities_dict(URL_PICKLE_FILE_NAME, LENGTH_OF_GRAPHDB, entities)
    if not os.path.isfile(URL_PICKLE_FILE_NAME):
        dc.to_pkl_file(URL_PICKLE_FILE_NAME, urls_for_samples)

    knowledge_graph_df = dc.entities_to_df(GRAPH_CONTENT_DF_PICKLE_FILE_NAME, column_labels, entities)
    if not os.path.isfile(GRAPH_CONTENT_DF_PICKLE_FILE_NAME):
        dc.to_pkl_file(GRAPH_CONTENT_DF_PICKLE_FILE_NAME, knowledge_graph_df)
    nlp = spacy.load("en_core_web_lg")
    knowledge_graph = KnowledgeGraph(knowledge_graph_df, nlp)


