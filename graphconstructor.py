import numpy as np
from collections import Counter
import datacleaning as dc
import pandas as pd
from segmentation import segment
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import pickle
from entityextraction import EntityExtraction
import os
from googlesearch import search
pd.options.mode.chained_assignment = None  

DEBUG = False  


class KnowledgeGraph:
    def __init__(self, knowledge_graph_df, nlp):
        self.pkl_file_name = "segmentedgraphtriples.pkl"
        self.sim_matrix_pkl_file_name = "simmatrix.pkl"
        self.url_indices_mappings_pkl_file_name = "urlindices.pkl"
        self.url_domain_terms_pkl_file_name = "urldomainterms.pkl"
        self.url_concepts_pkl_file_name = "urlconceptsterms.pkl"
        self.knowledge_graph_df = knowledge_graph_df
        self.nlp = nlp
        self.entity_extraction = EntityExtraction(nlp)
        self.segmentation_mappings = {}
        self.segmentation_mappings_inverted = {}
        if DEBUG:
            if not os.path.isfile(self.url_indices_mappings_pkl_file_name):
                self.indices_for_urls = {}
            else:
                with open(self.url_indices_mappings_pkl_file_name, 'rb') as f:
                    self.indices_for_urls = pickle.load(f)
            if not os.path.isfile(self.sim_matrix_pkl_file_name):
                self.sim_matrix = None
            else:
                with open(self.sim_matrix_pkl_file_name, 'rb') as f:
                    self.sim_matrix = pickle.load(f)
            if not os.path.isfile(self.url_domain_terms_pkl_file_name):
                self.url_domain_terms = {}
            else:
                with open(self.url_domain_terms_pkl_file_name, 'rb') as f:
                    self.url_domain_terms = pickle.load(f)
            if not os.path.isfile(self.url_concepts_pkl_file_name):
                self.url_concepts = {}
            else:
                with open(self.url_concepts_pkl_file_name, 'rb') as f:
                    self.url_concepts = pickle.load(f)
        else:
            self.indices_for_urls = {}
            self.sim_matrix = None
            self.url_domain_terms = {}
            self.url_concepts = {}
        self.c = 0.6
        self.edit_distances = {} 
        if not os.path.isfile(self.pkl_file_name):
            self.triples = self.generate_triples()
            dc.to_pkl_file(self.pkl_file_name, self.triples)
        if os.path.isfile(url_pkl_file_name):
                with open(url_pkl_file_name, 'rb') as f:
                    entities_1 = pickle.load(f)
                if len(entities_1.keys()) != graph_db_length:
                    print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
                print(f".pkl file loaded in {time.time() - start} seconds")
        else:
            with open(self.pkl_file_name, 'rb') as f:
                self.triples = pickle.load(f)
        self.entity_types = self.triples["Entity"].unique()
        self.relation_types = self.triples["Relation"].unique()
        self.concept_types = self.triples["Segmented Concept"].unique()

    def generate_triples(self):
        start = time.time()
    entities = defaultdict()
    if os.path.isfile(dict_pickle_file_name):
        with open(dict_pickle_file_name, 'rb') as f:
            entities = pickle.load(f)
        if len(entities.keys()) != graph_db_length:
            print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
        print(f".pkl file loaded in {time.time() - start} seconds")
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
        triples_df["Segmented Concept"] = triples_df["Concept"].apply(self.segment_concept_names)


    def segment_concept_names(self, concept):
        domain_terms = []
        try:
            r = requests.get(url, headers=headers, timeout=1)
        except Exception:
            print("Request failed")
            return domain_terms
        metadata_tags = self.collect_metadata(file)
        if isinstance(metadata_tags, str):
            try:
                description = self.nlp_model(metadata_tags)
            except ValueError:
                description = self.nlp_model(metadata_tags[:1000000])

            useful_entity_types = ['EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'ORG', 'PERSON', 'PRODUCT',
                                   'WORK_OF_ART']
            domain_terms = [ent.text for ent in description.ents if ent.label_ in useful_entity_types]
        else:
            domain_terms = metadata_tags
        return [x for x, y in Counter(domain_terms).most_common(10)]

    def get_relation_types(self):
        relation_types = self.knowledge_graph_df["Relation"].unique()
        relation_types = list(map(lambda x: x.partition("concept:"), relation_types))
        return dict.fromkeys(list(map(lambda x: x[2] if x[0] == '' else x[0], relation_types)))

    def get_similarity_of_two_urls(self, url1, url2):
        if url1 in self.url_domain_terms:
            domain_terms = self.url_domain_terms[url1]
        if os.path.isfile(url_pkl_file_name):
                with open(url_pkl_file_name, 'rb') as f:
                    entities_1 = pickle.load(f)
                if len(entities_1.keys()) != graph_db_length:
                    print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
                print(f".pkl file loaded in {time.time() - start} seconds")
        else:
            domain_terms = self.entity_extraction.get_domain_terms_from_url(url1)
            self.url_domain_terms.update({url1: domain_terms})
        if url2 in self.url_domain_terms:
            domain_terms_1 = self.url_domain_terms[url2]
        else:
            domain_terms_1 = self.entity_extraction.get_domain_terms_from_url(url2)
            self.url_domain_terms.update({url2: domain_terms_1})
        print("Similarity between:")
        print(url1)
        print(url2)
        if url1 in self.url_concepts:
            concept_list = self.url_concepts[url1]
        else:
            concept_list = []
            for term in domain_terms:
                concept = self.determine_concept_of_unknown_term(term)
                if concept != "unknown_concept":
                    concept_list.append(concept)
                if DEBUG:
                    print(f"Most likely concept for {term}: " + concept)
                if os.path.isfile(url_pkl_file_name):
                    with open(url_pkl_file_name, 'rb') as f:
                        entities_1 = pickle.load(f)
                if len(entities_1.keys()) != graph_db_length:
                    print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
                print(f".pkl file loaded in {time.time() - start} seconds")
            self.url_concepts.update({url1: concept_list})
        if url2 in self.url_concepts:
            concept_list_1 = self.url_concepts[url2]
        else:
            concept_list_1 = []
            for term in domain_terms_1:
                concept = self.determine_concept_of_unknown_term(term)
                if concept != "unknown_concept":
                    concept_list_1.append(concept)
                if DEBUG:
                    print(f"Most likely concept for {term}: " + concept)
            self.url_concepts.update({url2: concept_list_1})
        domain_term_matching_score = self.direct_domain_term_matches(domain_terms, domain_terms_1)[0][0]
        concept_matching_score = self.direct_domain_term_matches(concept_list, concept_list_1)[0][0]
        if DEBUG:
            print("Domain term matching", domain_term_matching_score)
            print("Concept matching score", concept_matching_score)
        print("Total similarity", self.harmonic_mean(domain_term_matching_score, concept_matching_score))
        return self.harmonic_mean(domain_term_matching_score, concept_matching_score)

    def pagerank(self, k):
        i = 1
        v = np.full(self.sim_matrix.shape[0], 1/self.sim_matrix.shape[0])
        u = v
        while i < 50:
            u_new = ((1-self.c) * np.dot(self.sim_matrix, u)) + (self.c * v)
            u = u_new
            i += 1
        top_urls = np.argsort(-u)
        if DEBUG:
            print(top_urls)
        best_domain_terms = []
        for index in top_urls[:k]:
            url = self.indices_for_urls[index]
            best_domain_terms += self.url_domain_terms[url]
        if DEBUG:
            dc.to_pkl_file(self.url_concepts_pkl_file_name, self.url_concepts)
            dc.to_pkl_file(self.url_domain_terms_pkl_file_name, self.url_domain_terms)
        if os.path.isfile(url_pkl_file_name):
                with open(url_pkl_file_name, 'rb') as f:
                    entities_1 = pickle.load(f)
                if len(entities_1.keys()) != graph_db_length:
                    print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
                print(f".pkl file loaded in {time.time() - start} seconds")
        print([x for x, y in Counter(best_domain_terms).most_common(15)])
        return [x for x, y in Counter(best_domain_terms).most_common(15)]


    def get_recommendations(self, k):
        best_domain_terms = self.pagerank(k)
        recs = []
        for i in range(0, len(best_domain_terms), 3):
            if DEBUG:
                print("Query: ", ' '.join(best_domain_terms[i:i + 3]))
            if os.path.isfile(url_pkl_file_name):
                with open(url_pkl_file_name, 'rb') as f:
                    entities_1 = pickle.load(f)
                if len(entities_1.keys()) != graph_db_length:
                    print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
                print(f".pkl file loaded in {time.time() - start} seconds")
            for j in search(' '.join(best_domain_terms[i:i + 3]), tld="co.in", num=2, stop=2, pause=1):
                recs.append(j)
        for url in [x for x, y in Counter(recs).most_common() if x not in self.url_concepts][:10]:
            print(url)
        return [x for x, y in Counter(recs).most_common() if x not in self.url_concepts][:10]

    def construct_similarity_matrix(self, url_list):
        if(sim_matrix):
            sim_matrix = self.sim_matrix
            if DEBUG:
                print(sim_matrix.shape)
            starting_index = len(self.indices_for_urls)
            url_list = [u for u in url_list if u not in self.url_domain_terms]
            for i in range (starting_index, starting_index + len(url_list)):
                self.indices_for_urls.update({i: url_list[i-starting_index]})
            zero_row = np.zeros((len(url_list), self.sim_matrix.shape[0]))
            sim_matrix = np.concatenate((sim_matrix, zero_row), axis=0)
            zero_col = np.zeros((len(url_list), sim_matrix.shape[0]))
            sim_matrix = np.concatenate((sim_matrix, zero_col.T), axis=1)
            if DEBUG:
                print(sim_matrix.shape)
            if os.path.isfile(url_pkl_file_name):
                with open(url_pkl_file_name, 'rb') as f:
                    entities_1 = pickle.load(f)
                if len(entities_1.keys()) != graph_db_length:
                    print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
                print(f".pkl file loaded in {time.time() - start} seconds")
            for i in range(sim_matrix.shape[1]):
                for j in range(sim_matrix.shape[0] - len(url_list), sim_matrix.shape[0]):
                    if DEBUG:
                        print(i, j)
                    url1, url2 = url_list[j-sim_matrix.shape[0]], self.indices_for_urls[i]
                    if url1 == url2 or i == j:
                        sim_matrix[i][j] = 1
                        continue
                    sim_matrix[i][j] = self.get_similarity_of_two_urls(url1, url2)
            for i in range(sim_matrix.shape[0]):
                for j in range(sim_matrix.shape[1]):
                    sim_matrix[j][i] = sim_matrix[i][j]
            self.sim_matrix = sim_matrix
            if DEBUG:
                print(sim_matrix)
                self.save_sim_matrix_to_pkl_file()
                self.save_indices_to_pkl_file()
            return sim_matrix
        else:
            print("Sim matrix exists, adding to the one that's already there")
            sim_matrix = self.sim_matrix
            if DEBUG:
                print(sim_matrix.shape)
            starting_index = len(self.indices_for_urls)
            url_list = [u for u in url_list if u not in self.url_domain_terms]
            for i in range (starting_index, starting_index + len(url_list)):
                self.indices_for_urls.update({i: url_list[i-starting_index]})
            zero_row = np.zeros((len(url_list), self.sim_matrix.shape[0]))
            sim_matrix = np.concatenate((sim_matrix, zero_row), axis=0)
            zero_col = np.zeros((len(url_list), sim_matrix.shape[0]))
            sim_matrix = np.concatenate((sim_matrix, zero_col.T), axis=1)
            if DEBUG:
                print(sim_matrix.shape)
            if os.path.isfile(url_pkl_file_name):
                with open(url_pkl_file_name, 'rb') as f:
                    entities_1 = pickle.load(f)
                if len(entities_1.keys()) != graph_db_length:
                    print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
                print(f".pkl file loaded in {time.time() - start} seconds")
            for i in range(sim_matrix.shape[1]):
                for j in range(sim_matrix.shape[0] - len(url_list), sim_matrix.shape[0]):
                    if DEBUG:
                        print(i, j)
                    url1, url2 = url_list[j-sim_matrix.shape[0]], self.indices_for_urls[i]
                    if url1 == url2 or i == j:
                        sim_matrix[i][j] = 1
                        continue
                    sim_matrix[i][j] = self.get_similarity_of_two_urls(url1, url2)
            for i in range(sim_matrix.shape[0]):
                for j in range(sim_matrix.shape[1]):
                    sim_matrix[j][i] = sim_matrix[i][j]
            self.sim_matrix = sim_matrix
            if DEBUG:
                print(sim_matrix)
                self.save_sim_matrix_to_pkl_file()
                self.save_indices_to_pkl_file()
            return sim_matrix

    def harmonic_mean(self, a, b):
        return (a+b)/2

    def direct_domain_term_matches(self, domain_terms1, domain_terms2):
        if os.path.isfile(url_pkl_file_name):
                with open(url_pkl_file_name, 'rb') as f:
                    entities_1 = pickle.load(f)
                if len(entities_1.keys()) != graph_db_length:
                    print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
                print(f".pkl file loaded in {time.time() - start} seconds")
        domain_terms1 = list(map(dc.transform_entities_to_match_graph_concept_format, domain_terms1))
        domain_terms2 = list(map(dc.transform_entities_to_match_graph_concept_format, domain_terms2))
        domain_terms1, domain_terms2 = self.remove_duplicates(domain_terms1, domain_terms2)
        full_set_of_domain_terms = list(set(domain_terms1 + domain_terms2))
        domain_terms1 = dict.fromkeys(domain_terms1)
        domain_terms2 = dict.fromkeys(domain_terms2)

        vector1 = list(map(lambda x: 1 if x in domain_terms1 else 0, full_set_of_domain_terms))
        vector2 = list(map(lambda x: 1 if x in domain_terms2 else 0, full_set_of_domain_terms))

        return cosine_similarity(np.array([vector1]), np.array([vector2]))

    def remove_duplicates(self, terms1, terms2):
        if os.path.isfile(url_pkl_file_name):
                with open(url_pkl_file_name, 'rb') as f:
                    entities_1 = pickle.load(f)
                if len(entities_1.keys()) != graph_db_length:
                    print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
                print(f".pkl file loaded in {time.time() - start} seconds")
        ps = PorterStemmer()
        terms1 = list(map(lambda x: ps.stem(x), terms1))
        terms2 = list(map(lambda x: ps.stem(x), terms2))
        for i in range(len(terms1)):
            for j in range(len(terms2)):
                t, t1 = terms1[i], terms2[j]
                if t == t1:
                    continue
                elif t in t1:
                    terms1[i] = terms2[j]
                elif t1 in t:
                    terms2[j] = terms1[i]
        return terms1, terms2


    def save_triples_df_to_pkl_file(self):
        dc.to_pkl_file(self.pkl_file_name, self.triples)

    def save_sim_matrix_to_pkl_file(self):
        dc.to_pkl_file(self.sim_matrix_pkl_file_name, self.sim_matrix)

    def save_indices_to_pkl_file(self):
        dc.to_pkl_file(self.url_indices_mappings_pkl_file_name, self.indices_for_urls)

    def edit_distance(self, string1, string2):
        if os.path.isfile(url_pkl_file_name):
                with open(url_pkl_file_name, 'rb') as f:
                    entities_1 = pickle.load(f)
                if len(entities_1.keys()) != graph_db_length:
                    print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
                print(f".pkl file loaded in {time.time() - start} seconds")
        if string1 in self.edit_distances:
            return self.edit_distances[string1]
        memo = np.zeros((len(string1) + 1, len(string2) + 1))
        for i in range(len(memo)):
            memo[i][0] = i
        for i in range(len(memo[0])):
            memo[0][i] = i
        for i in range(1, len(memo)):
            for j in range(1, len(memo[0])):
                if string1[i - 1] == string2[j - 1]:
                    memo[i][j] = memo[i - 1][j - 1]
                else:
                    memo[i][j] = 1 + max(memo[i - 1][j], memo[i][j - 1], memo[i - 1][j - 1])
        self.edit_distances[string1] = memo[-1][-1]
        return memo[-1][-1]
  
    def determine_concept_of_unknown_term(self, term):
  
        term_modified = dc.transform_entities_to_match_graph_concept_format(term)
        candidates = self.triples.loc[(self.triples["Relation"] == "generalizations") &
                                      (self.triples["Entity"].str.contains(term_modified))]
        if os.path.isfile(url_pkl_file_name):
                with open(url_pkl_file_name, 'rb') as f:
                    entities_1 = pickle.load(f)
                if len(entities_1.keys()) != graph_db_length:
                    print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
                print(f".pkl file loaded in {time.time() - start} seconds")
        if candidates.shape[0] != 0:
            entities_to_check = candidates.loc[:, 'Entity']
            candidates.loc[:, "NELL Match Sim"] = entities_to_check.apply(self.edit_distance, string2=term_modified)
 
            if candidates.loc[candidates["NELL Match Sim"].idxmin()]["NELL Match Sim"] == 0:

                return candidates.loc[candidates["NELL Match Sim"].idxmin()]["Segmented Concept"]
            candidates["Matches"] = candidates["Entity Literal Strings"].apply(
                lambda x: True if term_modified in x else False)

            if candidates["Matches"].any():
                direct_matches = candidates.loc[candidates["Matches"]]["Segmented Concept"].reset_index(drop=True)
                return direct_matches.loc[0]


        def get_semantic_similarity(concept, term):
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

        concept_similarities = pd.DataFrame(self.concept_types, columns=["Concept"])
        concept_similarities["Scores"] = pd.Series(self.concept_types).apply(get_semantic_similarity, term=term)
        concept_similarities = concept_similarities.loc[concept_similarities["Scores"] >= 0]
        if (concept_similarities["Scores"] == 0).all():

            if candidates.shape[0] != 0:
                if DEBUG:
                    print("Closest match", candidates.loc[candidates["NELL Match Sim"].idxmin()]["Entity"])
                return candidates.loc[candidates["NELL Match Sim"].idxmin()]["Segmented Concept"]
            if DEBUG:
                print("Unknown concept", term)
            return "unknown_concept"
        concept_similarities = concept_similarities.sort_values(by=["Scores"], ascending=False)
        self.edit_distances.clear()
        best_concept = concept_similarities.loc[concept_similarities["Scores"].idxmax()]["Concept"]
        best_concept_unsegmented = self.triples.loc[self.triples["Segmented Concept"] ==
                                                    best_concept]["Concept"].iloc[0]
        new_row = pd.DataFrame([[term_modified, "generalizations", best_concept_unsegmented, best_concept, 1, "", ""]],
                               columns=["Entity", "Relation", "Concept", "Segmented Concept", "Concept Counts",
                                        "Value Literal Strings", "Entity Literal Strings"])
        self.triples = self.triples.append(new_row, ignore_index=True)
        return best_concept


