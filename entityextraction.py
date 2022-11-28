from collections import Counter
from bs4 import BeautifulSoup
import requests
import re


class EntityExtraction:
    def __init__(self, nlp):
        self.nlp_model = nlp

    
    def get_text_from_paragraph(self, file):
        soup = BeautifulSoup(file, 'html.parser')
        tags = soup.findAll("p")
        text = []
        for i in tags:
            text += ''.join(i.findAll(text=True))
        return ''.join(text)

    def get_domain_terms(self, url):

        domain_terms = []
        headers = {
            "User-Agent": 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/51.0.2704.103 Safari/537.36',
            'Content-Type': 'text/html; charset=utf-8',
        }
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

    def extract_web_domain(self, top_level_domains, url):

        for d in top_level_domains:
            find_domain = url.partition(d)
            if len(find_domain[1]) > 0:
                website = find_domain[0] + find_domain[1]
                extension = find_domain[2]
                return website, extension
        return "", ""

    def collect_metadata(self, file):
        soup = BeautifulSoup(file, 'html.parser')
        meta_tag = soup.find("meta", {"name": re.compile(".*keyword*")})

        tags = []
        try:

            tags = re.split('[,;]', meta_tag["content"])
            if len(tags) == 1 and tags[0] == "null":
                tags = self.get_text_from_paragraph(file)
        except (KeyError, TypeError):
            tags = self.get_text_from_paragraph(file)
        return tags


