import requests
import os
import pickle


class URLVerification:
    def __init__(self, top_level_domains):
        self.full_seen_urls = {}
        self.valid_urls = {}
        self.defunct_urls = {}
        self.top_level_domains = top_level_domains

    def extract_web_domain_from_url(self, url):
        website, extension = self.extract_web_domain_from_url(url)
        if website in self.valid_urls:
            return True
        if website in self.defunct_urls:
            return False
        try:
            r = requests.get(website, timeout=1)
            if website in self.valid_urls:
                self.valid_urls[website].append(extension)
            else:
                self.valid_urls[website] = [extension]
        except requests.exceptions.RequestException as e:
            if website not in self.defunct_urls:
                self.defunct_urls[website] = True
            return False
        return True
        
    def load_valid_urls_from_pkl_file(self, valid_urls_filepath):
        if os.path.isfile(valid_urls_filepath):
            with open(valid_urls_filepath, 'rb') as f:
                self.valid_urls = pickle.load(f)

    def load_defunct_urls_from_pkl_file(self, defunct_urls_filepath):
        if os.path.isfile(defunct_urls_filepath):
            with open(defunct_urls_filepath, 'rb') as f:
                self.defunct_urls = pickle.load(f)

    def url_is_valid(self, url):

        for d in self.top_level_domains:
            find_domain = url.partition(d)
            if len(find_domain[1]) > 0:
                website = find_domain[0] + find_domain[1]
                extension = find_domain[2]
                return website, extension
        return "broken", "broken"
