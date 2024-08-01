from typing import Optional, Union
import time

import faiss
import torch
import requests
from lxml import etree
from metapub import PubMedFetcher, PubMedArticle, FindIt
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling


def split_pubmed_xml(xml_string):
    root = etree.fromstring(xml_string)
    xmls = []
    for article in root.xpath("//PubmedArticle | //PubmedBookArticle"):
        new_root = etree.Element("PubmedArticleSet")
        new_root.append(article)
        xmls.append(etree.tostring(new_root, encoding='unicode'))
    return xmls


def query_articles(query: str, retmax: int=100):
    fetch = PubMedFetcher()
    pmids = fetch.pmids_for_query(query=query, retmax=retmax)
    fetch_xml = fetch.qs.efetch({'db': 'pubmed', 'id': ','.join(pmids), 'retmax': retmax})
    xmls = split_pubmed_xml(fetch_xml)
    articles = []
    for xml in xmls:
        article = PubMedArticle(xml)
        if article.abstract is None:
            article.abstract = ""
        articles.append(article)
    return articles


def download_pdf(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as pdf_file:
            pdf_file.write(response.content)
        print(f"PDF downloaded successfully and saved to {save_path}")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")


def download_pubmed_pdfs(articles, save_dir):
    for article in articles:
        fd = FindIt(article.pmid)
        if fd.url is not None:
            print("pmid: ", article.pmid, "url: ", fd.url)
            download_pdf(fd.url, f"{save_dir}/{article.pmid}.pdf")


class CustomizeSentenceTransformer(SentenceTransformer):
    def _load_auto_model(
            self,
            model_name_or_path: str,
            token: Optional[Union[bool, str]],
            cache_folder: Optional[str],
            revision: Optional[str] = None,
            trust_remote_code: bool = False,
    ):
        print("No sentence-transformers model found with name {}. Creating a new one with CLS pooling.".format(model_name_or_path))
        transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]


class Retriever:
    def __init__(self, keywords: str, retmax: int=100, download_pdf=False):
        # search for the articles through PubMed api
        start_time = time.time()
        print("Searching for articles...")
        self.articles = query_articles(query=keywords, retmax=retmax)
        print(f"{len(self.articles)} articles were found.")
        print("Time taken to search for articles: ", time.time() - start_time)

        # download the pdfs
        if download_pdf:
            download_pubmed_pdfs(self.articles, save_dir="pdfs")

        # construct the embedding for the articles
        start_time = time.time()
        print("Constructing the embedding...")
        self.embedding, self.h_dim = self.__construct_embedding()
        print("Embedding constructed.")
        print("Time taken to construct the embedding: ", time.time() - start_time)

        # construct the index for the embedding
        start_time = time.time()
        print("Constructing the index...")
        self.index = self.__construct_index()
        print("Index constructed.")
        print("Time taken to construct the index: ", time.time() - start_time)

        self.embedding_function = CustomizeSentenceTransformer("ncbi/MedCPT-Query-Encoder",
                                                               device="cuda" if torch.cuda.is_available() else "cpu")

    def get_relevant_documents(self, question, k=10):
        assert isinstance(question, str)
        question = [question]

        start_time = time.time()

        with torch.no_grad():
            query_embed = self.embedding_function.encode(question)
        res_ = self.index.search(query_embed, k=k)
        indices = res_[1][0].tolist()

        texts = self.__idx2txt(indices)
        scores = res_[0][0].tolist()

        print("Time taken to retrieve the documents: ", time.time() - start_time)
        return texts, scores

    def __construct_embedding(self):
        model = CustomizeSentenceTransformer("ncbi/MedCPT-Article-Encoder",
                                             device="cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        with torch.no_grad():
            texts = [[item.title, item.abstract] for item in self.articles]
            embed_articles = model.encode(texts)
            return embed_articles, model.encode([""]).shape[-1]

    def __construct_index(self):
        index = faiss.IndexFlatIP(self.h_dim)
        index.add(self.embedding)
        return index

    def __idx2txt(self, indices):  # return List of dict
        return [
            {
                "pmid": self.articles[idx].pmid,
                "title": self.articles[idx].title,
                "abstract": self.articles[idx].abstract
            }
            for idx in indices
        ]


if __name__ == "__main__":
    start_time = time.time()
    keywords = "Stomatology"
    retriever = Retriever(keywords=keywords, retmax=100, download_pdf=True)
    question = "What is the efficacy of the COVID-19 vaccine?"
    texts, scores = retriever.get_relevant_documents(question, k=20)
    # print(texts, scores)
    print("Total Time taken: ", time.time() - start_time)