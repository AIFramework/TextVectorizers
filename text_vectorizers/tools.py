import nltk
from nltk.data import find as nltk_find


def nltk_resource_download(resource_name):
  '''Загружает ресурсы nltk, если они не загружены.'''
    try:
        nltk_find('tokenizers/' + resource_name)
    except LookupError:
        nltk.download(resource_name)