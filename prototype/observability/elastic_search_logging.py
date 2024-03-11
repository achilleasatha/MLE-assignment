import subprocess  # nosec
from time import sleep

from elasticsearch import Elasticsearch

from config import AppConfigSettings


def elastic_search_setup(config: AppConfigSettings):
    subprocess.Popen(
        [config.elasticsearch.exec_path],  # nosec
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    sleep(10)
    elastic_client = Elasticsearch(hosts=[config.elasticsearch.host])
    return elastic_client
