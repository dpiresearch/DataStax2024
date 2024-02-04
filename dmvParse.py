import os
import bentoml
from openai import OpenAI

client = OpenAI()
client.key = os.getenv("OPENAI_API_KEY")

API_KEY=os.environ["LLAMA_INDEX_API_KEY"]
os.environ["LLAMA_CLOUD_API_KEY"]=API_KEY

from llama_parse import LlamaParse

# documents = LlamaParse(result_type="text").load_data("./docs/examples/data/attention/attention.pdf")
# documents = LlamaParse(result_type="text").load_data("./docs/examples/data/entourage/entourage_2007.pdf")
# documents = LlamaParse(result_type="text").load_data("./docs/examples/data/pge/PGE.pdf")
# documents = LlamaParse(result_type="text").load_data("./docs/examples/data/DMV/dmv.pdf")
documents = LlamaParse(result_type="markdown").load_data("./docs/examples/data/DMV/dmv.pdf")

# print(documents[0].text[6000:7000])

from llama_index.node_parser import MarkdownElementNodeParser
from llama_index.llms import OpenAI

node_parser = MarkdownElementNodeParser(llm=OpenAI(model="gpt-3.5-turbo-0613"), num_workers=8)

nodes = node_parser.get_nodes_from_documents(documents)
base_nodes, node_mapping = node_parser.get_base_nodes_and_mappings(nodes)

from llama_index import VectorStoreIndex, ServiceContext
from llama_index.embeddings import OpenAIEmbedding

ctx = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4"), 
    embed_model=OpenAIEmbedding(model="text-embedding-3-small"), 
    chunk_size=512
)

recursive_index = VectorStoreIndex(nodes=base_nodes, service_context=ctx)
raw_index = VectorStoreIndex.from_documents(documents, service_context=ctx)

from llama_index.retrievers import RecursiveRetriever

retriever = RecursiveRetriever(
    "vector", 
    retriever_dict={
        "vector": recursive_index.as_retriever(similarity_top_k=15)
    },
    node_dict=node_mapping,
)

from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(top_n=5, model="BAAI/bge-reranker-large")

recursive_query_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[reranker], service_context=ctx)

raw_query_engine = raw_index.as_query_engine(similarity_top_k=15, node_postprocessors=[reranker], service_context=ctx)

# query = "What do you do a right turn at an intersection?"
query = "What happens when you get a DUI?"

# response_0 = baseline_pdf_query_engine.query(query)
# print("***********Baseline PDF Query Engine***********")
# print(response_0)


# response_1 = raw_query_engine.query(query)
# print("\n***********New LlamaParse+ Basic Query Engine***********")
# print(response_1)

response_2 = recursive_query_engine.query(query)
print("\n***********New LlamaParse+ Recursive Retriever Query Engine***********")
print(response_2)

result=""
query_string = "Summarize the following text in less than 5 words: " + str(response_2)
# with bentoml.SyncHTTPClient("https://mistralai-mistral-7-b-instruct-v-0-2-service-barr-322ce7e8.mt-guc1.bentoml.ai") as client:
#    result = client.generate(
#        promtp=query_string,
#    )

# response_2_1 = recursive_query_engine.query(query_string)
# print("\n***********New LlamaParse+ Second Recursive Retriever Query Engine***********")
# print(response_2)

search_str=""
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": query_string},
                    ],
                }
            ],
            max_tokens=300,
        )
search_str = response.choices[0].message.content
print("Summary small: " + str(search_str))
'''
print("\n***********New Bento result***********")
print(result)

with bentoml.SyncHTTPClient("https://vllm-crw7-322ce7e8.mt-guc1.bentoml.ai") as client:
    response_generator = client.generate(
        prompt="Explain superconductors like I'm five years old",
        tokens=None
    )
    for response in response_generator:
        print(response)

'''

import re
from urllib.parse import quote

def make_url_friendly(s):
    # Convert to lowercase
    s = s.lower()
    # Replace spaces and underscores with hyphens
    s = re.sub(r'[\s_]+', '-', s)
    # Remove all characters that are not alphanumeric or hyphens
    s = re.sub(r'[^\w-]', '', s)
    # URL encode any remaining characters that are not URL safe
    s = quote(s)
    return s

search_string='California DMV ' + search_str
url_friendly_string = make_url_friendly(search_string)
print(url_friendly_string)

import requests

# The URL you want to send the GET request to
url = 'https://youtube.googleapis.com/youtube/v3/search?part=snippet&maxResults=25&q='+url_friendly_string+'&key=AIzaSyDnRyiGdn8t7ZCvcIqJOYcfFcQFcJAx0Gs'

# Headers including the Authorization with the API key and the content type
headers = {
    'Content-Type': 'application/json',
}

# Making the GET request
response = requests.get(url, headers=headers)

# Assuming the response's content is JSON, parse it into a Python dictionary
data = response.json()

# Print the data to see the response
# print(data)

video_id=data["items"][0]["id"]["videoId"]

num_videos = data["items"]

for video in num_videos:
    print("https://www.youtube.com/watch?v="+str(video["id"]["videoId"]))

