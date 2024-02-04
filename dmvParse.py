import os

API_KEY=os.environ["LLAMA_INDEX_API_KEY"]
os.environ["LLAMA_CLOUD_API_KEY"]=API_KEY

from llama_parse import LlamaParse

# documents = LlamaParse(result_type="text").load_data("./docs/examples/data/attention/attention.pdf")
# documents = LlamaParse(result_type="text").load_data("./docs/examples/data/entourage/entourage_2007.pdf")
# documents = LlamaParse(result_type="text").load_data("./docs/examples/data/pge/PGE.pdf")
# documents = LlamaParse(result_type="text").load_data("./docs/examples/data/DMV/dmv.pdf")
documents = LlamaParse(result_type="markdown").load_data("./docs/examples/data/DMV/dmv.pdf")

print(documents[0].text[6000:7000])

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


response_1 = raw_query_engine.query(query)
print("\n***********New LlamaParse+ Basic Query Engine***********")
print(response_1)

response_2 = recursive_query_engine.query(query)
print("\n***********New LlamaParse+ Recursive Retriever Query Engine***********")
print(response_2)


