import streamlit as st
import os
import bentoml
from llama_parse import LlamaParse
from llama_index.node_parser import MarkdownElementNodeParser
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.embeddings import OpenAIEmbedding
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SentenceTransformerRerank

from llama_index.llms import OpenAI


API_KEY=os.environ["LLAMA_INDEX_API_KEY"]
os.environ["LLAMA_CLOUD_API_KEY"]=API_KEY

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


local_file_path = './local.pdf'
if ("file" not in st.session_state):
    uploaded_file = st.file_uploader("Choose a PDF file to upload")

    if (uploaded_file is not None):
        print("uploaded file: "+str(uploaded_file))
        with open(local_file_path, 'wb') as file:
            file.write(uploaded_file.read())

        documents = LlamaParse(result_type="markdown").load_data(local_file_path)

        node_parser = MarkdownElementNodeParser(llm=OpenAI(model="gpt-3.5-turbo-0613"), num_workers=8)

        nodes = node_parser.get_nodes_from_documents(documents)
        base_nodes, node_mapping = node_parser.get_base_nodes_and_mappings(nodes)

        ctx = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-4"),
            embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
            chunk_size=512
        )

        recursive_index = VectorStoreIndex(nodes=base_nodes, service_context=ctx)
        raw_index = VectorStoreIndex.from_documents(documents, service_context=ctx)

        retriever = RecursiveRetriever(
            "vector",
            retriever_dict={
                "vector": recursive_index.as_retriever(similarity_top_k=15)
            },
            node_dict=node_mapping,
        )

        reranker = SentenceTransformerRerank(top_n=5, model="BAAI/bge-reranker-large")

        global recursive_query_engine
        recursive_query_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[reranker], service_context=ctx)

        raw_query_engine = raw_index.as_query_engine(similarity_top_k=15, node_postprocessors=[reranker], service_context=ctx)

        st.session_state["file"] = local_file_path
        st.session_state["rqe"] = recursive_query_engine
        print("Defined recursive_query_engine and set sessionstate file")

query = st.text_input('Ask your question')
clicked = st.button('Ask')
if clicked:
    recursive_query_engine=st.session_state["rqe"]
    response_2 = recursive_query_engine.query(query)
    print("\n***********New LlamaParse+ Recursive Retriever Query Engine***********")
    print(response_2)
    response_2.response

    import requests

    # The URL to which the POST request is made
    url = 'https://xtts-yiut-322ce7e8.mt-guc1.bentoml.ai/synthesize'

    # The headers for the POST request
    headers = {
        'Content-Type': 'application/json',
    }

    # The JSON data to be sent with the POST request
    data = {
        "lang": "en",
        "text": response_2.response
    }

    # Make the POST request and receive the response
    response = requests.post(url, json=data, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the content of the response (binary data) to an MP3 file
        with open('Output.mp3', 'wb') as file:
            file.write(response.content)
        print("The audio was successfully saved to Output.mp3")
    else:
        print(f"Failed to retrieve audio. Status code: {response.status_code}")

    from elevenlabs import generate, play
    play(response.content)

    result=""
    query_string = "Summarize the following text in less than 5 words: " + str(response_2)

    from openai import OpenAI
    client = OpenAI()
    client.key = os.getenv("OPENAI_API_KEY")

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

    search_string=search_str
    url_friendly_string = make_url_friendly(search_string)
    print(url_friendly_string)

    import requests
    GOOGLE_YOUTUBE_API_KEY=os.environ["GOOGLE_YOUTUBE_API_KEY"]
    # The URL you want to send the GET request to
    url = 'https://youtube.googleapis.com/youtube/v3/search?part=snippet&maxResults=25&q='+url_friendly_string+'&key='+GOOGLE_YOUTUBE_API_KEY

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
        print(str(video["snippet"]["title"]));
        print(str(video["snippet"]["thumbnails"]["default"]["url"]))
        st.image(str(video["snippet"]["thumbnails"]["default"]["url"]), caption=str(video["snippet"]["title"]), width=200)
        "https://www.youtube.com/watch?v="+str(video["id"]["videoId"])
        str(video["snippet"]["title"]);
        str(video["snippet"]["thumbnails"]["default"]["url"])

