import os

from dataclasses_json.stringcase import spinalcase
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"]="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"
os.environ["HF_TOKEN"] = "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"

from dataclasses import FrozenInstanceError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

'step 1 A indexing Document loading '

video_id = "bMZBLfDriHI"

try :
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id, languages=['en'])
    'transcript = " ".join(item["text"] for item in transcript_list)'
    transcript = " ".join(item.text for item in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")

'step 1 B indexing TEXT SPLITTING'


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.create_documents([transcript])
#chunks = splitter.split_text(transcript)
print(len(chunks))

'step 1c 1d embedding generation and storing in vector stores'
#embedding = OpenAIEmbeddings(model="text-embedding-3-small" )
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #Hugging face is a Embedding model
vector_store = FAISS.from_documents(chunks, embedding) #FAISS is a vector store
print(vector_store.index_to_docstore_id)
first_id =vector_store.index_to_docstore_id[0] #we cannot use the UUIDs because the FAISS generates new id each time
print(vector_store.get_by_ids([first_id]))

"STEP 2 RETRIVAL input= querry output =list of doc"

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})
print(retriever)
print()
print()

#print(retriever.invoke("who is Vijay"))

"step 3 Augmented"

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.2)

prompt = PromptTemplate(
    template="""
    you are a helpful assistant .
    Answer the question only fro the provided transcription context.
    if the context is insufficient, just say you don't know
    
    {context}
    Question: {question}
    """,
    input_variables=[ 'context', 'question']
    )
question ="is the topic DMK discussed in this video? if yes then what was it"
print(retriever.invoke(question))