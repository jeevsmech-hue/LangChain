import os
from dataclasses import FrozenInstanceError

from dataclasses_json.stringcase import spinalcase
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (RunnableLambda, RunnableParallel,
                                      RunnablePassthrough)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import (NoTranscriptFound, TranscriptsDisabled,
                                    VideoUnavailable, YouTubeTranscriptApi)

load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

'step 1 A indexing Document loading '

video_id = "bMZBLfDriHI"

try:
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
# chunks = splitter.split_text(transcript)
print(len(chunks))

'step 1c 1d embedding generation and storing in vector stores'
# embedding = OpenAIEmbeddings(model="text-embedding-3-small" )
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Hugging face is an Embedding model
vector_store = FAISS.from_documents(chunks, embedding)  # FAISS is a vector store
print(vector_store.index_to_docstore_id)
first_id = vector_store.index_to_docstore_id[0]  # we cannot use the UUIDs because the FAISS generates new id each time
print(vector_store.get_by_ids([first_id]))

"STEP 2 RETRIVAL input= query output =list of doc"

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print(retriever)
print()
print()

print(retriever.invoke("who is Vijay"))

"step 3 Augmented"

# llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.2)

llm = ChatGroq(
    model="llama3-8b-8192",
    groq_api_key="GROQ_API_KEY"
)
# llm = ChatHuggingFace(llm=llm_endpoint)

prompt = PromptTemplate(
    template="""
    you are a helpful assistant .
    Answer the question only fro the provided transcription context.
    if the context is insufficient, just say you don't know

    {context}
    Question: {question}
    """,
    input_variables=['context', 'question']
)
question = "is the topic DMK discussed in this video? if yes then what was it"
docs = (retriever.invoke(question))
print(docs)
context_text = "\n\n".join(doc.page_content for doc in docs)
print(context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})
print(final_prompt)

"STEP 4 -Generation"

answer = llm.invoke(final_prompt)
print(answer.content)

"BUILDING CHAIN"


def format_docs(docs):
    context_text = "\n\n".join(doc.page_content for doc in docs)
    return context_text


parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough(),
})
print(parallel_chain.invoke('who is Vijay'))