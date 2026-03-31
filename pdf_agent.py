#indexing

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv() 
#document loading and indexing

pdf_path =r"C:\Users\Jeeva\Downloads\pc-dmis.pdf"
try :
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    print(document)
except Exception as e:
    print(f"Error loading PDF: {e}")

#text splitting
recrussive_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=9)
chunks = recrussive_splitter.split_documents(document)
print()
print()
print(chunks)
print(len(chunks))

from langchain_community.vectorstores import FAISS
#embedding and vector store
from langchain_huggingface import HuggingFaceEmbeddings

embedding =HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Hugging face is an Embedding model
vector_store = FAISS.from_documents(chunks, embedding) # FAISS is a vector store
print()
print()
print(vector_store.index_to_docstore_id)     
first_sentence_id = vector_store.index_to_docstore_id[0] 
print()
'print(vector_store.get_by_ids([first_sentence_id]))'

#STEP 2 RETRIVAL input= query output =list of doc

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}) #similarity search

query = "What is the purpose of the pc-dmis software?"
print()
print()

retriever.invoke(query)

#step 3 Augmented generation
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

#1. llm = ChatGroq(model="llama3-8b-8192",groq_api_key="GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile")

#2. prompt template
prompt = PromptTemplate(
    template="""you are a helpful assistant.
                        Answer the question based on the provided context,
                        if the context is insufficient please say you dont know.
                        {context}
                        Question: {question}
                        """,
    input_variables=['context', 'question'],
)
question = "is the topic of PC-DMIS mentioned in the document?if yes, What is the purpose of the pc-dmis software?"
retrieve_docs = retriever.invoke(question)
'print(retrieve_docs)'

context_text="\n\n".join([doc.page_content for doc in retrieve_docs])
print()
print()
print()

'print(context_text)'

final_prompt = prompt.invoke({"context": context_text, "question": question})
print()
print()
print()
'print(final_prompt)'



answer = llm.invoke(final_prompt)

print()
print()
print()
'print(answer.content)'

#Chain = retriever + prompt + llm
chain =({"context": retriever, "question": RunnablePassthrough()})

def format_docs(retrieved_docs):
    return "\n\n".join([doc.page_content for doc in retrieved_docs])

parallel_chain=RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

print(parallel_chain.invoke("What is the area of usage in the document?"))

parser = StrOutputParser()
final_chain = parallel_chain | prompt | llm | parser
print(final_chain.invoke("summarize the document?"))