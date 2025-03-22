import os
from langchain import hub
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

os.environ['GROQ_API_KEY'] = GROQ_API_KEY
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# LLM MODEL
llm = ChatGroq(api_key=GROQ_API_KEY, model='gemma2-9b-it')

# EMBEDDING MODEL
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

# DOCUMENT LOADER
text_loader = TextLoader(file_path='Documents/bridge-progress-report.md', autodetect_encoding=True)

doc = text_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(doc)

# VECTOR STORE
vector_store = FAISS.from_documents(all_splits, embeddings)

# INDEX CHUNK
_ = vector_store.add_documents(documents=all_splits)

# PROMPT
prompt = hub.pull('rlm/rag-prompt')

# Define the state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    
# application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state['question'])
    print(f"Retrived Docs: \n\n{retrieved_docs}\n\n____________________________")
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = '\n\n'.join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
    
response = graph.invoke({"question":"Is this Project profitable?"})
print(response["answer"])