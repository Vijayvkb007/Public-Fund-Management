import os
from abc import ABC, abstractmethod
from langchain import hub
from langsmith import Client
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from dotenv import load_dotenv
load_dotenv()

# Load environment and get keys
os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')

# define a AI blueprint
class RAG_LLM(ABC):
    @abstractmethod
    def set_chat(self):
        pass
    
    @abstractmethod
    def set_embeddings(self):
        pass
    
    def load_and_split(self):
        pass
        
    @abstractmethod
    def vector_store(self):
        pass
    
    @abstractmethod
    def prompt_(self, pull_from):
        pass
    
# Load llm models
llm = ChatGroq(api_key=GROQ_API_KEY, model='gemma2-9b-it')

# define the state for application
class AnalysisState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    
class DecisionState(TypedDict):
    analysis_results: str
    decision: str
    
class AnalysisModel(RAG_LLM):
    def __init__(self, file_path, pull_from='project-fund-management-rag'):
        """Load the model and fetch the env variables"""
        print(f'Initiated the Analysis Model')
        self.set_chat()
        self.set_embeddings()
        self.load_and_split(file_path)
        self.vector_store()
        self.prompt_(pull_from)
        self.build_graph()
        self.results = ""
        
    def set_chat(self):
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model='gemma2-9b-it')
    
    def set_embeddings(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        
    def load_and_split(self, file_path):
        self.text_loader = TextLoader(file_path=file_path, autodetect_encoding=True)
        self.doc = self.text_loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.all_splits = self.text_splitter.split_documents(self.doc)
        return self.all_splits
    
    def vector_store(self):
        self.vectorStore = FAISS.from_documents(self.all_splits, self.embeddings)
        self.index = self.vectorStore.add_documents(documents=self.all_splits)
        
    def prompt_(self, pull_from):
        self.client = Client(api_key=LANGSMITH_API_KEY)
        self.prompt = self.client.pull_prompt(pull_from)
        
    def retrieve(self, state:AnalysisState):
        self.retrived_docs = self.vectorStore.similarity_search(state['question'])
        self.context = {"context":self.retrived_docs}
        return self.context
    
    def generate(self, state:AnalysisState):
        self.docs_content = '\n'.join(doc.page_content for doc in state['context'])
        self.message = self.prompt.invoke({"question":state['question'], "context":self.docs_content})
        self.response = self.llm.invoke(self.message)
        return {"answer": self.response.content}
    
    def build_graph(self):
        self.graph_builder = StateGraph(AnalysisState).add_sequence([self.retrieve, self.generate])
        self.graph_builder.add_edge(START, "retrieve")
        self.graph = self.graph_builder.compile()
    
    def get_response(self, question):
        self.response = self.graph.invoke({"question": question})
        self.results += self.response["answer"]
        # print(self.response["answer"])
        return self.response['answer']
        
class DecisionModel(RAG_LLM):
    def __init__(self, analysis_result, pull_from='final-decision-maker'):
        print(f"\n{"_"*10}\nInitiating DecisionModel")
        self.set_chat()
        self.set_embeddings()
        self.load_and_split(analysis_result)
        # self.vector_store()
        self.prompt_(pull_from)
        self.build_graph()
        self.response = self.graph.invoke({"analysis_results":analysis_result})
        self.final_decision = self.response["decision"]
        
    def verdict(self):
        print(self.final_decision)
        return self.final_decision
    
    def set_chat(self):
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model='gemma2-9b-it')
    
    def set_embeddings(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    
    def load_and_split(self, analysis_result):
        self.all_splits = analysis_result
    
    def vector_store(self):
        # ig no need
        self.vectorStore = FAISS.from_documents(self.all_splits, self.embeddings)
        self.index = self.vectorStore.add_documents(documents=self.all_splits)
        
    def prompt_(self, pull_from):
        self.client = Client(api_key=LANGSMITH_API_KEY)
        self.prompt = self.client.pull_prompt(pull_from)
        
    def generate(self, state:DecisionState):
        self.docs_content = ''.join(doc for doc in state['analysis_results'])
        self.message = self.prompt.invoke({"analysis_results":self.docs_content})
        self.repsonse = self.llm.invoke(self.message)
        return {"decision": self.repsonse.content}
    
    def build_graph(self):
        self.graph_builder = StateGraph(DecisionState).add_sequence([self.generate])
        self.graph_builder.add_edge(START, 'generate')
        self.graph = self.graph_builder.compile()

if __name__ == "__main__":
    analysis_model = AnalysisModel(file_path='Documents/bridge-progress-report.md', pull_from='rlm/rag-prompt')
    # Define standard evaluation questions
    STANDARD_QUESTIONS = [
        "What is the amount of budget installment approved from government?",
        "What are the main objectives of the project?",
        "What is the timeline for project implementation?",
        "What specific outcomes or deliverables are expected?",
        "how fund is being utilized for different work, and does this match with the expenditure?",
        "Is there a detailed breakdown of how funds will be utilized?",
        "Does the project align with government priorities and policies?",
        "Is there evidence of proper planning and risk management?",
        "Is the fund released by government matches with the expenditure?",
        "Are there any red flags or concerns in the document?",
        "is there any disperencies in fund utilization?",
    ]
    
    analysis_result = ""
    for question in STANDARD_QUESTIONS:
        analysis_result += analysis_model.get_response(question)
        print(analysis_model.get_response(question))
        
    decision_model = DecisionModel(analysis_result, pull_from='final-decision-maker')
    decision_model.verdict()
        
# class AnalysisModel(RAG_LLM):
#     def __init__(self, file_path, LLM_Model=ChatGroq, llm_submodel='gemma2-9b-it',
#                  llm_api_key=os.getenv('GROQ_API_KEY'), EmbeddingModel=GoogleGenerativeAIEmbeddings,
#                  embedding_submodel='models/embedding-001',_TextLoader=TextLoader, 
#                  TextSplitter=RecursiveCharacterTextSplitter, VectorModel=FAISS, 
#                  pull_from='project-fund-management-rag'):
#         """Load the model and fetch the envirnoment variables"""
#         print(f'Initiated the Model')
#         self.set_chat(LLM_Model, llm_submodel, llm_api_key)
#         self.set_embeddings(EmbeddingModel, embedding_submodel)
#         self.load_and_split(file_path, _TextLoader, TextSplitter)
#         self.vector_store(VectorModel)
#         self.prompt(pull_from)
#         self.build_graph()
#         self.results = ""
        
        
#     def set_chat(self, LLM_Model=ChatGroq, sub_model='gemma2-9b-it', api_key=os.getenv('GROQ_API_KEY')):
#         self.llm = LLM_Model(api_key=api_key, model=sub_model)
    
#     def set_embeddings(self, EmbeddingModel=GoogleGenerativeAIEmbeddings, model='models/embedding-001'):
#         self.embedding_model = EmbeddingModel(model=model)
    
#     def load_and_split(self, file_path, _TextLoader=TextLoader, TextSplitter=RecursiveCharacterTextSplitter):
#         self.text_loader = _TextLoader(file_path=file_path, autodetect_encoding=True)
#         self.doc = self.text_loader.load()
#         self.text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
#         self.all_splits = self.text_splitter.split_documents(self.doc)
#         return self.all_splits
    
#     def vector_store(self, VectorModel=FAISS):
#         self.vectorStore = VectorModel.from_documents(self.all_splits, self.embedding_model)
#         self.index = self.vectorStore.add_documents(documents=self.all_splits)
        
#     def prompt(self, pull_from='project-fund-management-rag'):
#         self.client = Client(api_key=os.getenv('LANGSMITH_API_KEY'))
#         self.prompt_ = self.client.pull_prompt(pull_from)   
        
#     def retrieve(self, state: State):
#         self.retrived_docs = self.vectorStore.similarity_search(state['question'])
#         self.context = {"context": self.retrived_docs}
#         return self.context
        
#     def generate(self, state: State):
#         self.docs_content = '\n'.join(doc.page_content for doc in state['context'])
#         self.message = self.prompt_.invoke({"question":state["question"], "context":self.docs_content})
#         self.response = self.llm.invoke(self.message)
#         return {"answer": self.response.content}
    
#     def build_graph(self):
#         self.graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
#         self.graph_builder.add_edge(START, 'retrieve')
#         self.graph = self.graph_builder.compile()
        
#     def get_response(self, question):
#         self.response = self.graph.invoke({"question":question})
#         self.results += self.response["answer"]
#         print(self.response["answer"])
        
# if __name__ == "__main__":
#     model = AnalysisModel(file_path='Documents/bridge-progress-report.md', pull_from='rlm/rag-prompt')
#     model.get_response("Is the fund released by government matches with the expenditure?")
#     model.get_response("how fund is being utilized for different work, and does this match with the expenditure?")
#     print(model.results)
    
#     print(f"Initiated the Decision making bot:\n")
#     llm = ChatGroq(api_key=GROQ_API_KEY, model='gemma2-9b-it')
#     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    
#     vector_store = FAISS.from_documents(model.results, embeddings)
#     _ = vector_store.add_documents(documents=model.results)
    
#     client = Client(os.getenv('LANGSMITH_API_KEY'))
#     prompt = client.pull_prompt('final-decision-maker')
    
#     def retrieve(state: State2):
#         retrieved_docs = vector_store.similarity_search(state['analysis_results'])
#         return {"context": retrieved_docs}

#     def generate(state: State2):
#         docs_content = '\n'.join(doc.page_content for doc in state["analysis_results"])
#         messages = prompt.invoke({"analysis_results": state["analysis_results"]})
#         response = llm.invoke(messages)
#         return {"answer": response.content}
    
    
#     graph_builder = StateGraph(State).add_sequence([retrieve, generate])
#     graph_builder.add_edge(START, "retrieve")
#     graph = graph_builder.compile()
#     response = graph.invoke({"analysis_results": model.results})
#     print(response)