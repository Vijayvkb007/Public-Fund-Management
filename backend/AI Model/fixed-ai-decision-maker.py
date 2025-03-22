import os
from abc import ABC, abstractmethod
from langchain import hub
from langsmith import Client
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict, Dict, Optional, Annotated, Literal
from dotenv import load_dotenv
load_dotenv()

# Load environment and get keys
os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')

# define a AI blueprint
class RAG_LLM(ABC):
    @abstractmethod
    def set_chat(self, LLM_Model, sub_model):
        pass
    
    @abstractmethod
    def set_embeddings(self, EmbeddingModel, model):
        pass
    
    @abstractmethod
    def load_and_split(self, file_path, _TextLoader, TextSplitter):
        pass
        
    @abstractmethod
    def vector_store(self, VectorModel):
        pass
    
    @abstractmethod
    def prompt(self, pull_from):
        pass
    
# define the state for application
class State(TypedDict):
    question: str
    questions: Optional[List[str]]
    context: List[Document]
    answer: str
    answers: Optional[Dict[str, str]]
    current_question_index: Optional[int]
    finished: Optional[bool]
    
class DecisionModel(RAG_LLM):
    def __init__(self, file_path, LLM_Model=ChatGroq, llm_submodel='gemma2-9b-it',
                 llm_api_key=os.getenv('GROQ_API_KEY'), EmbeddingModel=GoogleGenerativeAIEmbeddings,
                 embedding_submodel='models/embedding-001',_TextLoader=TextLoader, 
                 TextSplitter=RecursiveCharacterTextSplitter, VectorModel=FAISS, 
                 pull_from='project-fund-management-rag'):
        """Load the model and fetch the envirnoment variables"""
        print(f'Initiated the Model')
        self.set_chat(LLM_Model, llm_submodel, llm_api_key)
        self.set_embeddings(EmbeddingModel, embedding_submodel)
        self.load_and_split(file_path, _TextLoader, TextSplitter)
        self.vector_store(VectorModel)
        self.prompt(pull_from)
        self.build_graph()
        self.build_chain_qna_graph()
        
    def set_chat(self, LLM_Model=ChatGroq, sub_model='gemma2-9b-it', api_key=os.getenv('GROQ_API_KEY')):
        self.llm = LLM_Model(api_key=api_key, model=sub_model)
    
    def set_embeddings(self, EmbeddingModel=GoogleGenerativeAIEmbeddings, model='models/embedding-001'):
        self.embedding_model = EmbeddingModel(model=model)
    
    def load_and_split(self, file_path, _TextLoader=TextLoader, TextSplitter=RecursiveCharacterTextSplitter):
        self.text_loader = _TextLoader(file_path=file_path, autodetect_encoding=True)
        self.doc = self.text_loader.load()
        self.text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
        self.all_splits = self.text_splitter.split_documents(self.doc)
        return self.all_splits
    
    def vector_store(self, VectorModel=FAISS):
        self.vectorStore = VectorModel.from_documents(self.all_splits, self.embedding_model)
        self.index = self.vectorStore.add_documents(documents=self.all_splits)
        
    def prompt(self, pull_from='project-fund-management-rag'):
        self.client = Client(api_key=os.getenv('LANGSMITH_API_KEY'))
        self.prompt_ = self.client.pull_prompt(pull_from)   
    
    # Fixed retrieve function to properly return the state update    
    def retrieve(self, state: State):
        retrieved_docs = self.vectorStore.similarity_search(state['question'])
        # Return the updated state with context
        return {"context": retrieved_docs}
    
    # Fixed generate function to work with the state
    def generate(self, state: State):
        docs_content = '\n\n'.join(doc.page_content for doc in state['context'])
        message = self.prompt_.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(message)
        return {"answer": response.content}
    
    # Chain QnA functions
    def initialize_chain(self, state: State):
        """Initialize the chain with the first question."""
        if "questions" not in state or not state["questions"]:
            return {"finished": True}
        
        return {
            "current_question_index": 0,
            "question": state["questions"][0],
            "answers": {},
            "finished": False
        }
    
    def retrieve_for_chain(self, state: State):
        """Retrieves context for the current question in the chain."""
        if state.get("finished", False):
            return {}
            
        current_question = state["questions"][state["current_question_index"]]
        retrieved_docs = self.vectorStore.similarity_search(current_question)
        return {"context": retrieved_docs}
    
    def generate_for_chain(self, state: State):
        """Generates an answer for the current question in the chain."""
        if state.get("finished", False):
            return {}
            
        current_question = state["questions"][state["current_question_index"]]
        docs_content = '\n\n'.join(doc.page_content for doc in state['context'])
        message = self.prompt_.invoke({"question": current_question, "context": docs_content})
        response = self.llm.invoke(message)
        
        # Update answers dictionary with the current question's answer
        answers = state.get("answers", {})
        answers[current_question] = response.content
        
        return {"answer": response.content, "answers": answers}
    
    def should_continue(self, state: State) -> Literal["continue", "end"]:
        """Determine if we should continue to the next question or end."""
        if state.get("finished", False):
            return "end"
            
        if state["current_question_index"] < len(state["questions"]) - 1:
            return "continue"
        return "end"
    
    def next_question(self, state: State):
        """Move to the next question in the chain."""
        next_index = state["current_question_index"] + 1
        return {
            "current_question_index": next_index,
            "question": state["questions"][next_index]
        }
    
    def finish(self, state: State):
        """Mark the chain as finished."""
        return {"finished": True}
    
    def build_graph(self):
        """Build the single question graph."""
        self.graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        self.graph_builder.add_edge(START, 'retrieve')
        self.graph = self.graph_builder.compile()
    
    def build_chain_qna_graph(self):
        """Build the chain QnA graph for handling multiple questions."""
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("initialize", self.initialize_chain)
        workflow.add_node("retrieve", self.retrieve_for_chain)
        workflow.add_node("generate", self.generate_for_chain)
        workflow.add_node("next_question", self.next_question)
        workflow.add_node("finish", self.finish)
        
        # Add edges
        workflow.add_edge(START, "initialize")
        workflow.add_conditional_edges(
            "initialize",
            self.should_continue,
            {
                "continue": "retrieve",
                "end": "finish"
            }
        )
        workflow.add_edge("retrieve", "generate")
        workflow.add_conditional_edges(
            "generate",
            self.should_continue,
            {
                "continue": "next_question",
                "end": "finish"
            }
        )
        workflow.add_edge("next_question", "retrieve")
        workflow.add_edge("finish", END)
        
        # Compile graph
        self.chain_graph = workflow.compile()
        
    def get_response(self, question):
        """Get response for a single question."""
        self.response = self.graph.invoke({"question": question})
        print(self.response["answer"])
        return self.response["answer"]
    
    def get_chain_responses(self, questions):
        """Get responses for a list of questions in a single call."""
        if isinstance(questions, str):
            questions = [questions]
            
        chain_response = self.chain_graph.invoke({"questions": questions},{"recursion_limit": 100})
        print("\n==== Chain QnA Results ====")
        for q, a in chain_response.get("answers", {}).items():
            print(f"\nQ: {q}\nA: {a}\n{'-'*50}")
        return chain_response.get("answers", {})
        
if __name__ == "__main__":
    model = DecisionModel(file_path='Documents/bridge-progress-report.md', pull_from='prompt2')
    
    # Single question example
    # model.get_response("Is this Project Profitable?")
    
    # Chain QnA example
    questions = [
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
    model.get_chain_responses(questions)