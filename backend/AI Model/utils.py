
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
load_dotenv()

## load the GROQ API KEY 
groq_api_key = os.getenv('GROQ_API_KEY')

# Set your Groq API key
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

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

# Decision criteria prompt
DECISION_PROMPT = """
Based on the analysis of the provided government document, please review the following aspects:

{analysis_results}

Given this information, determine if the funding request should be APPROVED, REJECTED, or REVIEW (approved with conditions).
Provide a detailed justification for your decision, focusing on:
1. approved budget matching with the expenditure
2. Project details are clear and well-defined
3. no signs of fund misuse or discrepancies
4. Identified risks or concerns
5. Expected impact and outcomes

Your response should first state "DECISION: [APPROVED/REJECTED/REVIEW]" followed by your detailed justification.
do not just give review for all the documents, give a proper justification for the decision. also give a score out of 100 which tells the confidence level of the decision.
if confdence is near 60, give it for review, if it is above 70, give it for approval, if it is below 50, give it for rejection.
"""

def load_document(file):
    """Load a document from various file formats."""
    file_ext = os.path.splitext(file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(file.read())
        temp_path = temp_file.name
    
    try:
        if file_ext == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif file_ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(temp_path)
        elif file_ext in [".txt", ".md"]:
            loader = TextLoader(temp_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        documents = loader.load()
        return documents
    finally:
        os.unlink(temp_path)  # Clean up temp file

def create_rag_system(documents):
    """Create a RAG system from documents."""
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    # Using HuggingFace embeddings as an alternative to OpenAI embeddings
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     model_kwargs={'device': 'cpu'}
    # )
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )
    
    # Create QA chain with Groq instead of OpenAI
    llm = ChatGroq(
        model_name="llama3-70b-8192", 
        temperature=0
    )
    
    qa_prompt_template = """
    You are a government funding reviewer analyzing documents to determine if projects should receive funding.
    Use the following context to answer the question. If you don't know the answer, say "Information not found in document" rather than making up information.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    
    qa_prompt = PromptTemplate(
        template=qa_prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt}
    )
    
    return qa_chain

def analyze_document(qa_chain, questions=None):
    """Run a list of questions through the QA chain."""
    # Use standard questions if no custom questions provided
    if questions is None:
        questions = STANDARD_QUESTIONS
        
    results = []
    for question in questions:
        answer = qa_chain.invoke({"query": question})
        results.append({
            "Question": question,
            "Answer": answer["result"]
        })
    return results

def make_decision(analysis_results):
    """Make a funding decision based on analysis results."""
    llm = ChatGroq(
        model_name="llama3-70b-8192", 
        temperature=0.2
    )
    
    # Format analysis results for the prompt
    formatted_results = ""
    for result in analysis_results:
        formatted_results += f"Question: {result['Question']}\nAnswer: {result['Answer']}\n\n"
    
    decision_prompt = DECISION_PROMPT.format(analysis_results=formatted_results)
    decision = llm.invoke(decision_prompt)
    
    return decision.content

def process_document(file, custom_questions=None):
    """Process document and return analysis results and decision."""
    # Load document
    documents = load_document(file)
    
    # Create RAG system
    qa_chain = create_rag_system(documents)
    
    # Merge standard questions with custom questions if provided
    questions = STANDARD_QUESTIONS.copy()
    if custom_questions:
        questions.extend(custom_questions)
    
    # Run analysis
    analysis_results = analyze_document(qa_chain, questions)
    
    # Make approval decision
    decision_text = make_decision(analysis_results)
    
    # Determine status
    status = "REVIEW"  # Default
    if "DECISION: APPROVED" in decision_text:
        status = "APPROVED"
    elif "DECISION: REJECTED" in decision_text:
        status = "REJECTED"
    elif "DECISION: REVIEW" in decision_text:
        status = "REVIEW"
    
    # Generate report
    report = {
        "analysis": analysis_results,
        "decision": decision_text
    }
    
    return {
        "status": status,
        "report": report
    }