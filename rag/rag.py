from langchain_core.documents import Document
# from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain_community.embeddings import BedrockEmbeddings
import os
import re
BASE_DIR = os.path.dirname(__file__)
KB_PATH = os.path.join(BASE_DIR, "base1.txt")
VECTORSTORE_PATH = "vectorstore/sales_kb"
REGION = "us-east-1"


def load_knowledge():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return f.read()

def extract_section_name(text):
    """Extract section name from chunk text"""
    section_pattern = r'Section\s+(\d+):\s+(.+?)(?:\n|$)'
    match = re.search(section_pattern, text, re.MULTILINE)
    
    if match:
        section_num = match.group(1)
        section_title = match.group(2).strip()
        return f"Section {section_num}: {section_title}"
    return None

def chunk_knowledge(text: str):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"],
        chunk_size=500, # Large enough so it doesn't split mid-point
        chunk_overlap=0
    )

    chunks = splitter.split_text(text)
    current_section = "Introduction"
    documents = []
    
    for chunk in chunks:
        # Check if this chunk contains a section header
        section_name = extract_section_name(chunk)
        if section_name:
            current_section = section_name
        
        # Create document with section metadata
        doc = Document(
            page_content=chunk,
            metadata={"section": current_section}
        )
        documents.append(doc)
    
    return documents


def build_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

    return vectorstore
_EMBEDDINGS_MODEL = None
def get_retriever(transcript_segment, section_name):
    """
    It filters the 100-point database by the specific Section Metadata.
    """
    global _EMBEDDINGS_MODEL
    
    # 1. Initialize once and reuse to save memory and avoid Torch errors
    if _EMBEDDINGS_MODEL is None:
        _EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Explicitly set device to CPU
            encode_kwargs={'normalize_embeddings': False}
        )
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        _EMBEDDINGS_MODEL,
        allow_dangerous_deserialization=True
    )


    # The filter ensures the search only looks at the relevant 'aisle'
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5, 
            "filter": {"section": section_name}
        }
    )
    
    return retriever.invoke(transcript_segment)    

# def get_retriever(k: int = 4):
#     embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )


#     vectorstore = FAISS.load_local(
#         VECTORSTORE_PATH,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

#     return vectorstore.as_retriever(
#         search_kwargs={"k": k}
#     )
if __name__ == "__main__":
    # raw_text = load_knowledge()
    # docs = chunk_knowledge(raw_text)
    # build_vectorstore(docs)    
    # retriever = get_retriever(k=10)

    query = "Great to speak with you today thanks for taking the time. No problem we're looking at a few different options right now so I'm happy to chat. Great to start how many seats are you looking to fill currently? About 50 for the initial rollout. Okay and what software are you currently using for your lead tracking? We’re mostly on Excel and some Trello boards. Got it and what's your current monthly volume of inbound leads? Around 400. Okay are you using any marketing automation tools right now? Not really to be honest the current setup is just really frustrating for the team. I hear that a lot well our CRM is designed to be very user-friendly it’s actually one of the fastest on the market. That sounds great we definitely need something faster. Perfect I can show you how the speed helps with productivity during the demo by the way does your system do automated territory mapping? Yes we have a built-in module for that. Okay good send me some more info on that. I'd be happy to I’ll send over a PDF and a proposal by the end of the day. Great I'll need to think about it and talk to my boss. No problem at all I'll check back in with you in about two weeks."
    # docs = retriever.invoke(query)
    section_to_search = "Section 2: Objection Handling (Validate, Isolate, Reframe)"
    docs = get_retriever(query,section_to_search)
    print(f"\n🔍 Query: {query}\n")
    print(f"📄 Retrieved {len(docs)} documents:\n")

    for i, doc in enumerate(docs, 1):
        print(f"--- Document {i} ---")
        print(doc.page_content)
        print("Metadata:", doc.metadata)
        print()    
        
        
    