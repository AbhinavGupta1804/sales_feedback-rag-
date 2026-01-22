from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
load_dotenv()
import os
import re
import time

# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.dirname(__file__)
KB_PATH = os.path.join(BASE_DIR, "base1.txt")
PINECONE_INDEX_NAME = "sales-kb"  # Change this to your preferred index name

# -------------------------
# Load Knowledge Base
# -------------------------
def load_knowledge():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------
# Section Extraction
# -------------------------
def extract_section_name(text):
    """
    Extract section name from chunk text
    Example: Section 2: Objection Handling (Validate, Isolate, Reframe)
    """
    section_pattern = r"Section\s+(\d+):\s+(.+?)(?:\n|$)"
    match = re.search(section_pattern, text, re.MULTILINE)
    if match:
        section_num = match.group(1)
        section_title = match.group(2).strip()
        return f"Section {section_num}: {section_title}"
    return None

# -------------------------
# Chunking
# -------------------------
def chunk_knowledge(text: str):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"],
        chunk_size=500,
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

        doc = Document(
            page_content=chunk,
            metadata={"section": current_section}
        )
        documents.append(doc)

    return documents

# -------------------------
# Build Vectorstore
# -------------------------
def build_vectorstore(documents):
    """
    Builds Pinecone index using OpenAI embeddings
    """
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Check if index exists, if not create it
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # text-embedding-3-small dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Change to your preferred region
            )
        )
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            time.sleep(1)
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create vector store
    vectorstore = PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    
    return vectorstore

# -------------------------
# Retriever
# -------------------------
_EMBEDDINGS_MODEL = None

def get_retriever(transcript_segment, section_name):
    """
    Filters the database by Section Metadata and retrieves relevant chunks
    """
    global _EMBEDDINGS_MODEL
    
    # Initialize once and reuse
    if _EMBEDDINGS_MODEL is None:
        _EMBEDDINGS_MODEL = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # Load vectorstore
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=_EMBEDDINGS_MODEL
    )
    
    # Create retriever with metadata filter
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {"section": {"$eq": section_name}}
        }
    )
    
    return retriever.invoke(transcript_segment)

# -------------------------
# Test Run
# -------------------------
if __name__ == "__main__":
    # STEP 1: Build index once (uncomment to rebuild)
    # raw_text = load_knowledge()
    # docs = chunk_knowledge(raw_text)
    # build_vectorstore(docs)
    
    print("OPENAI KEY FOUND?", os.getenv("OPENAI_API_KEY") is not None)
    print("PINECONE KEY FOUND?", os.getenv("PINECONE_API_KEY") is not None)
    
    # STEP 2: Query
    query = """
    Great to speak with you today thanks for taking the time.
    No problem we're looking at a few different options right now so I'm happy to chat.
    Great to start how many seats are you looking to fill currently?
    About 50 for the initial rollout.
    Okay and what software are you currently using for your lead tracking?
    We're mostly on Excel and some Trello boards.
    Got it and what's your current monthly volume of inbound leads?
    Around 400.
    Okay are you using any marketing automation tools right now?
    Not really to be honest the current setup is just really frustrating for the team.
    I hear that a lot well our CRM is designed to be very user-friendly it's actually one of the fastest on the market.
    That sounds great we definitely need something faster.
    Perfect I can show you how the speed helps with productivity during the demo by the way does your system do automated territory mapping?
    Yes we have a built-in module for that.
    Okay good send me some more info on that.
    I'd be happy to I'll send over a PDF and a proposal by the end of the day.
    Great I'll need to think about it and talk to my boss.
    No problem at all I'll check back in with you in about two weeks.
    """
    raw_text = load_knowledge()
    docs = chunk_knowledge(raw_text)
    build_vectorstore(docs)    
    section_to_search = "Section 2: Objection Handling (Validate, Isolate, Reframe)"
    docs = get_retriever(query, section_to_search)
    
    print(f"\n🔍 Query:\n{query}\n")
    print(f"📄 Retrieved {len(docs)} documents:\n")
    for i, doc in enumerate(docs, 1):
        print(f"--- Document {i} ---")
        print(doc.page_content)
        print("Metadata:", doc.metadata)
        print()