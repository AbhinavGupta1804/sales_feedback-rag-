# Sales Coaching RAG Platform

AI-powered system that transcribes sales calls, analyzes conversations using LLM agents, and generates structured coaching feedback using a RAG knowledge base.

---

## Prerequisites

- Python 3.10+
- Docker
- Git
- AWS CLI (configured)
- OpenAI API Key
- Pinecone API Key + Index

---

## Clone Repository

```bash
git clone https://github.com/your-username/sales-coaching-rag.git
cd sales-coaching-rag
```

---

## Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_key

PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=sales-kb

AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1

S3_BUCKET_NAME=your_s3_bucket_name
```

---

## Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open: http://localhost:8000

---

## Run with Docker

```bash
docker build -t salesfeedback .
docker run -p 8000:8000 --env-file .env salesfeedback
```

---

