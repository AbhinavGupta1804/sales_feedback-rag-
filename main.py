from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from graph.graph import final_graph
import boto3, uuid

from services.transcribe import transcribe_audio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3 = boto3.client("s3", region_name="us-east-1")

BUCKET = "sales-audio-us"

# ✅ Root endpoint - HTML file serve karega
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index2.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/transcribe")
async def upload_and_transcribe(file: UploadFile = File(...)):
    try:
        key = f"uploads/{uuid.uuid4()}-{file.filename}"

        s3.upload_fileobj(
            file.file,
            BUCKET,
            key
        )

        s3_uri = f"s3://{BUCKET}/{key}"

        transcript = transcribe_audio(
            s3_uri=s3_uri,
            media_format=file.filename.split(".")[-1]
        )
        init_state = {
            "transcript":transcript
            
        }
        final_state = final_graph.invoke(init_state)
        
        return {
            "status": "success",
            "s3_uri": s3_uri,
            "analysis": {
                "transcript": transcript,
                "call_summary": final_state.get("call_summary"),
                "customer_intent": final_state.get("customer_intent"),
                "rep_performance": final_state.get("rep_performance"),
                "what_went_well": final_state.get("what_went_well"),
                "what_to_improve": final_state.get("what_to_improve"),
                "recommended_next_actions": final_state.get("recommended_next_actions"),
                "objection_analysis": final_state.get("objection_analysis")
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from graph.graph import final_graph
# import boto3, uuid

# from services.transcribe import transcribe_audio  # your file

# app = FastAPI()

# s3 = boto3.client("s3", region_name="us-east-1")

# BUCKET = "sales-audio-us"   # your bucket name

# @app.post("/transcribe")
# async def upload_and_transcribe(file: UploadFile = File(...)):
#     try:
#         # 1️⃣ Create unique S3 key
#         key = f"uploads/{uuid.uuid4()}-{file.filename}"

#         # 2️⃣ Upload audio to S3
#         s3.upload_fileobj(
#             file.file,
#             BUCKET,
#             key
#         )

#         # 3️⃣ Create S3 URI
#         s3_uri = f"s3://{BUCKET}/{key}"

#         # 4️⃣ Call AWS Transcribe
#         transcript = transcribe_audio(
#             s3_uri=s3_uri,
#             media_format=file.filename.split(".")[-1]
#         )
#         init_state = {
#             "transcript":transcript
            
#         }
#         final_state = final_graph.invoke(init_state)
        
#         return {
#             "status": "success",
#             "s3_uri": s3_uri,
#             "analysis": {
#                 "transcript": transcript,
#                 "call_summary": final_state.get("call_summary"),
#                 "customer_intent": final_state.get("customer_intent"),
#                 "rep_performance": final_state.get("rep_performance"),
#                 "what_went_well": final_state.get("what_went_well"),
#                 "what_to_improve": final_state.get("what_to_improve"),
#                 "recommended_next_actions": final_state.get("recommended_next_actions"),
#                 "objection_analysis": final_state.get("objection_analysis")
#             }
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
