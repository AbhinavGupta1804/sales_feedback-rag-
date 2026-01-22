import boto3, time, uuid, requests

transcribe = boto3.client("transcribe", region_name="us-east-1")               #creates connection to aws transcribe service

def transcribe_audio(s3_uri: str, media_format="mp3"):  #s3_uri- location of audio file
    job_name = f"job-{uuid.uuid4()}"                    #unique job name

    transcribe.start_transcription_job(                 #reads audio directly from s3
        TranscriptionJobName=job_name,                  # & starts async transcription job 
        Media={"MediaFileUri": s3_uri},
        MediaFormat=media_format,
        LanguageCode="en-US"
    )

    while True:
        job = transcribe.get_transcription_job(
            TranscriptionJobName=job_name
        )["TranscriptionJob"]

        status = job["TranscriptionJobStatus"]
        if status == "COMPLETED":
            url = job["Transcript"]["TranscriptFileUri"]
            break
        if status == "FAILED":
            raise Exception("Transcription failed")
        time.sleep(5)

    data = requests.get(url).json()
    return data["results"]["transcripts"][0]["transcript"]

