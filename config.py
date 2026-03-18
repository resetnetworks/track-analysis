import os
from dotenv import load_dotenv
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

S3_BUCKET = os.getenv("S3_BUCKET")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
TEMP_DIR = os.getenv("TEMP_DIR", "./temp")

os.makedirs(TEMP_DIR, exist_ok=True)