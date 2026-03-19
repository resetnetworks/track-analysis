import boto3
import json
import os
from batch_analyze import analyze_track
from config import *

s3 = boto3.client("s3", region_name=AWS_REGION)
sqs = boto3.client("sqs", region_name=AWS_REGION)


def download_file(s3_key):
    local_path = os.path.join(TEMP_DIR, os.path.basename(s3_key))
    s3.download_file(S3_BUCKET, s3_key, local_path)
    return local_path


def extract_album_name(s3_key):
    parts = s3_key.split("/")
    if len(parts) >= 2:
        return parts[1]
    return None


def upload_result(result, original_key, msg_body):
    base_name = os.path.splitext(os.path.basename(original_key))[0]
    file_name = f"{base_name}.json"

    # this is seperation of album and track but useless now.
    # if msg_body.get("type") == "album":
    #     album_name = msg_body.get("album_name") or extract_album_name(original_key)

    #     output_key = f"output/albums/{album_name}/{file_name}"
    # else:
    #     output_key = f"output/singles/{file_name}"

    # single unified output path
    output_key = f"output/{file_name}"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=output_key,
        Body=json.dumps(result, indent=2),
        ContentType="application/json"
    )

    return output_key


def process_message(message):
    body = json.loads(message["Body"])
    # s3_key = body["file"]
    #code edited here for aws policy
    # Skip test events
    if "Records" not in body:
        print("Skipping non-S3 event")
        return message["ReceiptHandle"]

    record = body["Records"][0]
    s3_key = record["s3"]["object"]["key"]

    # decode URL encoding (spaces etc.)
    import urllib.parse
    s3_key = urllib.parse.unquote_plus(s3_key)

    print(f"Processing: {s3_key}")

    local_file = download_file(s3_key)

    result, _ = analyze_track(local_file)

    output_key = upload_result(result, s3_key, body)

    print(f"Done: {output_key}")

    # todo
    # need to add os.remove(local_file) as /temp will build up over-time

    return message["ReceiptHandle"]


def worker_loop():
    print("🚀 Worker started...")

    while True:
        response = sqs.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10
        )

        messages = response.get("Messages", [])

        for message in messages:
            try:
                receipt = process_message(message)

                sqs.delete_message(
                    QueueUrl=SQS_QUEUE_URL,
                    ReceiptHandle=receipt
                )

            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    worker_loop()
    