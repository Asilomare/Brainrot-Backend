import boto3
import cv2
import base64
import os
import json
import requests
from pinecone import Pinecone, ServerlessSpec
import tempfile
# import numpy as np
# from PIL import Image
# import io

# --- Environment Variables ---
PINECONE_API_SECRET_ARN = os.environ['PINECONE_API_SECRET_ARN']
AI_KEYS_SECRET_ARN = os.environ['AI_KEYS_SECRET_ARN']
PINECONE_ENVIRONMENT = os.environ['PINECONE_ENVIRONMENT']
PINECONE_INDEX_NAME = os.environ['PINECONE_INDEX_NAME']

# --- Constants ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
FRAME_EXTRACTION_RATE_SEC = 2  # Extract one frame every 2 seconds
EMBEDDING_MODEL = "text-embedding-ada-002"
VISION_MODEL = "google/gemini-flash-1.5"
EMBEDDING_DIMENSION = 1536 # For text-embedding-ada-002
PINECONE_UPSERT_BATCH_SIZE = 100

# --- Clients (initialized globally for reuse) ---
session = boto3.session.Session()
secrets_manager = session.client('secretsmanager')
s3_client = session.client('s3')

# --- Helper to get secrets ---
def get_secret(secret_arn):
    """Retrieve a secret from AWS Secrets Manager."""
    print(f"Retrieving secret from ARN: {secret_arn}")
    response = secrets_manager.get_secret_value(SecretId=secret_arn)
    if 'SecretString' in response:
        return json.loads(response['SecretString'])
    # If binary, decode it
    return json.loads(response['SecretBinary'].decode('utf-8'))

# --- Lazily initialize API keys and clients ---
pinecone_api_key = None
openrouter_api_key = None
pinecone_index = None

def init_clients():
    global pinecone_api_key, openrouter_api_key, pinecone_index
    if pinecone_api_key is None:
        pinecone_secret = get_secret(PINECONE_API_SECRET_ARN)
        pinecone_api_key = pinecone_secret.get('PINECONE_API_KEY')
        pc = Pinecone(api_key=pinecone_api_key)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        print("Pinecone client initialized.")

    if openrouter_api_key is None:
        ai_keys_secret = get_secret(AI_KEYS_SECRET_ARN)
        openrouter_api_key = ai_keys_secret.get('OPENROUTER_API_KEY')
        print("OpenRouter API key loaded.")


# --- Image and API Helpers ---

def get_frame_description(base64_frame):
    """Get a description of a single frame using OpenRouter's vision model."""
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in a concise sentence for a semantic video search. Focus on objects, actions, and the setting."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"}}
                ]
            }
        ]
    }
    print(f"Requesting description from {VISION_MODEL}...")
    response = requests.post(f"{OPENROUTER_API_URL}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    description = response.json()['choices'][0]['message']['content']
    print(f"Received description: {description}")
    return description

def get_text_embedding(text):
    """Get a text embedding using OpenRouter."""
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": [text]
    }
    print(f"Requesting embedding from {EMBEDDING_MODEL}...")
    response = requests.post(f"{OPENROUTER_API_URL}/embeddings", headers=headers, json=payload)
    response.raise_for_status()
    embedding = response.json()['data'][0]['embedding']
    print("Received embedding.")
    return embedding

def process_video(video_path, s3_key):
    """Extracts frames, gets descriptions & embeddings, and upserts to Pinecone."""
    print(f"Processing video: {video_path} for S3 key: {s3_key}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * FRAME_EXTRACTION_RATE_SEC)
    
    frame_number = 0
    vectors_to_upsert = []

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            break

        timestamp_sec = frame_number / fps

        # Convert frame to JPEG and then to base64
        _, buffer = cv2.imencode('.jpg', frame)
        base64_frame = base64.b64encode(buffer).decode('utf-8')

        try:
            description = get_frame_description(base64_frame)
            embedding = get_text_embedding(description)
            
            vector_id = f"{s3_key}|frame|{frame_number}"
            metadata = {
                "s3_key": s3_key,
                "description": description,
                "frame_number": frame_number,
                "timestamp_sec": round(timestamp_sec, 2)
            }
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

            # Upsert in batches
            if len(vectors_to_upsert) >= PINECONE_UPSERT_BATCH_SIZE:
                print(f"Upserting batch of {len(vectors_to_upsert)} vectors to Pinecone...")
                pinecone_index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []

        except requests.exceptions.HTTPError as e:
            print(f"API Error processing frame {frame_number}: {e.response.text}")
            # Decide if you want to skip the frame or stop the process
        except Exception as e:
            print(f"An unexpected error occurred at frame {frame_number}: {e}")

        frame_number += frame_interval

    # Upsert any remaining vectors
    if vectors_to_upsert:
        print(f"Upserting final batch of {len(vectors_to_upsert)} vectors...")
        pinecone_index.upsert(vectors=vectors_to_upsert)

    cap.release()
    print(f"Finished processing and upserting for {s3_key}")

# --- Lambda Handler ---

def lambda_handler(event, context):
    """Main entry point for the Lambda function."""
    print("Video analysis lambda triggered.")
    init_clients()

    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']
        
        # Avoid processing items from non-video folders or thumbnails if any
        if not object_key.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            print(f"Skipping non-video file: {object_key}")
            continue

        with tempfile.NamedTemporaryFile(suffix=f"_{os.path.basename(object_key)}") as tmp:
            video_path = tmp.name
            print(f"Downloading s3://{bucket_name}/{object_key} to {video_path}")
            s3_client.download_file(bucket_name, object_key, video_path)
            
            try:
                process_video(video_path, object_key)
            except Exception as e:
                print(f"Failed to process video {object_key}. Error: {e}")
                # Optional: add error handling/notification logic here
    
    return {
        'statusCode': 200,
        'body': json.dumps('Video analysis completed.')
    } 