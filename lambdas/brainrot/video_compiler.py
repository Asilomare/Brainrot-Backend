#!/usr/bin/env python3
"""
Video Compiler Script

This script creates video compilations from random videos with music overlay.
It handles both portrait and landscape videos and processes videos from S3.
"""

import os
import json
import random
import sys
import subprocess
import tempfile
import shutil
from datetime import datetime
import boto3
import uuid
import traceback
from os import environ
import math
import struct
from decimal import Decimal
from openai import OpenAI
from pinecone import Pinecone

# Custom JSON encoder for handling Decimal types from DynamoDB.
class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Decimal types from DynamoDB."""
    def default(self, o):
        if isinstance(o, Decimal):
            if o % 1 == 0:
                return int(o)
            else:
                return float(o)
        return super(DecimalEncoder, self).default(o)

# Initialize S3 client
s3_client = boto3.client('s3')

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
requests_table = dynamodb.Table(environ['MONTAGE_REQUESTS_TABLE'])

# --- Environment Variables ---
PINECONE_API_SECRET_ARN = os.environ.get('PINECONE_API_SECRET_ARN')
AI_KEYS_SECRET_ARN = os.environ.get('AI_KEYS_SECRET_ARN')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')
EMBEDDING_MODEL = "text-embedding-ada-002"

# --- Clients (initialized lazily) ---
pinecone_index = None
openai_client = None

# --- Helper to get secrets ---
def get_secret(secret_arn):
    """Retrieve a secret from AWS Secrets Manager."""
    if not secret_arn:
        raise ValueError("Secret ARN is not configured.")
    print(f"Retrieving secret from ARN: {secret_arn}")
    secrets_manager = boto3.client('secretsmanager')
    response = secrets_manager.get_secret_value(SecretId=secret_arn)
    if 'SecretString' in response:
        return json.loads(response['SecretString'])
    return json.loads(response['SecretBinary'].decode('utf-8'))

def init_ai_clients():
    """Initializes Pinecone and OpenAI clients."""
    global pinecone_index, openai_client
    if pinecone_index is None:
        pinecone_secret = get_secret(PINECONE_API_SECRET_ARN)
        pinecone_api_key = pinecone_secret.get('PINECONE_API_KEY')
        pc = Pinecone(api_key=pinecone_api_key)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        print("Pinecone client initialized.")

    if openai_client is None:
        ai_keys_secret = get_secret(AI_KEYS_SECRET_ARN)
        openai_api_key = ai_keys_secret.get('OPENAI_API_KEY')
        openai_client = OpenAI(api_key=openai_api_key)
        print("OpenAI client initialized.")

def get_text_embedding(text):
    """Get a text embedding using the OpenAI API."""
    init_ai_clients()
    print(f"Requesting embedding from OpenAI's {EMBEDDING_MODEL}...")
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    embedding = response.data[0].embedding
    print("Received embedding from OpenAI.")
    return embedding

def get_videos_from_prompt(prompt, num_clips):
    """Query Pinecone to get the most relevant video clips for a prompt."""
    print(f"Getting video clips for prompt: '{prompt}'")
    prompt_embedding = get_text_embedding(prompt)

    # Query Pinecone, get more than needed to allow for deduplication
    query_response = pinecone_index.query(
        vector=prompt_embedding,
        top_k=num_clips * 5,  # Fetch more to ensure variety
        include_metadata=True
    )

    # Deduplicate based on s3_key to get varied videos
    unique_s3_keys = {}
    for match in query_response['matches']:
        s3_key = match['metadata']['s3_key']
        if s3_key not in unique_s3_keys:
            unique_s3_keys[s3_key] = match

    # Get the top N unique video keys
    selected_videos = list(unique_s3_keys.keys())[:num_clips]

    if not selected_videos:
        raise ValueError(f"Could not find any relevant videos for the prompt: '{prompt}'")

    print(f"Found {len(selected_videos)} relevant videos from prompt.")
    return selected_videos

# Load configuration
def load_config():
    """Load configuration from config.json file."""
    print("Loading configuration from config.json")
    with open('config.json', 'r') as f:
        return json.load(f)

# Get random videos from S3 bucket
def get_random_videos(bucket_name, prefix, num_videos=None):
    """Get a list of random videos from the specified S3 bucket and prefix."""
    print(f"Getting random videos from bucket: {bucket_name}, prefix: {prefix}")
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=prefix
    )
    
    if 'Contents' not in response:
        raise ValueError(f"No files found in bucket '{bucket_name}' with prefix '{prefix}'")
    
    video_files = [item['Key'] for item in response['Contents'] 
                  if item['Key'].lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    
    if not video_files:
        raise ValueError(f"No video files found in bucket '{bucket_name}' with prefix '{prefix}'")
    
    if num_videos and num_videos < len(video_files):
        selected_videos = random.sample(video_files, num_videos)
        print(f"Selected {len(selected_videos)} random videos from {len(video_files)} available videos")
        return selected_videos
    print(f"Using all {len(video_files)} available videos")
    return video_files

# Get random music file from S3 bucket
def get_random_music(bucket_name, prefix):
    """Get a random music file from the specified S3 bucket and prefix."""
    print(f"Getting random music from bucket: {bucket_name}, prefix: {prefix}")
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=prefix
    )
    
    if 'Contents' not in response:
        raise ValueError(f"No files found in bucket '{bucket_name}' with prefix '{prefix}'")
    
    music_files = [item['Key'] for item in response['Contents'] 
                  if item['Key'].lower().endswith(('.mp3', '.wav', '.ogg'))]
    
    if not music_files:
        raise ValueError(f"No music files found in bucket '{bucket_name}' with prefix '{prefix}'")
    
    selected_music = random.choice(music_files)
    print(f"Selected music file: {selected_music}")
    return selected_music

# Download file from S3
def download_from_s3(bucket_name, key, local_path):
    """Download a file from S3 to a local path."""
    print(f"Downloading from S3: {bucket_name}/{key} to {local_path}")
    s3_client.download_file(bucket_name, key, local_path)
    return local_path

# Upload file to S3
def upload_to_s3(local_path, bucket_name, key):
    """Upload a file to S3."""
    print(f"Uploading to S3: {local_path} to {bucket_name}/{key}")
    s3_client.upload_file(local_path, bucket_name, key)
    return f"s3://{bucket_name}/{key}"

# Helper functions adapted from test.py for reading rotation from header
def _read_atom(datastream):
    """Read an atom and return a tuple of (size, type)."""
    try:
        size_bytes = datastream.read(4)
        type_bytes = datastream.read(4)
        if not size_bytes or not type_bytes:
            return None, None # Indicate end of stream or error
        size = struct.unpack(">L", size_bytes)[0]
        atom_type = type_bytes # Keep as bytes for comparison
        return size, atom_type
    except struct.error:
        print("Struct unpack error reading atom header.")
        return None, None

def _get_index(datastream):
    """Return an index of top-level atoms."""
    index = []
    start_pos = datastream.tell()
    datastream.seek(0, os.SEEK_END)
    file_size = datastream.tell()
    datastream.seek(start_pos)

    while datastream.tell() < file_size:
        current_pos = datastream.tell()
        atom_size, atom_type = _read_atom(datastream)

        if atom_size is None:
            print("Atom read failed, stopping index creation.")
            break

        if atom_size < 8:
            print(f"Warning: Atom size {atom_size} is less than 8 bytes. Stopping.")
            break # Avoid potential infinite loops or errors

        index.append((atom_type, current_pos, atom_size))
        next_pos = current_pos + atom_size

        if next_pos > file_size:
             print(f"Warning: Atom {atom_type.decode('latin-1', errors='ignore')} size {atom_size} exceeds file boundary. Stopping.")
             break

        try:
            datastream.seek(next_pos)
        except OSError as e:
            print(f"Error seeking to next atom position {next_pos}: {e}")
            break

    # Basic validation
    top_level_atoms = {item[0] for item in index}
    required_atoms = {b"ftyp", b"moov", b"mdat"}
    if not required_atoms.issubset(top_level_atoms):
        print(f"Warning: Missing required top-level atoms. Found: {[a.decode('latin-1', errors='ignore') for a in top_level_atoms]}")
        # Don't raise an error here, let the calling function handle it if 'moov' is missing later

    return index


def _find_atoms(size, datastream):
    """Generator yielding 'mvhd' or 'tkhd' atoms within a parent atom."""
    stop = datastream.tell() + size

    while datastream.tell() < stop:
        current_pos = datastream.tell()
        atom_size, atom_type = _read_atom(datastream)

        if atom_size is None or atom_type is None:
            print("Read atom failed during find_atoms.")
            break

        if atom_size < 8:
             print(f"Warning: Atom size {atom_size} < 8 during find_atoms. Stopping search in this branch.")
             break # Stop searching this branch

        end_of_atom = current_pos + atom_size
        if end_of_atom > stop:
            print(f"Warning: Atom {atom_type.decode('latin-1', errors='ignore')} size {atom_size} exceeds parent boundary {stop}. Skipping.")
            # Attempt to recover by seeking to parent boundary? Or just break? Let's break.
            break


        if atom_type == b"trak":
            # Search within 'trak' atom
            yield from _find_atoms(atom_size - 8, datastream)
        elif atom_type in [b"mvhd", b"tkhd"]:
            yield atom_type
            # Ensure stream position is correct after yielding
            datastream.seek(end_of_atom)
        else:
            # Ignore other atoms, seek to the end of it
            try:
                datastream.seek(end_of_atom)
            except OSError as e:
                 print(f"Error seeking past atom {atom_type.decode('latin-1', errors='ignore')} to position {end_of_atom}: {e}")
                 break # Stop searching this branch if seek fails

        # Defensive check against infinite loops if seek didn't advance
        if datastream.tell() <= current_pos:
             print(f"Warning: Stream position did not advance after processing atom {atom_type.decode('latin-1', errors='ignore')}. Breaking loop.")
             break


def _get_rotation_from_header(filename):
    """Attempt to read rotation information directly from the MOOV atom."""
    print(f"Attempting to read rotation from header for: {filename}")
    degrees = set()
    moov_pos = -1
    moov_size = 0

    try:
        with open(filename, "rb") as datastream:
            index = _get_index(datastream)

            # Find the 'moov' atom position and size
            for atom_type, pos, size in index:
                if atom_type == b"moov":
                    moov_pos = pos
                    moov_size = size
                    break
            else:
                print("Error: 'moov' atom not found in the file index.")
                return None # Indicate 'moov' not found

            if moov_pos == -1 or moov_size < 8:
                 print("Error: Invalid 'moov' atom position or size.")
                 return None


            # Seek to the start of 'moov' atom's content
            datastream.seek(moov_pos + 8)

            # Iterate through relevant atoms within 'moov'
            for atom_type in _find_atoms(moov_size - 8, datastream):
                try:
                    vf = datastream.read(4)
                    if len(vf) < 4: # Check if read was successful
                         print(f"Error: Could not read version/flags for {atom_type.decode('latin-1', errors='ignore')}")
                         continue # Skip this atom
                    version = struct.unpack(">Bxxx", vf)[0]
                    # flags = struct.unpack(">L", vf)[0] & 0x00ffffff # Flags not needed

                    # Determine offset to the matrix based on version and type
                    offset_to_matrix = -1
                    if version == 1:
                        if atom_type == b"mvhd":
                            offset_to_matrix = 28 + 16 # created + modified + timescale + duration + rate + volume + reserved
                        elif atom_type == b"tkhd":
                             offset_to_matrix = 32 + 16 # created + modified + trackid + reserved + duration + reserved + layer + alt_group + volume + reserved
                    elif version == 0:
                         if atom_type == b"mvhd":
                             offset_to_matrix = 16 + 16
                         elif atom_type == b"tkhd":
                             offset_to_matrix = 20 + 16
                    else:
                         print(f"Warning: Unknown atom version {version} for {atom_type.decode('latin-1', errors='ignore')}. Skipping matrix read.")
                         # Need to skip the rest of this atom correctly
                         # Assuming fixed size for known atoms - this part is brittle
                         skip_bytes = 0
                         if atom_type == b"mvhd": skip_bytes = 60 + 28 if version == 1 else 60 + 16 # matrix(36) + predefined(24) + next_track_id(4)
                         elif atom_type == b"tkhd": skip_bytes = 36 + 8 if version == 1 else 36 + 8 # matrix(36) + width(4) + height(4)
                         else: skip_bytes = 0 # fallback, likely incorrect

                         # Calculate the position *before* reading version/flags
                         atom_content_start_pos = datastream.tell() - 4
                         # Seek from the start of the content past the known fields + matrix + remaining
                         target_seek_pos = atom_content_start_pos + offset_to_matrix + skip_bytes if offset_to_matrix != -1 else datastream.tell() + skip_bytes # Rough estimate
                         print(f"Attempting to skip unknown version atom to position ~{target_seek_pos}")
                         try:
                              datastream.seek(target_seek_pos) # This might be inaccurate
                         except OSError as e:
                              print(f"Error seeking past unknown version atom: {e}")
                              # Consider breaking the loop or returning error
                         continue # Skip to next atom in _find_atoms


                    if offset_to_matrix != -1:
                         # Seek to the matrix start position
                         datastream.seek(offset_to_matrix - 4, os.SEEK_CUR) # Already read 4 bytes (vf)

                         matrix_bytes = datastream.read(36)
                         if len(matrix_bytes) < 36:
                              print(f"Error: Could not read the full 36-byte matrix for {atom_type.decode('latin-1', errors='ignore')}")
                              continue # Skip this atom

                         matrix = list(struct.unpack(">9l", matrix_bytes))

                         # Extract rotation from matrix elements (a, b, u, c, d, v, x, y, w)
                         # Rotation (theta) formulas:
                         # a = cos(theta), b = sin(theta)
                         # c = -sin(theta), d = cos(theta)
                         # We use b or c which are based on sin(theta)
                         a = float(matrix[0]) / (1 << 16)
                         b = float(matrix[1]) / (1 << 16)
                         # c = float(matrix[3]) / (1 << 16) # -sin(theta)
                         # d = float(matrix[4]) / (1 << 16) # cos(theta)

                         # Calculate angle from sin (b) or cos (a)
                         # Clamp values to prevent math domain error with asin/acos
                         a_clamped = max(min(a, 1.0), -1.0)
                         b_clamped = max(min(b, 1.0), -1.0)
                         angle_rad_from_sin = math.asin(b_clamped)
                         angle_rad_from_cos = math.acos(a_clamped)

                         # Basic check: sin^2 + cos^2 = 1
                         if not math.isclose(a**2 + b**2, 1.0, abs_tol=0.01):
                            print(f"Warning: Matrix for {atom_type.decode('latin-1', errors='ignore')} doesn't appear to be a simple rotation matrix ({a=}, {b=}).")
                            # Could be scaling or skew involved, angle calculation might be wrong.
                            # Let's try to infer based on common values if possible.
                            if math.isclose(a, 0) and math.isclose(b, 1): deg = 90.0
                            elif math.isclose(a, -1) and math.isclose(b, 0): deg = 180.0
                            elif math.isclose(a, 0) and math.isclose(b, -1): deg = 270.0
                            elif math.isclose(a, 1) and math.isclose(b, 0): deg = 0.0
                            else: deg = None # Cannot determine confidently
                         else:
                             # Use atan2 for better quadrant handling if we use both sin and cos
                             # angle_rad = math.atan2(b, a) # Gives angle relative to positive x-axis
                             # Or use the logic from original script (seems less robust)
                             deg_from_sin = math.degrees(angle_rad_from_sin)
                             deg_from_cos = math.degrees(angle_rad_from_cos)

                             # Simple reconciliation - prefer non-zero? Or check consistency?
                             # Original script logic: `deg = -math.degrees(math.asin(matrix[3])) % 360` -> uses c=-sin(theta)
                             # `-math.degrees(math.asin(c_normalized)) % 360`
                             # Or `if not deg: deg = math.degrees(math.acos(matrix[0]))` -> uses a=cos(theta)

                             # Let's use atan2(sin, cos) = atan2(b, a)
                             deg = math.degrees(math.atan2(b, a)) % 360


                         if deg is not None:
                             # Round to nearest 90 degrees if close, as that's standard for rotation metadata
                             if abs(deg - 90) < 5: deg = 90.0
                             elif abs(deg - 180) < 5: deg = 180.0
                             elif abs(deg - 270) < 5: deg = 270.0
                             elif abs(deg - 0) < 5 or abs(deg - 360) < 5 : deg = 0.0
                             # Only add common rotation values
                             if deg in [0.0, 90.0, 180.0, 270.0]:
                                 degrees.add(int(deg))
                                 print(f"Found potential rotation {int(deg)} from {atom_type.decode('latin-1', errors='ignore')} matrix.")
                         else:
                             print(f"Could not determine valid rotation angle from matrix for {atom_type.decode('latin-1', errors='ignore')}")

                         # Need to skip remaining fields in the atom if any (e.g., width/height in tkhd)
                         # This part is tricky without knowing the exact atom structure differences
                         # Let _find_atoms handle seeking based on atom size for now.

                    else:
                        # If matrix offset wasn't determined (unknown version), we need to skip the atom
                        # _find_atoms should handle this by seeking based on atom size.
                        print(f"Skipping matrix read for {atom_type.decode('latin-1', errors='ignore')} due to unknown version or offset calculation failure.")

                except struct.error as e:
                    print(f"Struct error processing atom {atom_type.decode('latin-1', errors='ignore')} within moov: {e}")
                    # Attempt to continue to the next atom if possible
                    # This requires _find_atoms to correctly seek past the problematic atom
                except Exception as e:
                    print(f"Unexpected error processing atom {atom_type.decode('latin-1', errors='ignore')} within moov: {e}")
                    # Decide whether to break or try continuing

        # Return logic based on findings
        if len(degrees) == 0:
            print("No rotation found in any mvhd/tkhd matrix.")
            return 0 # Assume 0 if none found
        elif len(degrees) == 1:
            found_deg = degrees.pop()
            print(f"Consensus rotation from header: {found_deg}")
            return found_deg
        else:
            # Multiple different rotation values found (e.g., in mvhd vs tkhd)
            # This is unusual. Maybe prioritize tkhd? Or return an error indicator?
            # Let's return the first one found or a common value if possible.
            # For now, return None to indicate ambiguity or error.
            print(f"Warning: Inconsistent rotation values found in header: {degrees}. Returning None.")
            return None

    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return None
    except Exception as e:
        print(f"Error reading rotation from header for {filename}: {e}")
        print(traceback.format_exc())
        return None # Indicate error


# Get video information using ffprobe
def get_video_info(video_path):
    """Get video information using ffprobe, accounting for rotation, with fallback."""
    print(f"Getting video info for: {video_path}")

    # Command to get width, height, and rotation using ffprobe
    info_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height:stream_tags=rotate',
        '-of', 'json',
        video_path
    ]

    result = subprocess.run(info_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = None
    width = 0
    height = 0
    rotation = 0 # Default rotation

    try:
        info = json.loads(result.stdout)
        print(f"ffprobe stream info result: {info}")
        if 'streams' in info and info['streams']:
            stream = info['streams'][0]
            width = int(stream.get('width', 0))
            height = int(stream.get('height', 0))

            # Check for rotation tag from ffprobe
            if 'tags' in stream and 'rotate' in stream['tags']:
                try:
                    rotation = int(stream['tags']['rotate'])
                    print(f"Rotation found via ffprobe: {rotation}")
                except (ValueError, TypeError):
                    print(f"Warning: Could not parse rotation tag value from ffprobe: {stream['tags']['rotate']}")
                    rotation = 0 # Reset to default if parsing fails
            else:
                 print("Rotation tag not found in ffprobe output.")
                 rotation = 0 # Explicitly set to 0 if tag is missing
        else:
             print("Warning: No video streams found in ffprobe output.")
             # Width/Height remain 0

    except json.JSONDecodeError:
        print(f"Error decoding ffprobe JSON output for stream info: {result.stdout.decode()}")
        print(f"ffprobe stderr: {result.stderr.decode()}")
        # Width/Height/Rotation remain 0

    except Exception as e:
         print(f"An unexpected error occurred processing ffprobe stream info: {e}")
         # Width/Height/Rotation remain 0


    # --- Fallback Rotation Check ---
    # If ffprobe didn't provide dimensions or rotation (or failed), try reading header
    # We need width/height *before* applying rotation swap, so we only call this if rotation is still 0
    # And we have valid dimensions from ffprobe OR if ffprobe failed entirely
    if rotation == 0:
         print("Rotation is 0 after ffprobe check. Attempting header read as fallback...")
         try:
             # Ensure the file exists before attempting to read header
             if os.path.exists(video_path):
                 rotation_from_header = _get_rotation_from_header(video_path)
                 if rotation_from_header is not None and rotation_from_header in [90, 180, 270]:
                     rotation = rotation_from_header # Use valid rotation from header
                     print(f"Rotation found from header: {rotation}")
                 elif rotation_from_header == 0:
                      print("Header reading returned 0 rotation.")
                      rotation = 0 # Explicitly keep 0
                 else:
                      print(f"Header reading returned invalid or ambiguous rotation ({rotation_from_header}). Keeping rotation 0.")
                      rotation = 0 # Keep default if header read fails or returns invalid/ambiguous
             else:
                  print(f"Video path {video_path} does not exist, skipping header read fallback.")
         except Exception as e:
             print(f"Error during fallback rotation check: {e}")
             rotation = 0 # Keep default on error


    # Get duration (separate ffprobe call) - do this regardless of rotation success
    duration = 0.0
    duration_cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'json',
        video_path
    ]
    try:
        duration_result = subprocess.run(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60) # Add timeout
        duration_info = json.loads(duration_result.stdout)
        if 'format' in duration_info and 'duration' in duration_info['format']:
            duration_str = duration_info['format']['duration']
            if duration_str is not None:
                try:
                    duration = float(duration_str)
                except (ValueError, TypeError):
                    print(f"Warning: Could not parse duration value: {duration_str}")
                    duration = 0.0
            else:
                print("Warning: Duration field is null.")
                duration = 0.0
        else:
            print("Warning: Duration information not found in format.")
    except json.JSONDecodeError:
        print(f"Error decoding ffprobe JSON output for duration: {duration_result.stdout.decode()}")
        print(f"ffprobe stderr: {duration_result.stderr.decode()}")
    except subprocess.TimeoutExpired:
        print("Timeout getting video duration.")
    except Exception as e:
        print(f"Error getting video duration: {e}")
        # Keep duration as 0.0

    # --- Final Dimension Calculation ---
    # Check if we have valid initial dimensions before proceeding
    if width <= 0 or height <= 0:
         # Attempt to get width/height again if the first ffprobe call failed but duration worked
         # This is a failsafe, but ideally the first call gets dimensions if the file is valid
         if info is None and duration > 0:
             print("Attempting to re-fetch dimensions as initial ffprobe failed but duration succeeded.")
             result = subprocess.run(info_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
             try:
                 info = json.loads(result.stdout)
                 if 'streams' in info and info['streams']:
                     stream = info['streams'][0]
                     width = int(stream.get('width', 0))
                     height = int(stream.get('height', 0))
                     print(f"Re-fetched dimensions: width={width}, height={height}")
                 else:
                      print("Failed to re-fetch dimensions.")
             except Exception as e:
                  print(f"Error re-fetching dimensions: {e}")

         # If still no valid dimensions, return None
         if width <= 0 or height <= 0:
              print(f"Error: Could not determine valid dimensions for: {video_path}. width={width}, height={height}")
              return None


    print(f"Initial dimensions: width={width}, height={height}. Final determined rotation: {rotation}")

    # Store original dimensions before swap
    original_width, original_height = width, height

    # Swap width and height IF rotation requires it
    if rotation in [90, 270]:
        print(f"Applying rotation {rotation}: Swapping width and height.")
        display_width, display_height = height, width
    else:
        display_width, display_height = width, height


    video_info = {
        'width': display_width, # Represents displayed width
        'height': display_height, # Represents displayed height
        'duration': duration,
        'is_portrait': display_height > display_width, # Calculated based on displayed dimensions
        'rotation': rotation, # Store the detected rotation
        'original_width': original_width, # Store original for reference if needed
        'original_height': original_height, # Store original for reference if needed
    }
    print(f"Final video info (adjusted for rotation): {video_info}")
    return video_info

# Extract a random clip from a video
def extract_random_clip(video_path, output_path, clip_duration):
    """Extract a random clip from the video with specified duration."""
    print(f"Extracting random clip from: {video_path} to {output_path} with duration: {clip_duration}")
    
    # Get video info
    video_info = get_video_info(video_path)
    if not video_info:
        print(f"Error: Could not get video information for {video_path}")
        return None
    
    # Ensure clip duration doesn't exceed video duration
    duration = min(clip_duration, video_info['duration'])
    
    # Determine start time
    if video_info['duration'] > duration:
        max_start = video_info['duration'] - duration
        start_time = random.uniform(0, max_start)
    else:
        start_time = 0
    
    print(f"Clip parameters: start_time={start_time}, duration={duration}")
    
    # Extract the clip using ffmpeg
    cmd = [
        'ffmpeg',
        '-y',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-c', 'copy',
        output_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Check if the command was successful
    if result.returncode != 0 or not os.path.exists(output_path):
        stderr = result.stderr.decode('utf-8')
        print(f"Error during clip extraction: {stderr}")
        return None
    
    # Verify the output file is valid and has video stream
    verify_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v',
        '-show_entries', 'stream=codec_type',
        '-of', 'json',
        output_path
    ]
    verify_result = subprocess.run(verify_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if verify_result.returncode != 0:
        stderr = verify_result.stderr.decode('utf-8')
        print(f"Output clip validation failed: {stderr}")
        return None
    
    # Return clip info
    clip_info = {
        'path': output_path,
        'width': video_info['width'],
        'height': video_info['height'],
        'duration': duration,
        'is_portrait': video_info['is_portrait']
    }
    print(f"Extracted clip info: {clip_info}")
    return clip_info

# Resize video to match target resolution
def resize_video(input_path, output_path, target_resolution):
    """
    Resize the video to match the target resolution.
    - Portrait inputs are scaled to fill the target height and cropped if needed.
    - Landscape/Square inputs are scaled to fit within the target and padded.
    """
    print(f"Resizing video: {input_path} to resolution {target_resolution}")
    target_width, target_height = target_resolution
    
    # Get video info
    video_info = get_video_info(input_path)
    if not video_info:
        print(f"Error: Could not get video information for {input_path}")
        return None

    # Ensure width and height are not zero
    if video_info['width'] == 0 or video_info['height'] == 0:
        print(f"Error: Invalid video dimensions for {input_path}: width={video_info['width']}, height={video_info['height']}")
        return None

    # Define ffmpeg base command options
    ffmpeg_opts = [
        'ffmpeg', '-y', '-v', 'error', '-i', input_path,
        '-c:v', 'libx264', '-preset', 'medium', '-pix_fmt', 'yuv420p',
        '-crf', '23', '-r', '30', '-c:a', 'aac', '-strict', 'experimental'
    ]
    
    filter_complex = ""

    if video_info['is_portrait']:
        print(f"Input video {input_path} is portrait.")
        # Check if it already matches the target resolution
        if video_info['width'] == target_width and video_info['height'] == target_height:
            print(f"Skipping resize for {input_path} as it already matches target resolution.")
            try:
                if os.path.abspath(input_path) != os.path.abspath(output_path):
                    shutil.copy(input_path, output_path)
                    print(f"Copied {input_path} to {output_path}")
                else:
                    print(f"Input and output paths are the same ({output_path}), no copy needed.")
                
                if os.path.exists(output_path):
                    return output_path
                else:
                    print(f"Error: Output file {output_path} does not exist after copy/skip.")
                    return None
            except Exception as e:
                print(f"Error copying or accessing file {input_path} to {output_path}: {e}")
                return None
        else:
            # Portrait input, different dimensions: Scale to fill height, crop width
            print(f"Scaling portrait video {input_path} to fill target height and cropping width.")
            filter_complex = f'[0:v]scale=-1:{target_height},crop={target_width}:{target_height}[outv]'
            cmd = ffmpeg_opts + ['-vf', filter_complex.replace('[0:v]', '').replace('[outv]', ''), output_path]
            # Alternate filter using scale and crop directly in -vf
            # filter_vf = f'scale=-1:{target_height},crop={target_width}:{target_height}'
            # cmd = ffmpeg_opts + ['-vf', filter_vf, output_path]


    else: # Landscape or Square input
        print(f"Input video {input_path} is landscape or square. Scaling and padding.")
        # Scale to fit within target dimensions and pad (existing logic)
        width_ratio = target_width / video_info['width']
        height_ratio = target_height / video_info['height']
        scale_factor = min(width_ratio, height_ratio)
        new_width = max(int(video_info['width'] * scale_factor), 2)
        new_height = max(int(video_info['height'] * scale_factor), 2)
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        print(f"Resize parameters: new_width={new_width}, new_height={new_height}, x_offset={x_offset}, y_offset={y_offset}")
        filter_complex = f"[0:v]scale={new_width}:{new_height},pad={target_width}:{target_height}:{x_offset}:{y_offset}:black[outv]"
        cmd = ffmpeg_opts + ['-vf', filter_complex.replace('[0:v]', '').replace('[outv]', ''), output_path]


    # Execute ffmpeg command if filter_complex was set (i.e., not skipped)
    if filter_complex:
        print(f"Executing FFmpeg command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
            
            if result.returncode != 0 or not os.path.exists(output_path):
                stderr = result.stderr.decode('utf-8')
                stdout = result.stdout.decode('utf-8')
                print(f"Error during video processing: {stderr}")
                # print(f"Command output: {stdout}") # Can be verbose
                print(f"Return code: {result.returncode}")
                print(f"Command: {' '.join(cmd)}")
                return None
        except subprocess.TimeoutExpired:
            print(f"Timeout expired during video processing for {input_path}")
            return None
        except Exception as e:
             print(f"An unexpected error occurred running ffmpeg: {e}")
             print(f"Command: {' '.join(cmd)}")
             return None


    # Verify the output file is valid after processing or skipping (if file exists)
    if os.path.exists(output_path):
        verify_cmd = ['ffprobe', '-v', 'error', output_path]
        verify_result = subprocess.run(verify_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if verify_result.returncode != 0:
            stderr = verify_result.stderr.decode('utf-8')
            print(f"Output file validation failed for {output_path}: {stderr}")
            return None
        else:
             print(f"Successfully processed/copied video to {output_path}")
             return output_path # Return path if valid
    else:
        # This case should ideally not be reached if logic is correct
        print(f"Error: Output file {output_path} not found after processing.")
        return None

# Concatenate videos using the concat filter for robust re-encoding
def concatenate_videos(clip_paths, output_path):
    """Concatenate videos using the ffmpeg concat filter to handle potentially incompatible stream formats."""
    print(f"Concatenating {len(clip_paths)} videos to {output_path} with re-encoding.")
    if not clip_paths:
        print("Error: No clips to concatenate")
        return None
    
    if len(clip_paths) == 1:
        # If only one clip, just copy it
        print("Only one clip, copying directly.")
        shutil.copy(clip_paths[0], output_path)
        return output_path
    
    # Prepare inputs for ffmpeg command
    inputs = []
    for clip_path in clip_paths:
        inputs.extend(['-i', clip_path])

    # Prepare the filter_complex string to concatenate video streams
    filter_inputs = "".join([f'[{i}:v:0]' for i in range(len(clip_paths))])
    # a=0 because audio is added separately later
    filter_complex = f"{filter_inputs}concat=n={len(clip_paths)}:v=1:a=0[v]"
    
    # Concatenate videos using ffmpeg's concat filter
    cmd = [
        'ffmpeg',
        '-y',
        *inputs,
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-r', '30',
        '-preset', 'medium',
        '-crf', '23',
        output_path
    ]
    
    print("Executing ffmpeg concat filter...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0 or not os.path.exists(output_path):
        stderr = result.stderr.decode('utf-8')
        print(f"Error during video concatenation: {stderr}")
        return None
    
    print(f"Successfully created concatenated video at {output_path}")
    
    return output_path

# Add audio to video
def add_audio_to_video(video_path, audio_path, output_path, config):
    """Add audio to video with volume adjustment and fade effects."""
    print(f"Adding audio {audio_path} to video {video_path}")
    audio_config = config['audio']
    
    # Get video duration
    video_info = get_video_info(video_path)
    if not video_info:
        print(f"Error: Could not get video information for {video_path}")
        return None
    
    # Create audio filter
    audio_filter = f"volume={audio_config['volume']},afade=t=in:st=0:d={audio_config['fade_in']},afade=t=out:st={video_info['duration'] - audio_config['fade_out']}:d={audio_config['fade_out']}"
    print(f"Audio filter: {audio_filter}")
    
    # Add audio to video using ffmpeg
    cmd = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', audio_path,
        '-filter_complex', f"[1:a]{audio_filter}[a]",
        '-map', '0:v',
        '-map', '[a]',
        '-shortest',
        '-c:v', 'copy',
        '-c:a', 'aac',
        output_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Check if the command was successful
    if result.returncode != 0 or not os.path.exists(output_path):
        stderr = result.stderr.decode('utf-8')
        print(f"Error during audio addition: {stderr}")
        return None
    
    # Verify the output file is valid
    verify_cmd = [
        'ffprobe',
        '-v', 'error',
        output_path
    ]
    verify_result = subprocess.run(verify_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if verify_result.returncode != 0:
        stderr = verify_result.stderr.decode('utf-8')
        print(f"Output file validation failed after adding audio: {stderr}")
        return None
    
    return output_path

# Update request status in DynamoDB
def update_request_status(request_id, status, result=None):
    """Update the status of a montage request in DynamoDB."""
    print(f"Updating request {request_id} to status {status}")
    update_expression = "SET #status = :status, updatedAt = :updatedAt"
    expression_attribute_names = {'#status': 'status'}
    expression_attribute_values = {
        ':status': status,
        ':updatedAt': datetime.now().isoformat()
    }
    
    if result:
        update_expression += ", #result = :result"
        expression_attribute_names['#result'] = 'result'
        expression_attribute_values[':result'] = result

    try:
        requests_table.update_item(
            Key={'pk': 'montage#requests', 'requestId': request_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attribute_names,
            ExpressionAttributeValues=expression_attribute_values
        )
        print(f"Successfully updated request status for {request_id}")
        return True
    except Exception as e:
        print(f"Error updating request status: {e}")
        return False

# Get videos from specific folder in S3 bucket
def get_videos_from_folder(bucket_name, folder_path, num_videos=None):
    """Get videos from the specified folder in the S3 bucket."""
    print(f"Getting videos from folder: {bucket_name}/{folder_path}")
    # Ensure folder path has a trailing slash
    if not folder_path.endswith('/'):
        folder_path += '/'
    
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=folder_path
    )
    
    if 'Contents' not in response:
        raise ValueError(f"No files found in bucket '{bucket_name}' with prefix '{folder_path}'")
    
    video_files = [item['Key'] for item in response['Contents'] 
                  if item['Key'].lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
                  and item['Key'] != folder_path]
    
    if not video_files:
        raise ValueError(f"No video files found in bucket '{bucket_name}' with prefix '{folder_path}'")
    
    print(f"Found {len(video_files)} videos in folder")
    
    if num_videos and num_videos < len(video_files):
        selected_videos = random.sample(video_files, num_videos)
        print(f"Selected {len(selected_videos)} random videos")
        return selected_videos
    return video_files

# Get music from specific folder in S3 bucket
def get_music_from_folder(bucket_name, folder_path):
    """Get a random music file from the specified folder in the S3 bucket."""
    print(f"Getting music from folder: {bucket_name}/{folder_path}")
    # If folder path is empty, use default music folder from config
    if not folder_path:
        config = load_config()
        folder_path = config['s3']['music_prefix']
        print(f"Using default music folder: {folder_path}")
    
    # Ensure folder path has a trailing slash
    if not folder_path.endswith('/'):
        folder_path += '/'
    
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=folder_path
    )
    
    if 'Contents' not in response:
        raise ValueError(f"No files found in bucket '{bucket_name}' with prefix '{folder_path}'")
    
    music_files = [item['Key'] for item in response['Contents'] 
                  if item['Key'].lower().endswith(('.mp3', '.wav', '.ogg'))
                  and item['Key'] != folder_path]
    
    if not music_files:
        raise ValueError(f"No music files found in bucket '{bucket_name}' with prefix '{folder_path}'")
    
    selected_music = random.choice(music_files)
    print(f"Selected music file: {selected_music}")
    return selected_music

# Main function to create video compilation
def create_video_compilation(event, context):
    """Create a video compilation based on the request parameters."""
    print(f"Starting video compilation with event: {event}")
    request_id = event['requestId']
    prompt = event.get('prompt')
    video_folder = event.get('videoFolder') # Kept for legacy requests
    music_folder = event['musicFolder']
    num_clips = int(event['numClips'])
    video_length = event['videoLength']
    clip_duration = event['clipDuration']
    is_music_included = event['isMusicIncluded']
    video_bucket = environ['MONTAGE_VIDEOS_BUCKET']
    music_bucket = environ['MONTAGE_MUSIC_BUCKET']
    output_bucket = environ['MONTAGE_OUTPUT_BUCKET']
    
    if not request_id:
        print("Error: Missing requestId")
        return {
            'statusCode': 400,
            'body': json.dumps({'message': 'Missing requestId'}, cls=DecimalEncoder)
        }
    
    # Check for prompt or video folder
    if not prompt and not video_folder:
        error_message = 'Request must include either a "prompt" or a "videoFolder".'
        print(f"Error: {error_message}")
        update_request_status(request_id, 'FAILED', {'error': error_message})
        return {
            'statusCode': 400,
            'body': json.dumps({'message': error_message}, cls=DecimalEncoder)
        }

    # Update status to PROCESSING before starting
    update_request_status(request_id, 'PROCESSING')
    
    # Create temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp(dir='/tmp')
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        # Load configuration
        config = load_config()
        
        # Calculate number of clips needed based on video_length and clip_duration if provided
        # if video_length and clip_duration and not num_clips:
        #     num_clips = max(1, int(video_length / clip_duration))
        #     print(f"Calculated number of clips: {num_clips} based on video_length: {video_length} and clip_duration: {clip_duration}")
        
        # Get video keys based on prompt or folder
        video_keys = []
        try:
            if prompt:
                # New: Get videos based on AI prompt
                video_keys = get_videos_from_prompt(prompt, num_clips)
            elif video_folder:
                # Legacy: Get videos from a specific folder
                video_keys = get_videos_from_folder(
                    video_bucket, 
                    video_folder, 
                    num_clips
                )
        except Exception as e:
            error_message = f"Error retrieving videos: {str(e)}"
            print(error_message)
            update_request_status(request_id, 'FAILED', {'error': error_message})
            return {
                'statusCode': 500,
                'body': json.dumps({'message': error_message}, cls=DecimalEncoder)
            }

        # Extract clips from videos
        clips = []
        clip_infos = []
        
        print(f"Processing {len(video_keys)} videos")
        for i, video_key in enumerate(video_keys):
            print(f"Processing video {i+1}/{len(video_keys)}: {video_key}")
            local_video_path = os.path.join(temp_dir, f"source_{i}.mp4")
            download_from_s3(video_bucket, video_key, local_video_path)
            
            clip_path = os.path.join(temp_dir, f"clip_{i}.mp4")
            clip_info = extract_random_clip(local_video_path, clip_path, clip_duration)
            
            if clip_info:
                clips.append(clip_path)
                clip_infos.append(clip_info)
        
        if not clips:
            raise ValueError("No valid video clips could be extracted")
        
        # Always use portrait mode for output
        is_output_portrait = True
        print(f"Output orientation: portrait (forced)")
        
        # Get target resolution for portrait mode
        output_resolution = config['video']['output_resolution']
        target_resolution = (
            output_resolution['portrait']['width'],
            output_resolution['portrait']['height']
        )
        print(f"Target resolution: {target_resolution}")
        
        # Resize clips to match target resolution
        resized_clips = []
        
        print(f"Resizing {len(clips)} clips to portrait mode")
        for i, clip_path in enumerate(clips):
            print(f"Resizing clip {i+1}/{len(clips)}")
            resized_path = os.path.join(temp_dir, f"resized_{i}.mp4")
            resized_clip = resize_video(clip_path, resized_path, target_resolution)
            
            if resized_clip:
                resized_clips.append(resized_path)
        
        if not resized_clips:
            raise ValueError("No clips could be resized")
        
        # Concatenate clips without transitions
        combined_video_path = os.path.join(temp_dir, "combined.mp4")
        combined_video = concatenate_videos(resized_clips, combined_video_path)
        
        # Add music if included
        if is_music_included and music_folder:
            print(f"Adding music from folder: {music_folder}")
            try:
                music_key = get_music_from_folder(music_bucket, music_folder)
                local_music_path = os.path.join(temp_dir, "music.mp3")
                download_from_s3(music_bucket, music_key, local_music_path)
                
                # Add music to video
                with_music_path = os.path.join(temp_dir, "with_music.mp4")
                with_music = add_audio_to_video(combined_video_path, local_music_path, with_music_path, config)
                
                # Use the video with music for the next step
                if with_music:
                    combined_video_path = with_music_path
                    print("Successfully added music to video")
            except Exception as e:
                print(f"Error processing audio: {e}")
                # Continue without music if there's an error
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        output_filename = (
            f"{config['video']['output_filename']}_{request_id}_"
            f"portrait_"
            f"{timestamp}_{unique_id}"
            f".{config['video']['output_format']}"
        )
        print(f"Output filename: {output_filename}")
        
        # Upload final video to S3
        output_key = f"{config['s3']['output_prefix']}/{output_filename}"
        s3_url = upload_to_s3(combined_video_path, output_bucket, output_key)
        
        # Generate public URL for the video
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': output_bucket, 'Key': output_key},
            ExpiresIn=config.get('s3', {}).get('url_expiration', 604800)  # Default 7 days
        )
        print(f"Generated presigned URL with expiration: {config.get('s3', {}).get('url_expiration', 604800)} seconds")
        
        result = {
            'bucket': output_bucket,
            'key': output_key,
            's3Url': s3_url,
            'publicUrl': presigned_url,
            'orientation': 'portrait',
            'completedAt': datetime.now().isoformat()
        }
        
        # Update request status to completed
        update_request_status(request_id, 'COMPLETED', result)
        
        print(f"Video compilation created successfully: {s3_url}")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Video compilation completed successfully',
                'requestId': request_id,
                'video': result
            }, cls=DecimalEncoder),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
        }
    
    except Exception as e:
        error_message = f"Error creating video compilation: {str(e)}"
        print(error_message)
        print(traceback.format_exc())
        update_request_status(request_id, 'FAILED', {'error': error_message})
        return {
            'statusCode': 500,
            'body': json.dumps({'message': error_message}, cls=DecimalEncoder)
        }
    
    finally:
        # Clean up temporary directory
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

def lambda_handler(event, context):
    """Lambda handler for the video compiler"""
    print(f"Lambda handler invoked with event: {event}")
    request_id = str(uuid.uuid4())[:8]

    try:
        body = json.loads(event.get('body', '{}'))
        body['requestId'] = request_id

        ts = datetime.now().isoformat()
        initial_item = {
            'pk': 'montage#requests',
            'requestId': request_id,
            'id': request_id,
            'status': 'PENDING',
            'createdAt': ts,
            'updatedAt': ts,
            'prompt': body.get('prompt'),
            'videoFolder': body.get('videoFolder'),
            'musicFolder': body.get('musicFolder'),
            'isMusicIncluded': body.get('isMusicIncluded'),
        }
        
        # Add numeric fields, converting to Decimal for DynamoDB to avoid float inaccuracies
        for key in ['numClips', 'videoLength', 'clipDuration']:
            value = body.get(key)
            if value is not None:
                try:
                    # Use string representation to avoid float precision issues with Decimal
                    initial_item[key] = Decimal(str(value))
                except Exception as e:
                    print(f"Warning: Could not convert '{key}' with value '{value}' to Decimal. Skipping. Error: {e}")

        requests_table.put_item(Item=initial_item)
        print(f"Created montage request {request_id}")

        return create_video_compilation(body, context)

    except Exception as e:
        print(f"Error during initial request processing or compilation: {str(e)}")
        # If we are here, the request might have been created but compilation failed.
        # Update its status to FAILED.
        update_request_status(request_id, 'FAILED', result={'error': f'Error creating video compilation: {str(e)}'})
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': f'Error creating video compilation: {str(e)}'
            }, cls=DecimalEncoder)
        }