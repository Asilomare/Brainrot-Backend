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

# Initialize S3 client
s3_client = boto3.client('s3')

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
requests_table = dynamodb.Table(environ['MONTAGE_REQUESTS_TABLE'])

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

# Get video information using ffprobe
def get_video_info(video_path):
    """Get video information using ffprobe."""
    print(f"Getting video info for: {video_path}")
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-select_streams', 'v:0', 
        '-show_entries', 'stream=width,height,duration', 
        '-of', 'json', 
        video_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = json.loads(result.stdout)
    
    if 'streams' in info and info['streams']:
        stream = info['streams'][0]
        width = int(stream.get('width', 0))
        height = int(stream.get('height', 0))
        
        # Get duration
        duration_cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'json', 
            video_path
        ]
        duration_result = subprocess.run(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        duration_info = json.loads(duration_result.stdout)
        duration = float(duration_info['format']['duration']) if 'format' in duration_info else 0
        
        video_info = {
            'width': width,
            'height': height,
            'duration': duration,
            'is_portrait': height > width
        }
        print(f"Video info: {video_info}")
        return video_info
    
    print(f"Failed to get video info for: {video_path}")
    return None

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
    """Resize the video to match the target resolution while maintaining aspect ratio."""
    print(f"Resizing video: {input_path} to resolution {target_resolution}")
    target_width, target_height = target_resolution
    
    # Get video info
    video_info = get_video_info(input_path)
    if not video_info:
        print(f"Error: Could not get video information for {input_path}")
        return None
    
    # Calculate scaling factor
    width_ratio = target_width / video_info['width']
    height_ratio = target_height / video_info['height']
    
    # Use the smaller ratio to ensure the video fits within the target resolution
    scale_factor = min(width_ratio, height_ratio)
    
    # Calculate new dimensions
    new_width = max(int(video_info['width'] * scale_factor), 2)
    new_height = max(int(video_info['height'] * scale_factor), 2)
    
    # Calculate padding
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    print(f"Resize parameters: new_width={new_width}, new_height={new_height}, x_offset={x_offset}, y_offset={y_offset}")
    
    # Resize and pad the video using ffmpeg - updated command with more options
    cmd = [
        'ffmpeg',
        '-y',
        '-v', 'verbose',
        '-i', input_path,
        '-vf', f'scale={new_width}:{new_height},pad={target_width}:{target_height}:{x_offset}:{y_offset}:black',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        '-r', '30',
        '-c:a', 'aac',
        '-strict', 'experimental',
        output_path
    ]
    
    # Add timeout to prevent endless hanging
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        
        # Check if the command was successful
        if result.returncode != 0 or not os.path.exists(output_path):
            stderr = result.stderr.decode('utf-8')
            stdout = result.stdout.decode('utf-8')
            print(f"Error during video resizing: {stderr}")
            print(f"Command output: {stdout}")
            print(f"Return code: {result.returncode}")
            print(f"Command: {' '.join(cmd)}")
            return None
    except subprocess.TimeoutExpired:
        print(f"Timeout expired during video resizing for {input_path}")
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
        print(f"Output file validation failed: {stderr}")
        return None
    
    return output_path

# Concatenate videos without transitions
def concatenate_videos(clip_paths, output_path):
    """Concatenate videos without transitions."""
    print(f"Concatenating {len(clip_paths)} videos to {output_path}")
    if len(clip_paths) < 1:
        print("Error: No clips to concatenate")
        return None
    
    if len(clip_paths) == 1:
        # If only one clip, just copy it
        print("Only one clip, copying directly")
        shutil.copy(clip_paths[0], output_path)
        return output_path
    
    # Create a temporary file for the concat list
    concat_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    concat_path = concat_file.name
    concat_file.close()
    
    # Write the concat file
    with open(concat_path, 'w') as f:
        for clip_path in clip_paths:
            print(clip_path)
            # filename = os.path.basename(clip_path)
            f.write(f"file '{clip_path}'\n")
    
    print(f"Created concat file at {concat_path}")
    
    # Concatenate videos using ffmpeg
    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_path,
        '-c', 'copy',
        output_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Clean up
    os.unlink(concat_path)

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
def update_request_status(request_id, status, result=None, video_folder=None):
    """Update the status of a montage request in DynamoDB."""
    print(f"Updating request status: {request_id} to {status}")
    update_expression = "SET #status = :status, updatedAt = :updatedAt"
    expression_attribute_names = {
        '#status': 'status'
    }
    expression_attribute_values = {
        ':status': status,
        ':updatedAt': datetime.now().isoformat()
    }
    
    if result:
        update_expression += ", #result = :result"
        expression_attribute_names['#result'] = 'result'
        expression_attribute_values[':result'] = result
    if video_folder:
        update_expression += ", #videoFolder = :videoFolder"
        expression_attribute_names['#videoFolder'] = 'videoFolder'
        expression_attribute_values[':videoFolder'] = video_folder
    try:
        requests_table.update_item(
            Key={'pk': 'montage#requests', 'ts': datetime.now().isoformat() + '#' + request_id},
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
    video_folder = event['videoFolder']
    music_folder = event['musicFolder']
    num_clips = event['numClips']
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
            'body': json.dumps({'message': 'Missing requestId'})
        }
    
    if not video_folder:
        print("Error: Missing videoFolder")
        update_request_status(request_id, 'FAILED', {'error': 'Missing videoFolder'})
        return {
            'statusCode': 400,
            'body': json.dumps({'message': 'Missing videoFolder'})
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
        
        # Get videos from the specified folder
        try:
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
                'body': json.dumps({'message': error_message})
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
        
        # Determine if most clips are portrait or landscape
        portrait_count = sum(1 for info in clip_infos if info['is_portrait'])
        landscape_count = len(clip_infos) - portrait_count
        
        is_output_portrait = portrait_count > landscape_count
        print(f"Output orientation: {'portrait' if is_output_portrait else 'landscape'} (portrait: {portrait_count}, landscape: {landscape_count})")
        
        # Get target resolution
        output_resolution = config['video']['output_resolution']
        target_resolution = (
            output_resolution['portrait' if is_output_portrait else 'landscape']['width'],
            output_resolution['portrait' if is_output_portrait else 'landscape']['height']
        )
        print(f"Target resolution: {target_resolution}")
        
        # Resize clips to match target resolution
        resized_clips = []
        
        print(f"Resizing {len(clips)} clips")
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
            f"{'portrait' if is_output_portrait else 'landscape'}_"
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
            'orientation': 'portrait' if is_output_portrait else 'landscape',
            'completedAt': datetime.now().isoformat()
        }
        
        # Update request status to completed
        update_request_status(request_id, 'COMPLETED', result, video_folder)
        
        print(f"Video compilation created successfully: {s3_url}")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Video compilation completed successfully',
                'requestId': request_id,
                'video': result
            }),
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
            'body': json.dumps({'message': error_message})
        }
    
    finally:
        # Clean up temporary directory
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

def lambda_handler(event, context):
    """Lambda handler for the video compiler"""
    print(f"Lambda handler invoked with event: {event}")
    # If the event has a requestId, assume it's a compilation request
    try:
        uuid_id = str(uuid.uuid4())[:8]
        print(f"Generated request ID: {uuid_id}")
        body = json.loads(event['body'])
        body['requestId'] = uuid_id
        return create_video_compilation(body, context)
    
    except Exception as e:
        print(f"Error creating video compilation: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': f'Error creating video compilation: {str(e)}'
            })
        }