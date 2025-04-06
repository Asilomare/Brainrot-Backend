FROM public.ecr.aws/lambda/python:3.11

# Copy function code
COPY video_compiler.py ${LAMBDA_TASK_ROOT}

# Install ffmpeg and ffprobe
COPY ffmpeg-7.0.2-amd64-static/ffmpeg /usr/local/bin/
COPY ffmpeg-7.0.2-amd64-static/ffprobe /usr/local/bin/

RUN chmod 777 /usr/local/bin/ffmpeg /usr/local/bin/ffprobe
# Create requirements.txt if it doesn't exist
RUN touch ${LAMBDA_TASK_ROOT}/requirements.txt
# Install dependencies
RUN pip3 install -r ${LAMBDA_TASK_ROOT}/requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler
CMD [ "video_compiler.lambda_handler" ]