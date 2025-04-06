FROM public.ecr.aws/lambda/python:3.11

RUN yum -y install wget tar

# Copy function code
COPY video_compiler.py ${LAMBDA_TASK_ROOT}

# Install ffmpeg and ffprobe
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
RUN tar -xf ffmpeg-release-amd64-static.tar.xz
RUN cp ffmpeg-release-amd64-static/ffprobe /usr/local/bin
RUN chmod +x /usr/local/bin/ffprobe
RUN cp ffmpeg-release-amd64-static/ffmpeg /usr/local/bin/ffmpeg
RUN chmod +x /usr/local/bin/ffmpeg

# Create requirements.txt if it doesn't exist
RUN touch ${LAMBDA_TASK_ROOT}/requirements.txt
# Install dependencies
RUN pip3 install -r ${LAMBDA_TASK_ROOT}/requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler
CMD [ "video_compiler.lambda_handler" ]