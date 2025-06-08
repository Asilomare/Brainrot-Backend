FROM public.ecr.aws/lambda/python:3.11

# Copy function code and config
COPY *.py /var/task/
COPY config.json /var/task/
COPY requirements.txt /var/task/

# Install dependencies
RUN python3.11 -m pip install -r requirements.txt -t /var/task

# Set the CMD to your handler
CMD ["video_compiler.lambda_handler"]