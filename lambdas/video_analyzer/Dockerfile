FROM public.ecr.aws/lambda/python:3.11

COPY lambda_function.py requirements.txt ./

RUN pip3 install -r requirements.txt

# Command can be overwritten by providing a different command in the template directly.
CMD ["lambda_function.lambda_handler"] 