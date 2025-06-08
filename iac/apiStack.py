from aws_cdk import (
    # Duration,
    Stack,
    aws_s3 as s3,
    Duration,
    aws_lambda_event_sources as lambda_event_sources,
    aws_sqs as sqs,
    aws_lambda as _lambda,
    aws_apigateway as apigw,
    aws_dynamodb as dynamodb,
    aws_secretsmanager as secretsmanager,
    aws_s3_notifications as s3n,
    BundlingOptions,
)
from constructs import Construct

class ApiStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        environment = self.node.try_get_context('env')
        ai_keys_secret_arn = environment.get('AI_KEYS_SECRET_ARN')

        ai_keys_secret = secretsmanager.Secret.from_secret_complete_arn(self, 'AiKeysSecret', 
            secret_complete_arn=ai_keys_secret_arn
        )

        pinecone_api_secret_arn = environment.get('PINECONE_API_SECRET_ARN')
        pinecone_api_secret = secretsmanager.Secret.from_secret_complete_arn(self, 'PineconeApiSecret',
            secret_complete_arn=pinecone_api_secret_arn
        )

        table = dynamodb.Table(
            self,
            'MontageRequestsTable',
            partition_key=dynamodb.Attribute(
                name='pk',
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name='requestId',
                type=dynamodb.AttributeType.STRING
            ),
            # billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST
        )

        table.add_global_secondary_index(
            index_name='by-creation-date',
            partition_key=dynamodb.Attribute(name='pk', type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name='createdAt', type=dynamodb.AttributeType.STRING),
        )

        video_bucket = s3.Bucket.from_bucket_name(self, 'MontageVideosBucket', f'{self.account}-video-uploads')
        music_bucket = s3.Bucket.from_bucket_name(self, 'MontageMusicBucket',  f'{self.account}-music-uploads')
        output_bucket = s3.Bucket.from_bucket_name(self, 'MontageOutputBucket', f'{self.account}-montage-uploads')

        queue = sqs.Queue(self, 'InboundQueue',
            visibility_timeout=Duration.hours(12),
            retention_period=Duration.days(4),
        )

        video_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED, 
            s3n.SqsDestination(queue)
        )
        

        # Layer for the video analyzer
        # video_analyzer_layer = _lambda.LayerVersion(self, 'VideoAnalyzerLayer',
        #     code=_lambda.Code.from_asset('layers/video_analyzer/',
        #         bundling=BundlingOptions(
        #             image=_lambda.Runtime.PYTHON_3_11.bundling_image,
        #             command=[
        #                 "bash", "-c",
        #                 "pip install -r requirements.txt -t /asset-output/python && cp -au . /asset-output/python"
        #             ]
        #         )
        #     ),
        #     compatible_runtimes=[_lambda.Runtime.PYTHON_3_11],
        #     description='Layer with dependencies for video analysis lambda'
        # )

        # Add an existing layer from an ARN (e.g., for OpenCV)
        # opencv_layer = _lambda.LayerVersion.from_layer_version_arn(
        #     self, 'OpenCVLayer',
        #     layer_version_arn='arn:aws:lambda:us-east-1:770693421928:layer:Klayers-python38-opencv-python-headless:11'
        # )

        # requests_layer = _lambda.LayerVersion.from_layer_version_arn(
        #     self, 'RequestsLayer',
        #     layer_version_arn='arn:aws:lambda:us-east-1:770693421928:layer:Klayers-p38-requests:18'
        # )

        # pillow_layer = _lambda.LayerVersion.from_layer_version_arn(
        #     self, 'PillowLayer',
        #     layer_version_arn='arn:aws:lambda:us-east-1:770693421928:layer:Klayers-p38-pillow:1'
        # )

        # numpy_layer = _lambda.LayerVersion.from_layer_version_arn(
        #     self, 'NumpyLayer',
        #     layer_version_arn='arn:aws:lambda:us-east-1:770693421928:layer:Klayers-p38-numpy:13'
        # )

        # pinecone_layer = _lambda.LayerVersion.from_layer_version_arn(
        #     self, 'PineconeLayer',
        #     layer_version_arn='arn:aws:lambda:us-east-1:704250834382:layer:pineconeLayer:1'
        # )

        # Lambda for video analysis
        lambda_video_analyzer = _lambda.DockerImageFunction(self, 'LambdaVideoAnalyzer',
            # runtime=_lambda.Runtime.PYTHON_3_8,
            # handler='lambda_function.lambda_handler',
            code=_lambda.DockerImageCode.from_image_asset('lambdas/video_analyzer/'),
            environment={
                'AI_KEYS_SECRET_ARN': ai_keys_secret.secret_arn,
                'PINECONE_API_SECRET_ARN': pinecone_api_secret.secret_arn,
                'PINECONE_ENVIRONMENT': environment.get('PINECONE_ENVIRONMENT'),
                'PINECONE_INDEX_NAME': environment.get('PINECONE_INDEX_NAME'),
            },
            memory_size=1024,
            timeout=Duration.minutes(5),
            # layers=[
            #     # video_analyzer_layer, 
            #     opencv_layer, 
            #     requests_layer, 
            #     pillow_layer, 
            #     numpy_layer, 
            #     pinecone_layer
            # ]
        )
        ai_keys_secret.grant_read(lambda_video_analyzer)
        pinecone_api_secret.grant_read(lambda_video_analyzer)
        video_bucket.grant_read(lambda_video_analyzer)

        lambda_video_analyzer.add_event_source(lambda_event_sources.SqsEventSource(queue,
            # throttle events
            batch_size=1,
            max_concurrency=2,
            max_batching_window=Duration.seconds(10)
        ))

        #    Lambda processor
        lambda_video_compiler = _lambda.DockerImageFunction(self, 'LambdaVideoCompiler',
            code=_lambda.DockerImageCode.from_image_asset('lambdas/brainrot/'),
            environment={
                'MONTAGE_REQUESTS_TABLE': table.table_name,
                'MONTAGE_VIDEOS_BUCKET': video_bucket.bucket_name,
                'MONTAGE_MUSIC_BUCKET': music_bucket.bucket_name,
                'MONTAGE_OUTPUT_BUCKET': output_bucket.bucket_name,
                'AI_KEYS_SECRET_ARN': ai_keys_secret.secret_arn,
                'PINECONE_API_SECRET_ARN': pinecone_api_secret.secret_arn,
                'PINECONE_ENVIRONMENT': environment.get('PINECONE_ENVIRONMENT'),
                'PINECONE_INDEX_NAME': environment.get('PINECONE_INDEX_NAME'),
            },
            memory_size=2048,
            timeout=Duration.minutes(5)
        )
        table.grant_read_write_data(lambda_video_compiler)
        video_bucket.grant_read_write(lambda_video_compiler)
        music_bucket.grant_read_write(lambda_video_compiler)
        output_bucket.grant_read_write(lambda_video_compiler)
        ai_keys_secret.grant_read(lambda_video_compiler)
        pinecone_api_secret.grant_read(lambda_video_compiler)

        lambda_requests_retriever = _lambda.Function(self, 'LambdaRequestsRetriever',
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler='lambda_function.lambda_handler',
            code=_lambda.Code.from_asset('lambdas/retrieve_montages/'),
            environment={
                'MONTAGE_REQUESTS_TABLE': table.table_name,
                'MONTAGE_VIDEOS_BUCKET': video_bucket.bucket_name,
                'MONTAGE_MUSIC_BUCKET': music_bucket.bucket_name,
                'MONTAGE_OUTPUT_BUCKET': output_bucket.bucket_name
            }
        )
        table.grant_read_data(lambda_requests_retriever)
        video_bucket.grant_read_write(lambda_requests_retriever)
        music_bucket.grant_read_write(lambda_requests_retriever)
        output_bucket.grant_read_write(lambda_requests_retriever)

        #    API Gateway
        api = apigw.LambdaRestApi(self, 'Api',
            handler=lambda_video_compiler,
            # proxy=False
            default_cors_preflight_options=apigw.CorsOptions(
                allow_headers=["*"],
                allow_methods=["*"],
                allow_origins=["*"],
            ),
        )

        montage_resource = api.root.add_resource('montage')

        montage_resource.add_method('POST', apigw.LambdaIntegration(lambda_video_compiler))
        montage_resource.add_method('GET', apigw.LambdaIntegration(lambda_requests_retriever))
        
        montage_item_resource = montage_resource.add_resource('{id}')
        montage_item_resource.add_method('GET', apigw.LambdaIntegration(lambda_requests_retriever))

        # New resource for prompt-based generation
        prompt_generation_resource = montage_resource.add_resource('generate-from-prompt')
        prompt_generation_resource.add_method('POST', apigw.LambdaIntegration(lambda_video_compiler))
