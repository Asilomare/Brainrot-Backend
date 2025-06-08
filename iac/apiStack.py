from aws_cdk import (
    # Duration,
    Stack,
    aws_s3 as s3,
    Duration,
    aws_lambda_event_sources as lambda_event_sources,
    # aws_sqs as sqs,
    aws_lambda as _lambda,
    aws_apigateway as apigw,
    aws_dynamodb as dynamodb,
    aws_secretsmanager as secretsmanager,
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
        
        # Lambda for video analysis
        lambda_video_analyzer = _lambda.DockerImageFunction(self, 'LambdaVideoAnalyzer',
            code=_lambda.DockerImageCode.from_image_asset('lambdas/video_analyzer/'),
            environment={
                'AI_KEYS_SECRET_ARN': ai_keys_secret.secret_arn,
                'PINECONE_API_SECRET_ARN': pinecone_api_secret.secret_arn,
                'PINECONE_ENVIRONMENT': 'us-east-1',
                'PINECONE_INDEX_NAME': 'video-search',
            },
            memory_size=1024, # Needs more memory for video processing
            timeout=Duration.minutes(5) # And more time
        )
        ai_keys_secret.grant_read(lambda_video_analyzer)
        pinecone_api_secret.grant_read(lambda_video_analyzer)
        video_bucket.grant_read(lambda_video_analyzer)

        lambda_video_analyzer.add_event_source(
            lambda_event_sources.S3EventSource(video_bucket,
                events=[s3.EventType.OBJECT_CREATED]
            )
        )

        #    Lambda processor
        lambda_video_compiler = _lambda.DockerImageFunction(self, 'LambdaVideoCompiler',
            code=_lambda.DockerImageCode.from_image_asset('lambdas/brainrot/'),
            environment={
                'MONTAGE_REQUESTS_TABLE': table.table_name,
                'MONTAGE_VIDEOS_BUCKET': video_bucket.bucket_name,
                'MONTAGE_MUSIC_BUCKET': music_bucket.bucket_name,
                'MONTAGE_OUTPUT_BUCKET': output_bucket.bucket_name
            },
            memory_size=512,
            timeout=Duration.minutes(3)
        )
        table.grant_read_write_data(lambda_video_compiler)
        video_bucket.grant_read_write(lambda_video_compiler)
        music_bucket.grant_read_write(lambda_video_compiler)
        output_bucket.grant_read_write(lambda_video_compiler)

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
