from aws_cdk import (
    # Duration,
    Stack,
    aws_s3 as s3,
    Duration,
    # aws_sqs as sqs,
    aws_lambda as _lambda,
    aws_apigateway as apigw,
    aws_dynamodb as dynamodb,
)
from constructs import Construct

class ApiStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        table = dynamodb.Table(
            self,
            'MontageRequestsTable',
            partition_key=dynamodb.Attribute(
                name='pk',
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name='ts',
                type=dynamodb.AttributeType.STRING
            ),
            # billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST
        )

        video_bucket = s3.Bucket.from_bucket_name(self, 'MontageVideosBucket', f'{self.account}-video-uploads')
        music_bucket = s3.Bucket.from_bucket_name(self, 'MontageMusicBucket',  f'{self.account}-music-uploads')
        output_bucket = s3.Bucket.from_bucket_name(self, 'MontageOutputBucket', f'{self.account}-montage-uploads')
        #    Lambda processor
        lambda_video_compiler = _lambda.DockerImageFunction(self, 'LambdaVideoCompiler',
            code=_lambda.DockerImageCode.from_image_asset('lambdas/brainrot/'),
            environment={
                'MONTAGE_REQUESTS_TABLE': table.table_name,
                'MONTAGE_VIDEOS_BUCKET': video_bucket.bucket_name,
                'MONTAGE_MUSIC_BUCKET': music_bucket.bucket_name,
                'MONTAGE_OUTPUT_BUCKET': output_bucket.bucket_name
            },
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
