import json
import boto3
import os
from datetime import datetime
from boto3.dynamodb.conditions import Key

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
requests_table = dynamodb.Table(os.environ['MONTAGE_REQUESTS_TABLE'])

def get_montage_requests(event, context):
    """
    Get all montage requests or a specific montage request by ID.
    This is an API Gateway Lambda proxy handler.
    """
    try:
        # Check if this is a GET request for a specific request
        path_parameters = event.get('pathParameters', {})
        request_id = path_parameters.get('id') if path_parameters else None
        
        if request_id:
            # Get a specific request by ID
            response = requests_table.get_item(Key={'id': request_id})
            
            if 'Item' not in response:
                return {
                    'statusCode': 404,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'message': f'Montage request with ID {request_id} not found'
                    })
                }
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'request': response['Item']
                })
            }
        else:
            # Query parameters for filtering
            query_params = event.get('queryStringParameters', {}) or {}
            status_filter = query_params.get('status')
            
            # Query by partition key and sort key (descending order)
            query_params = {
                'KeyConditionExpression': Key('pk').eq('montage#requests'),
                'ScanIndexForward': False  # Sort in descending order (most recent first)
            }
            
            # Add status filter if provided
            if status_filter:
                query_params['FilterExpression'] = Key('status').eq(status_filter)
            
            # Execute the query
            response = requests_table.query(**query_params)
            
            requests = response.get('Items', [])
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'requests': requests
                })
            }
    
    except Exception as e:
        print(f"Error getting montage requests: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': f'Error getting montage requests: {str(e)}'
            })
        }

def get_montage_by_status(event, context):
    """
    Get montage requests filtered by status.
    This is a helper function that can be used directly.
    """
    try:
        status = event.get('status', 'COMPLETED')  # Default to completed montages
        
        # Query by partition key and sort key (descending order)
        query_params = {
            'KeyConditionExpression': Key('pk').eq('montage#requests'),
            'FilterExpression': Key('status').eq(status),
            'ScanIndexForward': False  # Sort in descending order (most recent first)
        }
        
        # Execute the query
        response = requests_table.query(**query_params)
        
        requests = response.get('Items', [])
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'requests': requests
            })
        }
    
    except Exception as e:
        print(f"Error getting montage requests by status: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': f'Error getting montage requests by status: {str(e)}'
            })
        }

def lambda_handler(event, context):
    """
    Main entry point for the Lambda function.
    Routes to the appropriate handler based on the path.
    """
    # Get the HTTP method and resource path from the event
    http_method = event.get('httpMethod')
    resource = event.get('resource', '')
    
    if http_method == 'GET':
        return get_montage_requests(event, context)
    elif resource == '/montage/status' and http_method == 'POST':
        # Parse the request body for the status parameter
        body = json.loads(event.get('body', '{}'))
        event['status'] = body.get('status', 'COMPLETED')
        return get_montage_by_status(event, context)
    else:
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': 'Unsupported method or resource'
            })
        } 