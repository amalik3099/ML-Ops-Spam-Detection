import boto3
import logging
def lambda_handler(event, context):
    client = boto3.client('sagemaker')
    client.stop_notebook_instance(NotebookInstanceName='SageMaker-spam-detector')
    print("Notebook Started!")
    return 0