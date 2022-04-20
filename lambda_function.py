import json
import urllib.parse
import boto3
import urllib3
import datetime
import email
import string
import os
import sys
import numpy as np
from hashlib import md5
from botocore.exceptions import ClientError

from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences

REGION = "us-east-1"
headers = { "Content-Type": "application/json" }
s3_resource = boto3.resource('s3')
ses_client = boto3.client('ses', region_name=REGION)
vocabulary_length = 9013



def send_email(email_destination, email_recieve_date, email_subject, email_body, classification, classification_confidence_score):
    CHARSET = "UTF-8"
    
    email_content = f""""We received your email sent at {email_recieve_date} with the subject {email_subject}. 
                      Here is a 240 character sample of the email body: 
                      {email_body}
                      
                      The email was categorized as {classification} with a 
                      {classification_confidence_score}% confidence."""
                      
    response = ses_client.send_email(
        Destination={
            "ToAddresses": [
                email_destination,
            ],
        },
        Message={
            "Body": {
                "Html": {
                    "Charset": CHARSET,
                    "Data": email_content,
                }
            },
            "Subject": {
                "Charset": CHARSET,
                "Data": "Spam Classification",
            },
        },
        Source="ooctavius2022@gmail.com",
    )
    
    return response

def lambda_handler(event, context):
    # Get the object from the event and show its content type
    # print("EVENT")
    # print(event)
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    try:
        s3_object = s3_resource.Object(bucket, key)
        # get email content from email body
        body = s3_object.get()["Body"].read() 
        email_msg = email.message_from_bytes(body)
        # print("EMAIL CONTENT")
        # print(email_msg)
        
    except Exception as e:
        print(e)
        raise e
    
    # SageMaker information
    ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
    runtime = boto3.Session().client(service_name='sagemaker-runtime', region_name='us-east-1')
    
    parsed_body = ""

    if email_msg.is_multipart():
        for part in email_msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))
    
            # ignore text/plain (txt) attachments
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                parsed_body = part.get_payload()
                
    else:
        parsed_body = email_msg.get_payload()
    
    # print("PARSED BODY")
    # print(parsed_body)
    
    # Reference used for multipart parsing:
    # https://stackoverflow.com/questions/17874360/python-how-to-parse-the-body-from-a-raw-email-given-that-raw-email-does-not
    
    # trim new lines
    trimmed_body = [parsed_body.replace('\n', ' ').replace('\r', ' ')]
    print("TRIMMED BODY")
    print(trimmed_body)
    
    one_hot_email_msg = one_hot_encode(trimmed_body, vocabulary_length)
    vectorized_msg = vectorize_sequences(one_hot_email_msg, vocabulary_length)
    
    encoded_test_messages_np = np.array(vectorized_msg)[0]
    print(encoded_test_messages_np)
    print(np.where(encoded_test_messages_np > 0)[0])

    payload = json.dumps(vectorized_msg.tolist())
    # get runtime response
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=payload)
    
    result = json.loads(response["Body"].read().decode('utf-8'))
    print("RESULT")
    print(result)
    
    if result['predicted_label'][0][0] == 1.0:
        classification = "SPAM"
    else:
        classification = "HAM"
    
    print("CLASSIFICATION IS:")
    print(classification)
    
    reply_to = email_msg['Return-Path'].strip("<>")
    sent_date = email_msg['date']
    email_subject = email_msg['subject']
    confidence_score = round(result['predicted_probability'][0][0]*100, 2) if result['predicted_label'][0][0] == 1.0 else round((1-result['predicted_probability'][0][0])*100, 2)
    
    #send email to sender with all required information
    send_email(reply_to, sent_date, email_subject, trimmed_body, classification, confidence_score)
    
