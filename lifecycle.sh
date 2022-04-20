#!/bin/bash

set -e

ENVIRONMENT=mxnet_p37
NOTEBOOK=/home/ec2-user/SageMaker/smlambdaworkshop/training/sms_spam_classifier_mxnet.ipynb

export PATH="/home/ec2-user/anaconda3/bin:$PATH"
conda info --envs


ls /home/ec2-user/SageMaker/smlambdaworkshop/training/

# echo '{ "floatx": "float32", "epsilon": 1e-07, "backend": "mxnet", "image_data_format": "channels_last" }' | sudo tee -a /root/.keras/keras_mxnet.json
source /home/ec2-user/anaconda3/bin/activate "$ENVIRONMENT" && pip install mxnet && pip install --upgrade sagemaker

# nohup jupyter nbconvert "$NOTEBOOK_FILE" --to notebook --ExecutePreprocessor.kernel_name=python3 --execute &

jupyter nbconvert  --to notebook --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 --execute "$NOTEBOOK"

# jupyter nbconvert "$NOTEBOOK" --ExecutePreprocessor.kernel_name=python3 --execute

source /home/ec2-user/anaconda3/bin/deactivate

IDLE_TIME=600

echo "Fetching the autostop script"

wget https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples/master/scripts/auto-stop-idle/autostop.py

echo "Starting the SageMaker autostop script in cron"
(crontab -l 2>/dev/null; echo "*/1 * * * * /usr/bin/python $PWD/autostop.py --time $IDLE_TIME --ignore-connections") | crontab -


