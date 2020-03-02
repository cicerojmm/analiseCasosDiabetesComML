import numpy as np
import pickle
import boto3
import json
import sklearn

def lambda_handler(event, context):
    #permite acesso pela configuração de permissão que realizamos
    s3 = boto3.client('s3')
    
    #nome do bucket e arquivo que serão carregados do S3
    bucket_name = 'modelo-producao-diabetes'
    object_name = 'modelo.pkl'
    
    #carregando o arquivo do modelo em memória
    response = s3.get_object(Bucket=bucket_name, Key=object_name)
    model = pickle.loads(response['Body'].read())

    #recebendo os parâmetros passados pelo cliente por POST
    data = json.loads(event['body'])
    
    #realizando a predição para os dados informados pelo cliente
    prediction = model.predict(np.array([list(data.values())]))
    output = prediction[0]
    resposta = {'DIABETES': int(output)}
    
    #retornando a respota para o cliente
    return {
        'statusCode': 200,
        'body': json.dumps(resposta)
    }
