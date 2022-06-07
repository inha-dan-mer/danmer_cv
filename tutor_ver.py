# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 10:43:16 2022

@author: ksh76
"""

import urllib3
import requests
import io
import urllib.request
from requests import get
import sys
import os
import numpy as np
import torch
import boto3
import json

def s3_connection():
    try:
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2", # 자신이 설정한 bucket region
            aws_access_key_id = "{비공개}",
            aws_secret_access_key = "{비공개}",
        )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!")
        return s3


def s3_put_object(s3, bucket, filepath, access_key):
    """
    s3 bucket에 지정 파일 업로드
    :param s3: 연결된 s3 객체(boto3 client)
    :param bucket: 버킷명
    :param filepath: 파일 위치
    :param access_key: 저장 파일명
    :return: 성공 시 True, 실패 시 False 반환
    """  
    try:
        s3.upload_file(
            Filename=filepath,
            Bucket=bucket,
            Key=access_key,
            # ExtraArgs={"ContentType": "image/jpg", "ACL": "public-read"},
        )
    except Exception as e:
        return False
    return access_key

def s3_get_image_url(s3, filename):
    """
    s3 : 연결된 s3 객체(boto3 client)
    filename : s3에 저장된 파일 명
    """
    location = s3.get_bucket_location(Bucket='danmer-videos')["LocationConstraint"]
    return f"https://danmer-videos.s3.{location}.amazonaws.com/{filename}"

def download(url, file_name_):
    # open in binary mode
    file_name = file_name_ + 'tutor' + '.mp4'
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        response.raise_for_status()

        # write to file
        file.write(response.content)

    return file_name    


def VIBE(file_name):
    os.system("python demo.py --vid_file " + file_name + " --output_folder output/  --no_render")
    
if __name__ == '__main__':
    file_name = download(sys.argv[1], sys.argv[2]) # tutor s3 url/tutorid
    print(sys.argv[2])
    VIBE(file_name) 
    
    # TO DO:
    # npy 파일 s3에 저장
    s3 = s3_connection()
    title = s3_put_object(s3, 'danmer-videos', 'save_coordin.npy', "tutor_vibe/" + sys.argv[2] + "_tutor.npy")    
    urls = s3_get_image_url(s3, title)
    print(urls)
    # 백으로 s3 url tutor id sys.argv[2] 값 전달
    
    headers = {'Content-Type': 'application/json; charset=utf-8'}


    datas = {'coordinate_url': urls,'tutor_id': sys.argv[2]}
    response = requests.post('http://52.79.67.201:8000/deep/tutor', data = json.dumps(datas), headers = headers)

    print(response.status_code)
    
    

    
    