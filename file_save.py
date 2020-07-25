from PIL import Image
import numpy as np
import os
import json
import io
import requests
import cv2

def getImages(url):
    data = {'roleId': 1, 'condition': '', 'isHandle': -1, 'isValid': -1, 'isFever': -1, 'dateStart': '2020-06-16', 'dateEnd': '2020-06-18',
            'timeSort': 1, 'tempSort': -1, 'pageSize': 1, 'pageNum': 2}
    headers = {'content-type': "application/json"}
    req = requests.post(url=url, data=json.dumps(data),headers=headers)
    imgs_data = json.loads(req.content)
    print(imgs_data)




if __name__ == '__main__':
    url = 'https://infrared.hzncc.cn/api/infrared/alarm/list/spec/deviceId=1051'
    imgs = getImages(url)