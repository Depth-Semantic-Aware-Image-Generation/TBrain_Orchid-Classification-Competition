import numpy as np
from urllib.parse import urlparse, parse_qs
import requests
import re
import os
import json

###############################################################
#img_seed = "C:/flower_category/以圖搜圖/" #替换成自己的图像文件夹
###############################################################
for j in range(1,219):
    img_seed = "C:/flower_category/category/" + str(j) + '/'
    img_ls = os.listdir(img_seed)
    for img in img_ls:
        if('.jpg' not in img):
            continue
        img_path = img_seed+img
        data = {
            'image': open(img_path, 'rb')
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'
        }
        r = requests.post(
            'https://graph.baidu.com/upload?tn=pc&from=pc&image_source=PC_UPLOAD_IMAGE_MOVE&range={%22page_from%22:%20%22shituIndex%22}&extUiData%5bisLogoShow%5d=1', files=data, headers=headers).text
    
        url = json.loads(r)["data"]["url"]
        o = urlparse(url)
        q = parse_qs(o.query, True)
        sign = q['sign'][0]
        r1 = requests.get(url, headers=headers).text
        r0 = requests.get(
            "https://graph.baidu.com/ajax/pcsimi?sign={}".format(sign)).text
    
        l = json.loads(r0)["data"]["list"]
        img_path_list = []
        for i in l:
            img_path_list.append(i['thumbUrl'])
    
        n = 0
        for img_path in img_path_list:
            img_data = requests.get(url=img_path, headers=headers).content
            #save_path = img_seed+img.split('.')[0]+'/' #保存路径
            save_path = 'C:/flower_category/category2/'+ str(j)+'/' #保存路径
            if(not os.path.exists(save_path)):
                os.makedirs(save_path)
            img_path = save_path + img + str(n) + '.jpg'
            with open(img_path, 'wb') as fp:
                fp.write(img_data)
            n = n + 1
    print(j)