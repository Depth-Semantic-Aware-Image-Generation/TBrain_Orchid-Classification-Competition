import os, shutil, random, glob
import copy

import PIL.Image as img

dir_data = 'C:\\flower_category'            #训练集路径

dir_category = os.path.join(dir_data, 'category')
category_list = os.listdir(dir_category)
category_list.sort(key=lambda x:int(x.split('.')[0]))


i = 0
for c in category_list:
    category_list[i] = [os.path.abspath(fp) for fp in glob.glob(os.path.join(os.path.join(dir_category, c), '*.jpg'))]
    i = i+1


for i in range(219):    
    for j in range(10):
        im = img.open(category_list[i][j])
        #ng = im.transpose(img.ROTATE_180) #旋转 180 度角。
        #ng1 = im.transpose(img.FLIP_LEFT_RIGHT) #左右对换。
        #ng = im.transpose(img.FLIP_TOP_BOTTOM)  # 上下对换。
        ng2 = im.rotate(90)
        ng3 = im.rotate(-90)
        
        #ng1.save(category_list[i][j].replace('.jpg','_1.jpg'))
        ng2.save(category_list[i][j].replace('.jpg','_2.jpg'))
        ng3.save(category_list[i][j].replace('.jpg','_3.jpg'))
    print(j)

#tf.image.rotate(10)
