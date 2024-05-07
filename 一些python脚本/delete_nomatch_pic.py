import os

labels_path = r'C:\xukejian\Projects\dataset\yihualu_smoking1_totrain\labels'
pic_path = r'C:\xukejian\Projects\dataset\yihualu_smoking1_totrain\抽烟喝水数据集'

# for root,dirs,files in os.walk(labels_path):
#     for file in files:
#         src_file=os.path.join(root,file)
#         if os.path.getsize(src_file) == 0:
#             print(src_file)
#             os.remove(src_file)

# jpgs = os.listdir(pic_path)
# for jpg in jpgs:
#     name = os.path.splitext(jpg)[0]
#     jpg_path = os.path.join(pic_path,jpg)
#     label_path = os.path.join(labels_path,name+'.txt')
#     if os.path.exists(jpg_path) and not os.path.exists(label_path):
#         os.remove(jpg_path)
#         print(jpg_path)