import os
import re
import shutil
cop = re.compile("[^a-z^A-Z^0-9 ]") # 匹配不是中文、大小写、数字的其他字符

train_arr = []
test_arr = []
with open(r'F:\AISHELL-3\train\content.txt', 'r',  encoding="utf-8") as f:
    str = f.read()
    train_arr = str.split('\n')

with open(r'F:\AISHELL-3\test\content.txt', 'r',  encoding="utf-8") as f:
    str = f.read()
    test_arr = str.split('\n')

print(len(train_arr) + len(test_arr))

all_arr = train_arr + test_arr

from tqdm import tqdm

# for i in tqdm(train_arr):
#     b = i.split('\t')[-1]
#     txt = cop.sub('', b)[1:].replace("  ", " ")
#
#     a = i.split('\t')[0]
#     fold_name = a[:7]
#     # print(fold_name)
#     path = rf'F:\data_AISHELL\MFA\{fold_name}'
#     isExists = os.path.exists(path)
#     if not isExists:
#         # 如果不存在则创建目录
#         # 创建目录操作函数
#         os.makedirs(path)
#
#     src_wav_path = rf'F:\AISHELL-3\train\wav\{fold_name}\{a}'
#     des_wav_path = rf'{path}\{a}'
#     shutil.copy(src_wav_path, des_wav_path)
#
#     text_path = rf'{path}\{a[:-4]}.txt'
#     with open(text_path, 'w', encoding="utf-8") as f:
#         f.write(txt)

for i in tqdm(test_arr):
    b = i.split('\t')[-1]
    txt = cop.sub('', b)[1:].replace("  ", " ")

    a = i.split('\t')[0]
    fold_name = a[:7]
    # print(fold_name)
    path = rf'F:\data_AISHELL\MFA\{fold_name}'
    isExists = os.path.exists(path)
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

    src_wav_path = rf'F:\AISHELL-3\test\wav\{fold_name}\{a}'
    des_wav_path = rf'{path}\{a}'
    shutil.copy(src_wav_path, des_wav_path)

    text_path = rf'{path}\{a[:-4]}.txt'
    with open(text_path, 'w', encoding="utf-8") as f:
        f.write(txt)

