import pandas as pd
from tqdm import tqdm
import json
import os
import glob

test_list = pd.DataFrame(columns=['img_name', 'label'])
remove_file = []
test_loc = 0
file_path = fr'kids/test_data'
json_paths = sorted(glob.glob(f'{file_path}/**/*.json', recursive=True))
img_paths = sorted(glob.glob(f'{file_path}/**/*.jpg', recursive=True))
img_paths = sorted(img_paths)
json_paths = sorted(json_paths)
for json_index, json_name in enumerate(tqdm(json_paths)):
    a = img_paths[json_index].split('/')[-1].replace('.jpg', '')
    b = json_name.split('/')[-1].replace('.json', '')
    
    if a != b:
        print(a, b)
        print('이미지, json 다름')
        break

    with open('label_map.json') as f:
        label_data = json.load(f)

        with open(json_name, "r", encoding='utf-8-sig') as f:
            if len(test_list) % 10000 == 0:
                test_list.to_pickle(f'test_list{json_index}.pkl')
                test_loc = 0
                test_list = pd.DataFrame(columns=['img_name', 'label'])
                remove_file.append(f'test_list{json_index}.pkl')
            data = json.load(f)
            label = data['info']['pattern']
            for label_name in label:
                test_list.loc[test_loc, 'img_name'] = img_paths[json_index]
                test_list.loc[test_loc, 'label'] = label_name.strip()
                test_loc += 1
test_list.to_pickle('last.pkl')
remove_file.append('last.pkl')
test_list = pd.DataFrame()
for i in remove_file:
    test_list = pd.concat([test_list, pd.read_pickle(i)])
for temp in remove_file:
    os.remove(temp)
test_list['label'] = test_list['label'].apply(lambda x: label_data[x])
test_list = test_list.drop_duplicates('img_name', keep='last')
test_list.to_csv('test.txt', header=None, index=False, sep=' ')

with open('test.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 큰따옴표를 내용에서 제거
content_without_quotes = content.replace('"', '')

# 수정된 내용을 새로운 텍스트 파일에 쓰기
with open('test.txt', 'w', encoding='utf-8') as file:
    file.write(content_without_quotes)