import os,tqdm

import numpy as np

from sklearn.model_selection import train_test_split

temp = []
for i in tqdm.tqdm(os.listdir('seg_img')):
    temp.append('{}\t{}'.format('seg_img\{}'.format(i), 'seg_label\{}'.format(i)))

print(len(temp))
train, test = train_test_split(temp, test_size=0.2, random_state=42, shuffle=True)

with open('train.txt', 'w+') as f:
    f.write('\n'.join(train))

with open('test.txt', 'w+') as f:
    f.write('\n'.join(test))