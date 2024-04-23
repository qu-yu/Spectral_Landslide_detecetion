import cv2, os, math, random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DataGenerator(Dataset):
    def __init__(self, Path, target_size=(256, 256), valid=False):
        with open(Path) as f:
            imgPath = list(map(lambda x: x.strip(), f.readlines()))

        self.imgPath_list = np.array(imgPath)
        self.target_size = target_size
        self.indexes = np.arange(len(self.imgPath_list))
        self.valid = valid

    def __len__(self):
        return len(self.imgPath_list)

    def __getitem__(self, index):
        indexes = self.indexes[index:(index + 1)]

        x, y = self.__data_generation(self.imgPath_list[indexes][0])
        x = np.transpose(x, axes=[2, 0, 1])

        return x, y

    def __data_generation(self, img_path):
        img = np.load(img_path.split('\t')[0], allow_pickle=True)
        mask = np.load(img_path.split('\t')[1], allow_pickle=True)
        mask[mask > 1] = 1

        random_int = random.randint(1, 5)
        if self.valid:
            random_int = 3

        if random_int == 1:
            img, mask = self.random_crop(img, mask)
        elif random_int == 2:
            img, mask = self.flip(img, mask)
        else:
            pass

        img = cv2.resize(img, self.target_size)
        mask = cv2.resize(mask, self.target_size, cv2.INTER_NEAREST)

        return img, mask

    def random_crop(self, img, mask, scale=[0.8, 1.0], ratio=[3. / 4., 4. / 3.]):
        """
        随机裁剪
        """
        aspect_ratio = math.sqrt(np.random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio
        src_h, src_w = img.shape[:2]

        bound = min((float(src_w) / src_h) / (w ** 2),
                    (float(src_h) / src_w) / (h ** 2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = src_h * src_w * np.random.uniform(scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = np.random.randint(0, src_w - w + 1)
        j = np.random.randint(0, src_h - h + 1)

        img = img[j:j + h, i:i + w]
        mask = mask[j:j + h, i:i + w]
        return img, mask

    def flip(self, img, mask):
        """
        翻转
        :param img:
        :param mode: 1=水平翻转 / 0=垂直 / -1=水平垂直
        :return:
        """
        mode = np.random.choice([-1, 0, 1])
        return cv2.flip(img, flipCode=mode), cv2.flip(mask, flipCode=mode)


if __name__ == '__main__':
    import tqdm

    a = DataGenerator('train.txt')
    idx = 0
    for i, j in tqdm.tqdm(a):
        pass
    print(idx)
