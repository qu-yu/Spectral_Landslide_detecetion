import warnings

warnings.filterwarnings("ignore")
import os, cv2, shutil

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch, time, datetime, tqdm
import numpy as np

from dataset import DataGenerator
from models.modeling import deeplabv3_resnet50

from math import cos, pi
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from libtiff import TIFF


def predict(img):
    img = cv2.resize(img, (NETWORK_SIZE, NETWORK_SIZE))
    img = np.transpose(np.expand_dims(img, axis=0), (0, 3, 1, 2))
    img = torch.from_numpy(img).to(DEVICE).float()
    pred = np.argmax(model(img).cpu().detach().numpy()[0], axis=0).astype(np.uint8)
    pred = cv2.resize(pred, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_NEAREST)
    return np.array(pred, dtype=np.uint8)


def metrice(y_true, y_pred):
    y_true, y_pred = y_true.reshape((-1)), y_pred.reshape((-1))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    mpa_arr = np.diag(cm) / (cm.sum(axis=1) + 1e-7)
    mpa = np.nanmean(mpa_arr)
    return accuracy_score(y_true, y_pred), mpa, recall_score(y_true, y_pred), precision_score(y_true, y_pred), f1_score(
        y_true, y_pred)


if __name__ == '__main__':
    IMAGE_SIZE, NETWORK_SIZE = 256, 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    if os.path.exists('result'):
        shutil.rmtree('result')
    os.mkdir('result')

    model = torch.load('model.pkl')
    model.eval()
    model.to(DEVICE)

    with open('result.txt', 'w+') as f:
        pass

    for file in os.listdir('image'):
        since = time.time()
        image = TIFF.open('image\{}'.format(file), mode='r').read_image()
        img_shape = image.shape
        image_min, image_max = np.min(image), np.max(image)
        image = (image - image_min) / (image_max - image_min)

        label = TIFF.open('label\{}'.format(file), mode='r').read_image()
        label[label > 1] = 1

        img_mask = np.zeros(shape=img_shape[:-1])

        for i in range(img_shape[0] // IMAGE_SIZE):
            for j in range(img_shape[1] // IMAGE_SIZE):
                img_mask[IMAGE_SIZE * i:IMAGE_SIZE * (i + 1), IMAGE_SIZE * j:IMAGE_SIZE * (j + 1)] = predict(
                    image[IMAGE_SIZE * i:IMAGE_SIZE * (i + 1),
                    IMAGE_SIZE * j:IMAGE_SIZE * (j + 1)])

        for i in range(img_shape[0], 0, -IMAGE_SIZE):
            for j in range(img_shape[1], 0, -IMAGE_SIZE):
                try:
                    img_mask[i - IMAGE_SIZE:i, j - IMAGE_SIZE:j] = predict(image[i - IMAGE_SIZE:i, j - IMAGE_SIZE:j])
                except:
                    pass
                else:
                    if i != img_shape[0]:
                        break

        img_mask[-IMAGE_SIZE:, :IMAGE_SIZE] = predict(image[-IMAGE_SIZE:, :IMAGE_SIZE])
        img_mask[:IMAGE_SIZE, -IMAGE_SIZE:] = predict(image[:IMAGE_SIZE, -IMAGE_SIZE:])
        #img_mask = img_mask.astype(np.int16)
        label = label.astype(np.uint8)

        acc, mpa, recall, precision, f1score = metrice(label, img_mask)

        img_mask = np.array(img_mask, np.uint8)
        tif = TIFF.open('result/{}'.format(file), mode='w')
        tif.write_image(img_mask)
        print(
            '{} predict done! use time:{:.3f}s acc:{:.3f} mpa:{:.3f} recall:{:.3f} precision:{:.3f} f1-score:{:.3f}'.format(
                file,
                time.time() - since,
                acc,
                mpa,
                recall,
                precision,
                f1score))

        with open('result.txt', 'a+') as f:
            f.write(
                '{} predict done! use time:{:.3f}s acc:{:.3f} mpa:{:.3f} recall:{:.3f} precision:{:.3f} f1-score:{:.3f}\n'.format(
                    file, time.time() - since,
                    acc, mpa, recall, precision, f1score))
