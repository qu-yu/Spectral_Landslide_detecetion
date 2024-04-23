import warnings

warnings.filterwarnings("ignore")
import os, cv2, shutil

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch, time, datetime, tqdm
import numpy as np

from dataset import DataGenerator
from models.modeling import deeplabv3_resnet50
from libtiff import TIFF


def predict(img):
    img = cv2.resize(img, (NETWORK_SIZE, NETWORK_SIZE))
    img = np.transpose(np.expand_dims(img, axis=0), (0, 3, 1, 2))
    img = torch.from_numpy(img).to(DEVICE).float()
    pred = np.argmax(model(img).cpu().detach().numpy()[0], axis=0).astype(np.uint8)
    pred = cv2.resize(pred, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_NEAREST)
    return np.array(pred, dtype=np.uint8)


if __name__ == '__main__':
    IMAGE_SIZE, NETWORK_SIZE = 1024, 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    if os.path.exists('pred-result'):
        shutil.rmtree('pred-result')
    os.mkdir('pred-result')

    model = torch.load('model.pkl')
    model.eval()
    model.to(DEVICE)

    for file in os.listdir('img'):
        since = time.time()
        image = TIFF.open('img\{}'.format(file), mode='r').read_image()
        img_shape = image.shape
        image_min, image_max = np.min(image), np.max(image)
        image = (image - image_min) / (image_max - image_min)

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

        img_mask = np.array(img_mask, np.uint8)
        tif = TIFF.open('pred-result/{}'.format(file), mode='w')
        tif.write_image(img_mask)