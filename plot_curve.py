import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.log')
epoch = data['epoch']
pa = data['train_pa']
mpa = data['train_mpa']
loss = data['train_loss']
val_pa = data['test_pa']
val_mpa = data['test_mpa']
val_loss = data['test_loss']

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(epoch, loss,  label='train',color='b',linestyle='-')
plt.plot(epoch, val_loss, label='test',color='g',linestyle='--')
plt.legend()
plt.title('LOSS')

plt.subplot(2, 2, 2)
plt.plot(epoch, pa, label='train')
plt.plot(epoch, val_pa, label='test')
plt.legend()
plt.title('PA')

plt.subplot(2, 2, 3)
plt.plot(epoch, mpa, label='train')
plt.plot(epoch, val_mpa, label='test')
plt.legend()
plt.title('MPA')

plt.show()