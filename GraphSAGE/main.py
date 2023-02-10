# 多次调用 train.py
import os


def diff_epoch():
    epochs = 50
    for epoch in range(epochs):
        sent = 'python train.py --epoch {}'.format(epoch)
        print('\n' + '\n' + '\n' + '执行 {} 语句'.format(sent))
        os.system(sent)
    return


if __name__ == '__main__':
    diff_epoch()