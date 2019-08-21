import numpy as np
import cv2
import os
import random
from data_aug import data_augmentation
import gen_plates as gen
import model.LPRnet as model


def encode_label(label, char_dict):
    encode = [char_dict[c] for c in label]
    return encode

def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

class DataIterator:
    def __init__(self, img_dir, runtime_generate=False):
        self.img_dir = img_dir
        self.batch_size = model.BATCH_SIZE
        self.channel_num = model.CH_NUM
        self.img_w, self.img_h = model.IMG_SIZE

        if runtime_generate:
            self.generator = gen.ImageGenerator('./fonts', model.CHARS)
        else:
            self.init()

    def init(self):
        self.filenames = []
        self.labels = []
        fs = os.listdir(self.img_dir)
        for filename in fs:
            self.filenames.append(filename)
            label = filename.split('_')[0] # format: [label]_[random number].jpg
            label = encode_label(label, model.CHARS_DICT)
            self.labels.append(label)
        self.sample_num = len(self.labels)
        self.labels = np.array(self.labels)
        self.random_index = list(range(self.sample_num))
        random.shuffle(self.random_index)
        self.cur_index = 0

    def next_sample_ind(self):
        ret = self.random_index[self.cur_index]
        self.cur_index += 1
        if self.cur_index >= self.sample_num:
            self.cur_index = 0
            random.shuffle(self.random_index)
        return ret

    def next_batch(self):

        batch_size = self.batch_size
        images = np.zeros([batch_size, self.img_h, self.img_w, self.channel_num])
        labels = []

        for i in range(batch_size):
            sample_ind = self.next_sample_ind()
            fname = self.filenames[sample_ind]
            img = cv2.imread(os.path.join(self.img_dir, fname))
            #img = data_augmentation(img)
            img = cv2.resize(img, (self.img_w, self.img_h))
            images[i] = img

            labels.append(self.labels[sample_ind])

        sparse_labels = sparse_tuple_from(labels)

        return images, sparse_labels, labels

    def next_test_batch(self):

        start = 0
        end = self.batch_size
        is_last_batch = False

        while not is_last_batch:
            if end >= self.sample_num:
                end = self.sample_num
                is_last_batch = True

            #print("s: {} e: {}".format(start, end))

            cur_batch_size = end-start
            images = np.zeros([cur_batch_size, self.img_h, self.img_w, self.channel_num])

            for j, i in enumerate(range(start, end)):
                fname = self.filenames[i]
                img = cv2.imread(os.path.join(self.img_dir, fname))
                img = cv2.resize(img, (self.img_w, self.img_h))
                images[j, ...] = img

            labels = self.labels[start:end, ...]
            sparse_labels = sparse_tuple_from(labels)

            start = end
            end += self.batch_size

            yield images, sparse_labels, labels

    def next_gen_batch(self):

        batch_size = self.batch_size
        imgs, labels = self.generator.generate_images(batch_size)
        labels = [encode_label(label, model.CHARS_DICT) for label in labels]

        images = np.zeros([batch_size, self.img_h, self.img_w, self.channel_num])
        for i, img in enumerate(imgs):
            img = data_augmentation(img)
            img = cv2.resize(img, (self.img_w, self.img_h))
            images[i, ...] = img

        sparse_labels = sparse_tuple_from(labels)

        return images, sparse_labels, labels