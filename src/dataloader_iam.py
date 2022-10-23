import os
import pickle
import random
from collections import namedtuple
from typing import Tuple
import cv2
import lmdb
import numpy as np
from path import Path
import xml.etree.ElementTree as ET


def textFromXmlFile(xmlFilePath):
    tree = ET.parse(xmlFilePath)
    root = tree.getroot()
    text = root[0].text
    textList = list(text)
    i = 0
    for letter in textList:
        if letter == ' ':
            textList[i] = '|'
        i = i + 1
    return ''.join(textList)


Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size')


class DataLoaderIAM:
    """
    Loads data which corresponds to IAM format,
    see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    def __init__(self,
                 data_dir: Path,
                 batch_size: int,
                 data_split: float = 0.95,
                 fast: bool = True) -> None:
        """Loader for dataset."""

        assert data_dir.exists()

        self.fast = fast
        if fast:
            self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []

        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset
        # self.char_list, self.samples = self.loadAlifImages(data_dir, '/media/droualid/Bibliotheque/Doctorat/Data/ALiF/md_alif_train/')
        self.char_list, self.samples = self.loadImages(data_dir, bad_samples_reference)

        # split into training and validation set: 95% - 5%
        split_idx = int(data_split * len(self.samples))
        self.train_samples = self.samples[:split_idx]
        self.validation_samples = self.samples[split_idx:]

        # put words into lists
        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]

        # start with train set
        self.train_set()

        # list of all chars in dataset
        self.char_list = sorted(list(self.char_list))

    @staticmethod
    def loadImages(data_dir, bad_samples_reference):
        samples = []
        chars = set()
        f = open('wordsEncoded.txt')
        i = 0
        for line in f:
            line = line.split(',')  # ignore empty and comment lines
            if not line or line[0] == '#':
                continue
            path = data_dir + '/' + str(i) + '.png'
            text = ''.join(line)
            text = text[0:len(text) - 1]
            samples.append(Sample(text, path))
            chars = chars.union(set(list(text)))
            i = i + 1
        return chars, samples

    @staticmethod
    def loadImagesMultiFolders(data_dir, bad_samples_reference):
        samples = []
        chars = set()
        fontsDir = os.listdir(data_dir)
        for dirName in fontsDir:
            print(dirName)
            currentDir = data_dir + dirName + '/'
            i = 0
            wordsFile = open('wordsEncoded.txt')
            for line in wordsFile:
                if not line or line[0] == '#':
                    continue
                path = currentDir + str(i) + '.png'
                text = ''.join(line)
                text = text[0:len(text) - 1]
                samples.append(Sample(text, path))
                for char in text.split(','):
                    chars.add(char)
                i = i + 1
            wordsFile.close()
        print('*****************************************************************')
        print('*****************************************************************')
        print('*****************************************************************')
        # print(chars)
        print(samples[0])
        return chars, samples


    @staticmethod
    def loadAlifImages(imgsDir, xmlDir):
        samples = []
        chars = set()
        images = os.listdir(imgsDir)
        xmlFiles = os.listdir(xmlDir)
        for image in images:
            imagePath = imgsDir + image
            xmlFilePath = xmlDir + image[0:len(image) - 4] + '.xml'
            text = textFromXmlFile(xmlFilePath)
            samples.append(Sample(text, imagePath))
            chars = chars.union(set(list(text)))
        print(samples[0])
        return chars, samples

    def train_set(self) -> None:
        """Switch to randomly chosen subset of training set."""
        self.data_augmentation = True
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self) -> None:
        """Switch to validation set."""
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def get_iterator_info(self) -> Tuple[int, int]:
        """Current batch index and overall number of batches."""
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))  # train set: only full-sized batches
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))  # val set: allow last batch to be smaller
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self) -> bool:
        """Is there a next element?"""
        if self.curr_set == 'train':
            return self.curr_idx + self.batch_size <= len(self.samples)  # train set: only full-sized batches
        else:
            return self.curr_idx < len(self.samples)  # val set: allow last batch to be smaller

    def _get_img(self, i: int) -> np.ndarray:
        if self.fast:
            with self.env.begin() as txn:
                basename = Path(self.samples[i].file_path).basename()
                data = txn.get(basename.encode("ascii"))
                img = pickle.loads(data)
        else:
            img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)

        return img

    def get_next(self) -> Batch:
        """Get next element."""
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]
        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs))

