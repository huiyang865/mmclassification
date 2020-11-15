import glob
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import cv2
import lmdb
from loguru import logger

_10TB = 10 * (1 << 40)


class LmdbDataExporter(object):
    """
    making LMDB database
    """
    label_pattern = re.compile(r'/.*/.*?(\d+)$')

    def __init__(self,
                 img_dir=None,
                 output_path=None,
                 dir_level=5,
                 class_level=2,
                 train_ratio=0.8,
                 shape=(256, 256),
                 batch_size=100,
                 target_dir_name=''):
        """
            img_dir: imgs directory
            output_path: LMDB output path
        """
        self.img_dir = img_dir
        self.output_path = output_path
        self.dir_level = dir_level
        self.class_level = class_level
        self.train_ratio = train_ratio
        self.shape = shape
        self.batch_size = batch_size
        self.label_list = list()
        self.train_idx = 0
        self.val_idx = 0
        self.target_dir_name = target_dir_name

        if not os.path.exists(img_dir):
            raise Exception(f'{img_dir} is not exists!')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        train_output_path = os.path.join(output_path, 'train.lmdb')
        val_output_path = os.path.join(output_path, 'trainval.lmdb')
        if not os.path.exists(train_output_path):
            os.makedirs(train_output_path)
        if not os.path.exists(val_output_path):
            os.makedirs(val_output_path)

        # 最大10T
        self.train_lmdb_env = lmdb.open(
            train_output_path, map_size=_10TB, max_dbs=4)
        self.val_lmdb_env = lmdb.open(
            val_output_path, map_size=_10TB, max_dbs=4)

        self.label_dict = defaultdict(int)

    def persist_2_database(self, items, item_img, results, st, is_train):
        if len(items) < self.batch_size:
            if is_train:
                self.train_idx += 1
            else:
                self.val_idx += 1

            items.append(item_img)
            return st, items, results

        with ThreadPoolExecutor() as executor:
            results.extend(executor.map(self._extract_once, items))
            del items[:]

        if len(results) >= self.batch_size:
            self.save_to_lmdb(results, is_train)
            et = time.time()
            if is_train:
                logger.info(
                    f'time: {(et-st)}(s) training count: {self.train_idx}')
            else:
                logger.info(f'time: {(et-st)}(s) val count: {self.val_idx}')
            st = time.time()
            del results[:]

        return st, items, results

    def export(self):
        train_results = []
        val_results = []
        st = time.time()
        iter_img_lst = self.read_imgs()
        train_items, val_items = [], []
        for item_img in iter_img_lst:
            if random.random(
            ) > self.train_ratio and self.target_dir_name in item_img[-1]:
                item_img[0] = self.val_idx
                st, val_items, val_results = self.persist_2_database(
                    val_items, item_img, val_results, st, is_train=False)
            else:
                item_img[0] = self.train_idx
                st, train_items, train_results = self.persist_2_database(
                    train_items, item_img, train_results, st, is_train=True)

        with ThreadPoolExecutor() as executor:
            train_results.extend(executor.map(self._extract_once, train_items))
            val_results.extend(executor.map(self._extract_once, val_items))
            del train_items[:], val_items[:]

        self.save_to_lmdb(train_results, is_train=True)
        self.save_to_lmdb(val_results, is_train=False)

        self.save_total()
        del train_results[:]
        del val_results[:]

        et = time.time()
        logger.info(f'time: {(et-st)}(s)  count: {self.train_idx}')
        logger.info(f'time: {(et-st)}(s)  count: {self.val_idx}')

    def save_to_lmdb(self, results, is_train):
        """
        persist to lmdb
        """
        if is_train:
            with self.train_lmdb_env.begin(write=True) as txn:
                while results:
                    img_key, img_byte = results.pop()
                    if img_key is None or img_byte is None:
                        continue
                    txn.put(img_key, img_byte)
        else:
            with self.val_lmdb_env.begin(write=True) as txn:
                while results:
                    img_key, img_byte = results.pop()
                    if img_key is None or img_byte is None:
                        continue
                    txn.put(img_key, img_byte)

    def save_total(self):
        """
        persist all numbers of imgs
        """
        with self.train_lmdb_env.begin(write=True, buffers=True) as txn:
            txn.put('total'.encode(), str(self.train_idx).encode())
        with self.val_lmdb_env.begin(write=True, buffers=True) as txn:
            txn.put('total'.encode(), str(self.val_idx).encode())

    def _extract_once(self, item) -> Tuple[bytes, bytes]:
        full_path = item[-1]
        imageKey = '###'.join(map(str, item[:-1]))

        img = cv2.imread(full_path)
        if img is None:
            logger.error(f'{full_path} is a bad img file.')
            return None, None
        if img.shape != self.shape:
            img = self.fillImg(img)
        _, img_byte = cv2.imencode('.jpg', img)
        return (imageKey.encode(), img_byte.tobytes())

    def fillImg(self, img):
        width = img.shape[1]
        height = img.shape[0]
        top, bottom, left, right = 0, 0, 0, 0
        if width > height:
            diff = width - height
            top = int(diff / 2)
            bottom = diff - top
        else:
            diff = height - width
            left = int(diff / 2)
            right = diff - left
        fimg = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0])
        rimg = cv2.resize(fimg, self.shape, interpolation=cv2.INTER_AREA)
        return rimg

    def read_imgs(self):
        img_list, dir_level = [], ''
        for i in range(self.dir_level):
            dir_level += '*/'
            for img_postfix in ['jpg', 'png', 'JPEG']:
                img_list.extend(
                    glob.glob(
                        os.path.join(self.img_dir,
                                     f'{dir_level}*.{img_postfix}')))

        for idx, item_img in enumerate(img_list):
            if self.img_dir.endswith('/'):
                self.img_dir = self.img_dir[:-1]

            label = item_img.replace(self.img_dir,
                                     '').split('/')[self.class_level]

            if label not in self.label_list:
                self.label_list.append(label)

            item = [idx, self.label_list.index(label), item_img]
            yield item
