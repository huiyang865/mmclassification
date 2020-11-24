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
                 input_path=None,
                 output_path=None,
                 shape=(256, 256),
                 batch_size=100):
        """
            img_dir: imgs directory
            output_path: LMDB output path
        """
        self.input_path = input_path
        self.output_path = output_path
        self.shape = shape
        self.batch_size = batch_size
        self.label_list = list()
        self.idx = 0

        # 最大10T
        self.lmdb_env = lmdb.open(output_path, map_size=_10TB, max_dbs=4)

        self.label_dict = defaultdict(int)

    def persist_2_database(self, items, item_img, results, st):
        if len(items) < self.batch_size:
            self.idx += 1

            items.append(item_img)
            return st, items, results

        with ThreadPoolExecutor() as executor:
            results.extend(executor.map(self._extract_once, items))
            del items[:]

        if len(results) >= self.batch_size:
            self.save_to_lmdb(results)
            et = time.time()
            logger.info('time: {}(s) training count: {}'.format((et - st),
                                                                self.idx))
            st = time.time()
            del results[:]

        return st, items, results

    def export(self):
        st = time.time()
        results, items = [], []
        iter_img_lst = self.read_imgs()
        for item_img in iter_img_lst:
            item_img[0] = self.idx
            st, items, results = self.persist_2_database(
                items, item_img, results, st)

        with ThreadPoolExecutor() as executor:
            results.extend(executor.map(self._extract_once, items))
            del items[:]

        self.save_to_lmdb(results)

        self.save_total()
        del results[:]

        et = time.time()
        logger.info('time: {}(s)  count: {}'.format((et - st), self.idx))

    def save_to_lmdb(self, results):
        """
        persist to lmdb
        """
        with self.lmdb_env.begin(write=True) as txn:
            while results:
                img_key, img_byte = results.pop()
                if img_key is None or img_byte is None:
                    continue
                txn.put(img_key, img_byte)

    def save_total(self):
        """
        persist all numbers of imgs
        """
        with self.lmdb_env.begin(write=True, buffers=True) as txn:
            txn.put('total'.encode(), str(self.idx).encode())

    def _extract_once(self, item) -> Tuple[bytes, bytes]:
        full_path = item[-1]
        imageKey = '###'.join(map(str, item[:-1]))

        img = cv2.imread(full_path)
        if img is None:
            logger.error('{} is a bad img file.'.format(full_path))
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
        for idx, item_line in enumerate(
                open(self.input_path, 'r').read().split('\n')):
            split_items = item_line.split('\t')
            if len(split_items) != 3:
                continue

            label, item_img = split_items[1], split_items[2]

            if label not in self.label_list:
                self.label_list.append(label)

            item = [idx, int(label), item_img]
            yield item
