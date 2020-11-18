import os
import sys

import lmdb
import mmcv
from loguru import logger

from mmcls.datasets.persistences.persist_lmdb import LmdbDataExporter


def test_lmdb_exporter(img_dir='tests/data', output_path='tests/data'):
    exporter = LmdbDataExporter(img_dir=img_dir, output_path=output_path)

    assert exporter.img_dir is not None
    assert exporter.output_path is not None

    assert os.path.exists(exporter.img_dir)
    assert os.path.isdir(exporter.img_dir)

    logger.configure(
        **{'handlers': [
            {
                'sink': sys.stdout,
                'level': 'INFO',
            },
        ]})

    exporter.export()

    assert os.path.exists(os.path.join(exporter.output_path, 'train.lmdb'))
    assert os.path.exists(os.path.join(exporter.output_path, 'val.lmdb'))


def test_read_lmdb(ann_file='data/lmdb/train.lmdb'):
    env = lmdb.open(ann_file)
    txn = env.begin(write=False)
    max_label = 0
    for key, imgs in txn.cursor():
        key = key.decode()
        assert float(key.split('###')[-1]) < 5532
        assert '###' in key or 'total' in key
        assert imgs is not None
        max_label = max(max_label, float(key.split('###')[-1]))
        mmcv.imfrombytes(imgs, flag='color')
    print(max_label)


if __name__ == '__main__':
    # test_lmdb_exporter()
    test_read_lmdb()
