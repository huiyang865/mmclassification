import argparse
import sys

from loguru import logger

from mmcls.datasets.persistences.persist_lmdb import LmdbDataExporter


def parse_args():
    parser = argparse.ArgumentParser(
        description='Making LMDB database with multiprocess')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/home/yanghui/yanghui/openset_v2/dataset/train_data',
        help='the input dir of imgs')
    parser.add_argument(
        '--output_path',
        type=str,
        default='/home/yanghui/yanghui/open_git/mmclassification/data/lmdb',
        help='output path of LMDB')
    parser.add_argument(
        '--target_dir_name',
        default='other1',
        help='the hierarchy number of input_dir which contains the image files'
    )
    parser.add_argument(
        '--dir_level',
        default=5,
        help='the hierarchy number of input_dir which contains the image files'
    )
    parser.add_argument(
        '--class_level',
        default=2,
        help='the level number for the class directory')
    parser.add_argument(
        '--train_ratio',
        default=0.9,
        help='the level number for the class directory')
    parser.add_argument(
        '--shape',
        default=(256, 256),
        help='reshaping size of imgs before saving')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='batch size of each process to save imgs')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    exporter = LmdbDataExporter(
        args.input_dir,
        args.output_path,
        dir_level=args.dir_level,
        class_level=args.class_level,
        train_ratio=args.train_ratio,
        shape=args.shape,
        batch_size=args.batch_size,
        target_dir_name=args.target_dir_name)

    logger.configure(
        **{'handlers': [
            {
                'sink': sys.stdout,
                'level': 'INFO',
            },
        ]})

    exporter.export()
    logger.info(f'class num is: {len(exporter.label_list)}')
    logger.info(f'training img num is: {exporter.train_idx}')
    logger.info(f'val img num is: {exporter.val_idx}')


if __name__ == '__main__':
    main()
