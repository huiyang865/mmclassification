import argparse
import sys

from datasets.persist_lmdb import LmdbDataExporter
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Making LMDB database with multiprocess')
    parser.add_argument(
        '--input_path', type=str, default='', help='the input dir of imgs')
    parser.add_argument(
        '--output_path', type=str, default='', help='output path of LMDB')
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
        args.input_path,
        args.output_path,
        shape=args.shape,
        batch_size=args.batch_size)

    logger.configure(
        **{'handlers': [
            {
                'sink': sys.stdout,
                'level': 'INFO',
            },
        ]})

    exporter.export()
    logger.info('class num is: {}'.format(len(exporter.label_list)))
    logger.info('img num is: {}'.format(exporter.idx))


if __name__ == '__main__':
    main()
