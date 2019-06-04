import argparse
import json
from converter import tsv2coco

parser = argparse.ArgumentParser(description='convert tsv to coco dataset')

parser.add_argument('-c', '--categories', help='path to json categories', type=str, default=None)
parser.add_argument('-r', '--root', help='root folder', type=str, required=True)
parser.add_argument('-e', '--extension', help='image extension', type=str, default='.jpg')
parser.add_argument('-n', '--norm', help='compute norm', type=bool, default=False)
parser.add_argument('-imdir', '--image_dir', help='image directory', type=str, required=True)
parser.add_argument('-anndir', '--annotation_dir', help='annotation directory', type=str, required=True)
parser.add_argument('-d', '--data', help='train or validation set', required=True, choices=['train', 'val'])

args = vars(parser.parse_args())

coco_output = tsv2coco(args['categories'], args['root'], args['image_dir'], args['annotation_dir'], args['norm'])

with open('{0}/instances_{1}2017.json'.format(args['root'], args['data']), 'w') as output_json_file:
    json.dump(coco_output, output_json_file)
