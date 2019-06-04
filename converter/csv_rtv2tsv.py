import pandas as pd
import argparse
import os

# TODO : filter classes?

def csv_rtv2tsv(csv_file, annotation_dir):
    df = pd.read_csv(csv_file)
    cur_name = ''

    for index, row in df.iterrows():
        
        if row['filename'] != cur_name:

            if cur_name != '':
                new_df = pd.DataFrame.from_dict(cur_image)                

                path = os.path.join(annotation_dir, cur_name.split('.')[0] + '.tsv')
                new_df.to_csv(path, sep='\t', index=False)

            cur_name = row['filename']
            cur_image = {'class': [], 'xtl' : [], 'ytl' : [], 
                                      'xbr' : [], 'ybr' : []}

        class_ = '.'.join(row['sign_class'].split('_'))
        cur_image['class'].append(class_)

        xtl = row['x_from']
        cur_image['xtl'].append(xtl)

        ytl = row['y_from']
        cur_image['ytl'].append(ytl)

        xbr = xtl + row['width']
        cur_image['xbr'].append(xbr)
        
        ybr = ytl + row['height']
        cur_image['ybr'].append(ybr)


parser = argparse.ArgumentParser(description='Convert RTS csv to IceVision tsv')

parser.add_argument('-i', '--input', help='RTS csv file', type=str,  required=True)
parser.add_argument('-o', '--outdir', help='output folder', type=str, required=True)

args = vars(parser.parse_args())        

csv_rtv2tsv(args['input'], args['outdir'])        