'''
 Write code that executes like this python download_data.py --source-path "s3://my-bucket/data" --destination-dir ${{ github.workspace }}/data --sample. The code downloads iris dataset dataframe
'''

import argparse
import pandas as pd
import os





def download_data(source_path, destination_dir, sample):
    # df = pd.read_csv(source_path)
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    print(df.head())
    if sample:
        df = df.sample(frac=0.1)
    df.to_csv(os.path.join(destination_dir, 'data.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", type=str, help="source path")
    parser.add_argument("--destination-dir", type=str, help="destination directory")
    parser.add_argument("--sample", action="store_true", help="sample data")
    args = parser.parse_args()
    download_data(args.source_path, args.destination_dir, args.sample)
    # 
