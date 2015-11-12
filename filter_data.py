#!/usr/bin/env python3

import pandas as pd
import argparse
from collections import defaultdict
import numpy as np

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("ifile", help="input CSV file", type=argparse.FileType('r'))
	parser.add_argument("ofile", help="output CSV file", type=argparse.FileType('w'))
	args = parser.parse_args()
	src_fname = args.ifile
	dst_fname = args.ofile
	print("Source:", src_fname)
	print("Destination:", dst_fname)
	src_data = pd.read_csv(src_fname, index_col=False)#, nrows = 10)
	print(src_data.shape[0], "rows loaded")
	col_types = src_data.columns.to_series().groupby(src_data.dtypes).groups
	print("Original data set columns:")
	for dtype, cols in col_types.items():
		print("\t", dtype, ":", len(cols))
	print("Filtering only ints and floats")
	needed_cols = sorted(col_types[np.dtype("int64")] + col_types[np.dtype("float64")])
	filtered_cols = src_data[needed_cols]
	col_types = filtered_cols.columns.to_series().groupby(filtered_cols.dtypes).groups
	print("New data set columns:")
	for dtype, cols in col_types.items():
		print("\t", dtype, ":", len(cols))
	print("Filtering columns without NaNs")
	filtered_cols.dropna(axis=1, inplace=True)
	print(filtered_cols.shape[0], "rows remaining")
	col_types = filtered_cols.columns.to_series().groupby(filtered_cols.dtypes).groups
	print("New data set columns:")
	for dtype, cols in col_types.items():
		print("\t", dtype, ":", len(cols))
	print("Filtering rows without NaNs")
	filtered_cols.to_csv(dst_fname)
if __name__ == "__main__":
	main()
