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
	filter_nan_rows = False
	filter_nan_columns = True
	smart_drop_nans = False
	nan_column_drop_threshold = 0.01
	print("Source:", src_fname)
	print("Destination:", dst_fname)
	src_data = pd.read_csv(src_fname, index_col=False)#, nrows = 10)
	total_rows = src_data.shape[0]
	print(total_rows, "rows loaded")
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
	if smart_drop_nans:
		needed_cols = sorted(col_types[np.dtype("float64")])
		#maxes = filtered_cols[needed_cols].max(axis=0)
		#maxes.columns = ['VAR', 'max']
		#mins = filtered_cols[needed_cols].min(axis=0)
		#mins.columns = ['VAR', 'min']
		#minmax = pd.concat([mins, maxes], axis=1, join_axes = [mins.index])
		#pd.set_option('display.height', 500)
		#pd.set_option('display.max_rows', 500)
		#print(minmax)
		#new_mins = minmax.min(axis=0)
		#print("Mins:", new_mins)
		#new_maxes = minmax.max(axis=0)
		#print("Maxes:", new_maxes)
		nan_counts = filtered_cols[needed_cols].isnull().sum()
		nan_ratios = nan_counts / total_rows
		high_nan_ratios = nan_ratios[nan_ratios > nan_column_drop_threshold]
		print("The following columns have the percentage of NaNs above", 100*nan_column_drop_threshold, "% and will be dropped")
		print(high_nan_ratios * 100)
		filtered_cols.drop(high_nan_ratios.index, axis=1, inplace=True)
		col_types = filtered_cols.columns.to_series().groupby(filtered_cols.dtypes).groups
		print("New data set columns:")
		for dtype, cols in col_types.items():
			print("\t", dtype, ":", len(cols))
		print("Filtering rows without NaNs")
		filtered_cols.dropna(axis=0, inplace=True)
		print(filtered_cols.shape[0], "rows remaining")
	if filter_nan_columns:
		print("Filtering columns without NaNs")
		filtered_cols.dropna(axis=1, inplace=True)
		print(filtered_cols.shape[0], "rows remaining")
		col_types = filtered_cols.columns.to_series().groupby(filtered_cols.dtypes).groups
		print("New data set columns:")
		for dtype, cols in col_types.items():
			print("\t", dtype, ":", len(cols))
	if filter_nan_rows:
		print("Filtering rows without NaNs")
		filtered_cols.dropna(axis=0, inplace=True)
		print(filtered_cols.shape[0], "rows remaining")
		col_types = filtered_cols.columns.to_series().groupby(filtered_cols.dtypes).groups
		print("New data set columns:")
		for dtype, cols in col_types.items():
			print("\t", dtype, ":", len(cols))

	filtered_cols.to_csv(dst_fname)
if __name__ == "__main__":
	main()
