#!/usr/bin/env python3

import pandas as pd
import argparse
from collections import defaultdict
import numpy as np

def normalize_cols(filtered_cols, needed_cols, minmax):

	for var in minmax.index:
		if var in ['ID', 'target']:
			continue
		if np.isnan(minmax.loc[var, 'ranges']) or minmax.loc[var, 'ranges'] == 0:
			continue
		print(var)
		filtered_cols[var] = filtered_cols[var].subtract(minmax.loc[var, 'min'])
		filtered_cols[var] = filtered_cols[var].divide(minmax.loc[var, 'ranges'])
	return filtered_cols

def smart_nan_drop(filtered_cols, high_nan_ratios):
	filtered_cols_nandrop = filtered_cols.drop(high_nan_ratios.index, axis=1)
	col_types = filtered_cols_nandrop.columns.to_series().groupby(filtered_cols_nandrop.dtypes).groups
	print("New data set columns:")
	for dtype, cols in col_types.items():
		print("\t", dtype, ":", len(cols))
	print("Filtering rows without NaNs")
	filtered_cols_nandrop.dropna(axis=0, inplace=True)
	print(filtered_cols_nandrop.shape[0], "rows remaining")
	return filtered_cols_nandrop

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("ifile", help="input CSV file", type=argparse.FileType('r'))
	args = parser.parse_args()
	src_fname = args.ifile
	filter_nan_rows = False
	filter_nan_columns = False
	smart_drop_nans = False
	process_numeric_features = True
	normalize_columns = True
	nan_column_drop_threshold = 0.01
	print("Source:", src_fname)
	#print("Destination:", dst_fname)
	src_fname_str = str(src_fname.name)
	src_data = pd.read_csv(src_fname)#, nrows=10)
	total_rows = src_data.shape[0]
	print(total_rows, "rows loaded")
	col_types = src_data.columns.to_series().groupby(src_data.dtypes).groups
	print("Original data set columns:")
	for dtype, cols in col_types.items():
		print("\t", dtype, ":", len(cols))
	if process_numeric_features:
		print("Test fname:", src_fname_str)
		train_fname = src_fname_str.split('.')[:-1]
		train_fname = '.'.join(train_fname)
		train_fname = train_fname.split('/')[:-1]
		train_fname = '/'.join(train_fname + ['train'])
		print("Train fname:", train_fname)

		print("Filtering only ints and floats")
		needed_cols = pd.read_csv('.'.join([train_fname] + ['needed_cols', 'csv']))
		needed_cols.set_index('ID', inplace=True)
		needed_cols = needed_cols['VAR'].tolist()[:-1]
		print(needed_cols)
		filtered_cols = src_data[needed_cols]
		col_types = filtered_cols.columns.to_series().groupby(filtered_cols.dtypes).groups
		print("New data set columns:")
		for dtype, cols in col_types.items():
			print("\t", dtype, ":", len(cols))

		high_nan_ratios = pd.read_csv('.'.join([train_fname] + ['high_nan_ratios', 'nandrop', 'csv']))
		high_nan_ratios.set_index('VAR', inplace=True)
		print(high_nan_ratios)
		minmax = pd.read_csv('.'.join([train_fname] + ['minmax', 'csv']))
		minmax.set_index('VAR', inplace=True)
		minmax_nandrop = pd.read_csv('.'.join([train_fname] + ['minmax', 'nandrop', 'csv']))
		minmax_nandrop.set_index('VAR', inplace=True)

		filtered_cols_nandrop = smart_nan_drop(filtered_cols, high_nan_ratios)
		

		normalized_filtered_cols = normalize_cols(filtered_cols, needed_cols, minmax)
		normalized_filtered_cols.fillna(value=-1, inplace=True)

		normalized_filtered_cols_nandrop = normalize_cols(filtered_cols_nandrop, filtered_cols_nandrop.columns, minmax_nandrop)


		filtered_cols_nandrop.to_csv('.'.join(src_fname_str.split('.')[:-1] + ['filtered', 'nandrop', 'csv']), header=False)
		
		normalized_filtered_cols_nandrop.to_csv('.'.join(src_fname_str.split('.')[:-1] + ['filtered', 'nandrop', 'normalized', 'csv']), header=False)


		normalized_filtered_cols.to_csv('.'.join(src_fname_str.split('.')[:-1] + ['filtered', 'normalized', 'csv']), header=False)

		

	#filtered_cols.to_csv(dst_fname)
if __name__ == "__main__":
	main()
