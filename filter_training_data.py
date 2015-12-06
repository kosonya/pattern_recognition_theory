#!/usr/bin/env python3

import pandas as pd
import argparse
from collections import defaultdict
import numpy as np
import sys

def normalize_cols(filtered_cols, needed_cols):
	maxes = filtered_cols[needed_cols].max(axis=0)
	maxes.columns = ['VAR', 'max']
	mins = filtered_cols[needed_cols].min(axis=0)
	mins.columns = ['VAR', 'min']
	minmax = pd.concat([mins, maxes], axis=1, join_axes = [mins.index])
	minmax.columns = ['min', 'max']
	#pd.set_option('display.height', 500)
	#pd.set_option('display.max_rows', 500)
	print(minmax)
	new_mins = minmax.min(axis=0)
	print("Mins:", new_mins)
	new_maxes = minmax.max(axis=0)
	print("Maxes:", new_maxes)
	minmax['ranges'] = minmax['max'] - minmax['min']
	for var in minmax.index:
		if var in ['ID', 'target']:
			continue
		if np.isnan(minmax.loc[var, 'ranges']) or minmax.loc[var, 'ranges'] == 0:
			continue
		print(var)
		filtered_cols[var] = filtered_cols[var].subtract(minmax.loc[var, 'min'])
		filtered_cols[var] = filtered_cols[var].divide(minmax.loc[var, 'ranges'])
	return filtered_cols, minmax

def smart_nan_drop(filtered_cols, needed_cols, total_rows, nan_column_drop_threshold):
	nan_counts = filtered_cols[needed_cols].isnull().sum()
	nan_ratios = nan_counts / total_rows
	high_nan_ratios = nan_ratios[nan_ratios > nan_column_drop_threshold]
	high_nan_ratios = high_nan_ratios.to_frame()
	high_nan_ratios.columns = ['nan_freq']
	print(high_nan_ratios)
	print("The following columns have the percentage of NaNs above", 100*nan_column_drop_threshold, "% and will be dropped")
	print(high_nan_ratios * 100)
	filtered_cols_nandrop = filtered_cols.drop(high_nan_ratios.index, axis=1)
	col_types = filtered_cols_nandrop.columns.to_series().groupby(filtered_cols_nandrop.dtypes).groups
	print("New data set columns:")
	for dtype, cols in col_types.items():
		print("\t", dtype, ":", len(cols))
	print("Filtering rows without NaNs")
	filtered_cols_nandrop.dropna(axis=0, inplace=True)
	print(filtered_cols_nandrop.shape[0], "rows remaining")
	return filtered_cols_nandrop, high_nan_ratios

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

	scr_fname_str = str(src_fname.name)
	src_data = pd.read_csv(src_fname, index_col=False)#, nrows = 10)
	total_rows = src_data.shape[0]
	print(total_rows, "rows loaded")
	col_types = src_data.columns.to_series().groupby(src_data.dtypes).groups
	print("Original data set columns:")
	for dtype, cols in col_types.items():
		print("\t", dtype, ":", len(cols))
	if process_numeric_features:
		print("Filtering only ints and floats")
		needed_cols = sorted(col_types[np.dtype("int64")] + col_types[np.dtype("float64")])
		filtered_cols = src_data[needed_cols]
		col_types = filtered_cols.columns.to_series().groupby(filtered_cols.dtypes).groups
		print("New data set columns:")
		for dtype, cols in col_types.items():
			print("\t", dtype, ":", len(cols))
		needed_cols_frame = pd.Series(needed_cols, index=range(len(needed_cols))).to_frame() 
		needed_cols_frame.columns = ['VAR']
		print(needed_cols_frame)
		needed_cols_frame.to_csv('.'.join(scr_fname_str.split('.')[:-1] + ['needed_cols', 'csv']), header=True, index_label = 'ID')


		filtered_cols_nandrop, high_nan_ratios = smart_nan_drop(filtered_cols, needed_cols, total_rows, nan_column_drop_threshold)
		

		normalized_filtered_cols, minmax = normalize_cols(filtered_cols, needed_cols)
		normalized_filtered_cols.fillna(value=-1, inplace=True)

		normalized_filtered_cols_nandrop, minmax_nandrop = normalize_cols(filtered_cols_nandrop, filtered_cols_nandrop.columns)


		filtered_cols_nandrop.to_csv('.'.join(scr_fname_str.split('.')[:-1] + ['filtered', 'nandrop', 'csv']), header=False)
		high_nan_ratios.to_csv('.'.join(scr_fname_str.split('.')[:-1] + ['high_nan_ratios', 'nandrop', 'csv']), header=True, index_label = 'VAR')
		
		normalized_filtered_cols_nandrop.to_csv('.'.join(scr_fname_str.split('.')[:-1] + ['filtered', 'nandrop', 'normalized', 'csv']), header=False)
		minmax_nandrop.to_csv('.'.join(scr_fname_str.split('.')[:-1] + ['minmax', 'nandrop', 'csv']), header=True, index_label='VAR')

		normalized_filtered_cols.to_csv('.'.join(scr_fname_str.split('.')[:-1] + ['filtered', 'normalized', 'csv']), header=False)
		minmax.to_csv('.'.join(scr_fname_str.split('.')[:-1] + ['minmax', 'csv']), header=True, index_label='VAR')
		


if __name__ == "__main__":
	main()
