#!/usr/bin/env python3

import pandas
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ifile", help="input CSV file", type=argparse.FileType('r'))
	parser.add_argument("--ofile", help="output CSV file", type=argparse.FileType('w'))
	args = parser.parse_args()
	src_fname = args.ifile
	dst_fname = args.ofile
	print("Source:", src_fname)
	print("Destination:", dst_fname)

if __name__ == "__main__":
	main()
