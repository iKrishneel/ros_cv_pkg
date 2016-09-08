#!/usr/bin/env python

import numpy as np
import os
import glob
import sys

def read_directory_from_text(file_path):
    txtfile = open(file_path, 'r')
    txt_paths = []
    for line in txtfile:
        txt_paths.append(line.strip('\n'))
    txtfile.close()
    return txt_paths

def read_data_from_text(file_path):
    txtfile = open(file_path, 'r')
    data = []
    for icounter, line in enumerate(txtfile):
        feature = []
        for word in line.split(' '):
            if word:
                feature.append(float(word))
        data.append(feature)
    txtfile.close()
    return np.array(data)

def main(argv):
    data_dir = read_directory_from_text(argv[1])
    data = []
    for path in data_dir:
        d = read_data_from_text(path)
        data.append(d[0])
    print data
    np.savetxt(argv[2], data, fmt='%3.5f')
    

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print "USAGE: <path_to_data>, <name_of_save_file>"
        print "<path_to_data>: is the textfile with paths to other text files"
    else:
        main(sys.argv)
