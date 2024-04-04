# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:18:14 2024

@author: Carlos G Martin
"""

# Creating the yaml file.

import yaml

# Define the data to be written to the YAML file
data_yaml = dict(
    train ='../../datasets/train',
    val ='../../datasets/valid',
    test='../../datasets/test',
    nc =1,
    names =['drone']
)
def write():    
    # Write the data to the YAML file
    with open('data.yaml', 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)

def readPrint():
    # Read and print the contents of the YAML file
    with open('data.yaml', 'r') as infile:
        print(infile.read())
    
if __name__ == "__main__":
    #write()
    readPrint()