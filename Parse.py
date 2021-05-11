# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:51:56 2021

@author: HuyHoang
"""

import argparse

#Construct argument 
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True, help="name of the user")
args = vars(ap.parse_args())

#Display
print("Hi there {}, it's nice to meet you".format(args["name"]))