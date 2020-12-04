# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 18:17:18 2020

@author: Hector
"""

def rotate(li, n):
    new = [0] * len(li)
    for i,item in enumerate(li):
        new[(i + n) % len(li)] = item
    return new