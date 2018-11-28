#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main function
"""
from lib.common import print_start
from lib.gen.generator import GenVin
from net_work import NetWork
from lib.gen.static import *

if __name__=='__main__':
    print_start()
    width = 49
    height = 258
    label_size = 17
    classify_size = len(chars)
    generator = GenVin(width, height)
    network = NetWork(width, height, label_size, classify_size, generator)
    network.start(train=True)



