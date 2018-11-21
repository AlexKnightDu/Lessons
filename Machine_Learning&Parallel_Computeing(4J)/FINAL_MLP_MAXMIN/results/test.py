#!/usr/bin/env python
# -*- coding: utf-8 -*-

data = []
for i in range(0,100000):
    for j in range(0,1000000):
        for k in range(0,1000000):
            data += [(((i * j) + k * (j - i)))]

