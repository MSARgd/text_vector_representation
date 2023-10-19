#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:28:33 2023

@author: msa
"""
import random
import string

def spaced_text(text):
    txt_spaced = ''.join([c + ' ' * random.randint(0, 1) for c in text])
    return txt_spaced



# Create a WhitespaceTokenizer object
# tokenizer = WhitespaceTokenizer()

# Tokenize a string using the tokenizer's `tokenize` method
# text = "aaa bb  gdfdsjg  sdklf   dkfhhds"
# t_s_p_tokenized = tokenizer.tokenize(text)

# print(t_s_p_tokenized)