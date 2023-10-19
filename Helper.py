import random
import string

def spaced_text(text):
    txt_spaced = ''.join([c + ' ' * random.randint(0, 1) for c in text])
    return txt_spaced



