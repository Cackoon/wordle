"""
Take the list of wordle words and convert to a numpy array.


This is currently really ugly. I would like to conver the unicode or bytes
to hex value directly, but this isn't too slow.

Also, this doesn't save as much memory as I thought. Could further optomise by using 5 bits instead of 8? But not sure how to do that.
"""

from string import ascii_lowercase
import numpy as np

INPUT_FILEPATH = r"filestore/words.txt"
OUTPUT_FILEPATH = r"filestore/words.npy"

a = np.genfromtxt(
    INPUT_FILEPATH,
    delimiter=[1,1,1,1,1],
    autostrip=True,
    dtype='U'
)

d = np.zeros(a.shape, dtype=np.uint8)

for i, row in enumerate(a):
    for j, c in enumerate(row):
        for u, c1 in enumerate(ascii_lowercase):
            if c1 == c:
                d[i, j] = u

np.save(OUTPUT_FILEPATH, d)