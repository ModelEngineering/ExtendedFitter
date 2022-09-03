"""Creates Latin Cube Entries

Creates a CSV file whose rows are strips in the latin cube and whose
columns are random numbers within the strp.

"""

import fitterpp.constants as cn

import lhsmdu
import os
import pandas as pd
import numpy as np


NUM_PARAMETER = 500
NUM_LATINCUBE = 10
OUT_PATH = os.path.join(cn.DATA_DIR, "latin_cube.csv")


def make():
    """
    Constructs the table of random numbers.
    """
    samples = lhsmdu.sample(NUM_PARAMETER, NUM_LATINCUBE)
    df = pd.DataFrame(samples).transpose()
    df.to_csv(OUT_PATH, index=True)

def read():
    """
    Returns a dataframe of table numbers.
    """
    df = pd.read_csv(OUT_PATH)
    del_columns = [c for c in df.columns if "Unnamed" in str(c)]
    for column in del_columns:
        del df[column]
    df.index = range(1, len(df.index) + 1)
    df.index.name = "strip"
    return df


if __name__ == '__main__':
    make()
    print("***Latin cube written to %s" % OUT_PATH)
