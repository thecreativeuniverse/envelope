import numpy as np

DATA_SEPARATION = 30 / (60 * 24) # 30 minutes converted to days
INPUT_SIZE = int(np.floor(21 / DATA_SEPARATION)) # 3 week time frame per input