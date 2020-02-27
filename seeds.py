import numpy as np

def gen_seeds(size: int = None) -> np.ndarray:
    max_uint32 = np.iinfo(np.uint32).max
    return np.random.randint(
            max_uint32+1, size=size, dtype=np.uint32)

quick_seeds = [2144199730, 794209841]

test_seeds = [2144199730, 794209841, 2985733717, 2282690970, 1901557222,
        2009332812, 2266730407, 635625077, 3538425002, 960893189,
        497096336, 3940842554, 3594628340, 948012117, 3305901371,
        3644534211, 2297033685, 4092258879, 2590091101, 1694925034]

development_seed = 4143496719
