import lampld
from admix.data import read_int_mat
import numpy as np
from os.path import join
data_dir = "/Users/kangchenghou/work/LAMP-LD/test_data/ten_region/"

pos = np.loadtxt(join(data_dir, "pos.txt"))
admix_hap = read_int_mat(join(data_dir, "admix.hap"))
ref_haps = [
    read_int_mat(join(data_dir, "EUR.hap")),
    read_int_mat(join(data_dir, "AFR.hap"))
]

model = lampld.LampLD(n_snp = len(pos), n_anc = len(ref_haps), n_proto=4, window_size=300)
model.set_pos(pos)

model.fit(ref_haps)

inferred_lanc = model.infer_lanc(admix_hap)