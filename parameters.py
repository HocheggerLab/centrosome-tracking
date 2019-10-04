import os
data_dir = "/Users/Fabio/data/lab/"
compiled_data_dir = "/Users/Fabio/data/lab/compiled/"
out_dir = data_dir + "scripts_output/"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(compiled_data_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

dpi = 300
size_A3 = (11.7, 16.5)
line_err_kws = {'alpha': 0.3, 'lw': 1}
