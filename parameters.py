import os

lab_dir = "/Users/Fabio/data/lab/"
compiled_data_dir = os.path.join(lab_dir, "compiled")
out_dir = os.path.join(lab_dir, "scripts_output")
os.makedirs(lab_dir, exist_ok=True)
os.makedirs(compiled_data_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

dpi = 300
size_A3 = (11.7, 16.5)
line_err_kws = {'alpha': 0.3, 'lw': 1}
