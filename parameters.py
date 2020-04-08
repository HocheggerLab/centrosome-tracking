import os

base_dir = "."
compiled_data_dir = os.path.join(base_dir, "compiled")
out_dir = os.path.join(base_dir, "scripts_output")
os.makedirs(base_dir, exist_ok=True)
os.makedirs(compiled_data_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

dpi = 300
size_A3 = (11.7, 16.5)
line_err_kws = {'alpha': 0.3, 'lw': 1}
