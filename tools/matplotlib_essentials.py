import matplotlib as matplotlib
import matplotlib.pyplot as plt

# from matplotlib import font_manager

plt.style.use('bmh')

# Type 2/TrueType fonts.
matplotlib.rcParams.update({'pdf.fonttype': 42})
matplotlib.rcParams.update({'ps.fonttype': 42})

matplotlib.rcParams.update({'font.family': 'sans-serif'})
matplotlib.rcParams.update({'font.sans-serif': ['Arial']})

matplotlib.rcParams.update({'axes.titlesize': 8})
matplotlib.rcParams.update({'axes.labelsize': 7})
matplotlib.rcParams.update({'xtick.labelsize': 7})
matplotlib.rcParams.update({'ytick.labelsize': 7})
matplotlib.rcParams.update({'legend.fontsize': 7})

# logger.info(font_manager.OSXInstalledFonts())
# logger.info(font_manager.OSXFontDirectories)
