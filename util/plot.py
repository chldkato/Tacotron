import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
matplotlib.rc('font', family=font_name, size=14)


def plot_alignment(alignment, path, text, info=None, istrain=0):
  text = text.rstrip('_').rstrip('~')
  if istrain:
    alignment = alignment[:len(text)]
  else:
    alignment = alignment[:len(text), :len(alignment[1]) // 2]
  _, ax = plt.subplots(figsize=(len(text)/3, 5))
  ax.imshow(
    alignment.T,
    aspect='auto',
    origin='lower',
    interpolation='none')
  xlabel = 'Encoder timestep'
  if info is not None:
    xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Decoder timestep')
  text = [x if x != ' ' else '' for x in list(text)]
  plt.xticks(range(len(text)), text)
  plt.tight_layout()
  plt.savefig(path, format='png')
