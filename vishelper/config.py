import seaborn as sns

formatting = {'font.size': 16,
              'tick.labelsize': 14,
              'tick.size': 10,
              'markersize': 48,
              'figsize': [12.0, 8.0],
              'figure.figsize': [12.0, 8.0],
              'color.single': '#0067a0',
              'color.darks': ['#0067a0', '#53565a', '#009681', '#87189d', '#c964cf'],
              'color.mediums': ['#0085ca', '#888b8d', '#00c389', '#f4364c', '#e56db1'],
              'color.lights': ['#00aec7', '#b1b3b3', '#2cd5c4', '#ff671f', '#ff9e1b'],
              'color.all': ['#0067a0', '#53565a', '#009681', '#87189d', '#c964cf', '#0085ca', '#888b8d',
                            '#00c389', '#f4364c', '#e56db1', '#00aec7', '#b1b3b3', '#2cd5c4', '#ff671f',
                            '#ff9e1b'],
              'greens': ['#43b02a', '#78be20', '#97d700'],
              'axes.labelsize': 16,
              'axes.labelcolor': '#53565a',
              'axes.titlesize': 20,
              'lines.color': '#53565a',
              'lines.linewidth': 3,
              'legend.fontsize': 14,
              'legend.location': 'best',
              'legend.marker': 's',
              'text.color': '#53565a',
              'alpha.single': 0.8,
              'alpha.multiple': 0.7,
              'suptitle.x': 0.5,
              'suptitle.y': 1.025,
              'suptitle.size': 24}

cmaps = {'diverging': sns.diverging_palette(244.4, 336.7, s=71.2, l=41.6, n=20),
         'heatmap': sns.cubehelix_palette(8, start=.5, rot=-.75),
         'blues': sns.light_palette(formatting['color.single']),
         'reds': sns.light_palette(formatting['color.mediums'][4]),
         'teals': sns.light_palette(formatting["color.darks"][2]),
         'purples': sns.light_palette(formatting['color.darks'][4])}

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
map_to_days = dict(zip(range(7), days_of_week))