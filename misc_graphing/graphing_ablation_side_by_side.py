import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.ticker as mtick

ZERO_SHOT_FLAN = {
             'MedNLI': 0.808,
             'RadQA': 0.602,  # F1
             'CLIP': 0.178,  # Macro
            }

ZERO_SHOT = {
            'MedNLI': 0.805,
            'RadQA': 0.619,
            'CLIP': 0.146
        }

SCORES = {
            'MedNLI': 'Accuracy',
            'RadQA': 'F1 Score',
            'CLIP': 'Macro F1'
        }

XLABEL = {
        'CLIP': '# of Discharge Summaries',
        'MedNLI': '# of Sentences',
        'RadQA': '# of Questions',

        }

DATASET_SIZES = {
                'CLIP': 518, # 38327 * 2,
                'MedNLI': 11232,
                'RadQA': 4878,
                }

MARKERS = {
            'PubMedGPT': 'D',
            'GatorTron': '*',
            'RoBERTa': 'o',
            'RoBERTa-large-PM-M3-Voc': 'P',
            'Clinical-T5-Large': 'h'
        }

COLORS = {
            'PubMedGPT': 'blue',
            'GatorTron': 'purple',
            'RoBERTa': 'green',
            'RoBERTa-large-PM-M3-Voc': 'orange',
            'Clinical-T5-Large': 'red'
        }

LABELS = {
            'PubMedGPT': 'PubMedGPT',
            'GatorTron': 'GatorTron',
            'RoBERTa': 'RoBERTa',
            'RoBERTa-large-PM-M3-Voc': 'BioClinRoBERTa',
            'Clinical-T5-Large': 'Clinical-T5-Large'
        }

def parse_std(str_: str) -> (float, float):
    """Parse a given cell""" 
    split_txt = str_.split('+/-')
    return float(split_txt[0]), float(split_txt[1])


def parse_single_row(single_row: list[str]) -> (list[float], list[float]):
    """Parse a single row of our CSV. """
    means, stdvs = [], []
    for s in single_row:
        m, std = parse_std(s) 
        means.append(m)
        stdvs.append(std)

    return means, stdvs


def load_data(filepath: str, task: str) -> dict:
    """Load the data. """
    df = pd.read_csv(filepath, index_col=0)

    if task == 'RadQA':
        df = df.filter(items=['RadQA 1% (F1)', 'RadQA 5% (F1)', 'RadQA 10% (F1)', 'RadQA 25% (F1)', 'RadQA 100% (F1)'])

    if task == 'CLIP':
        to_keep = ['CLIP 1% (Macro)', 'CLIP 5% (Macro)', 'CLIP 10% (Macro)', 'CLIP 25% (Macro)', 'CLIP 100% (Macro)']
        df = df.filter(items=to_keep)

    # Now let's parse into floats and return 
    rows = {}
    for i in range(len(df)):
        rows[df.iloc[i].name] = parse_single_row(df.iloc[i].values)

    return rows

def plot_single_graph(rows: dict, task: str, ax):
    # plot lines w/ data
    dataset_splits = [int(i * DATASET_SIZES[task]) for i in [0.01, 0.05, 0.1, 0.25, 1]]

    # lines
    for i, k in enumerate(rows):
        y = dataset_splits[i]
        #ax.errorbar(dataset_splits, rows[k][0], yerr=rows[k][1], color=COLORS[k], fmt=MARKERS[k], ecolor=COLORS[k])
        ax.errorbar(dataset_splits, rows[k][0], yerr=0, color=COLORS[k], fmt=MARKERS[k], ecolor=COLORS[k])
        ax.plot(dataset_splits, rows[k][0], marker=MARKERS[k], color=COLORS[k], label=LABELS[k])

    zsp = ZERO_SHOT[task] 
    ax.axhline(y=zsp, linewidth=1, color='magenta', linestyle="dashed", label='GPT-3 Few-Shot')

    zsp = ZERO_SHOT_FLAN[task]
    ax.axhline(y=zsp, linewidth=1, color='gray', linestyle="dashed", label='Flan-T5 Few-Shot')
    #trans = transforms.blended_transform_factory(
    #    ax.get_yticklabels()[0].get_transform(), ax.transData
    #)

    #ax.text(1.05, zsp) #, "GPT-3\nFew-Shot ({})".format(zsp),
    #transform=trans, ha="left", va="center", fontsize=12,)

    if task == 'CLIP':
        ax.legend(loc="upper left", bbox_to_anchor=(1,1))

    #ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xticks(dataset_splits)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for label in ax.get_xticklabels():
        label.set_rotation(90)

    ax.set_ylabel(SCORES[task], fontsize=12, fontname="sans-serif")
    #if task != 'MedNLI':
    #    ax.set_xlabel(XLABEL[task], labelpad=5)
    #else:
    ax.set_xlabel(XLABEL[task], fontsize=12, fontname="sans-serif")
    ax.set_title(task, fontsize=12, fontname="sans-serif")
    
    #ax.set_xticks(rotation=90) #'Performance')
    #plt.xticks(rotation=90)

    #plt.xlabel("# of Training Samples")
    #plt.ylabel(SCORES[task])

    #plt.savefig(f'ablation_{task}.png', dpi=1000, bbox_inches='tight')
    #plt.savefig(f'ablation_{task}.pdf', dpi=1000, bbox_inches='tight')


if __name__ == '__main__':
    tasks = ['MedNLI', 'RadQA', 'CLIP']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)) #, sharex=True)

    for i, t in enumerate(tasks):
        df = load_data(f'data/{t}_ablation_final.csv', task=t)
        plot_single_graph(df, t, axes[i])
    
    fig.align_labels()
    plt.xticks(rotation=90)
    #plt.show()
    plt.savefig('ablation_all_no_error.pdf', bbox_inches='tight')  #.png, dpi=1000, bbox_inches='tight')
    
