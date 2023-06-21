import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

IS_CLINICAL = ['ClinicalBERT', 'ClinicalLongFormer', 'Clinical-T5-Sci', 'Clinical-T5-Base', 'Clinical-T5-Scratch', 'BioClinRoBERTa', 'GatorTron', 'Clinical-T5-Large', 'BioClinRoBERTa-base']
IS_BIOMEDICAL = ['SciFive', 'SciFive Large']

renamed = {
            'Clinical-T5-Scratch': 'Clinical-T5-Base',
            'PubMed GPT': 'PubMedGPT',
            'T5-3B': 'T5-XL'
          }

new_positions = {
        'MedNLI (Accuracy)': {'GatorTron': (-54, -3), 'BioClinRoBERTa': (-82, -3), 'PubMedGPT': (-64, -3), 'RoBERTa': (-48, -3), 'T5-Large': (-49, -3), 'Clinical-T5-Large': (-2, 4), 'Clinical-T5-Base': (5, -4), 'T5-XL': (-3, -13)},
        'RadQA (F1)': {'GatorTron': (-54, -3), 'BioClinRoBERTa': (-72, -15), 'PubMedGPT': (-64, -3), 'RoBERTa': (-48, -3), 'T5-Large': (-49, -3)},
        'CLIP (Macro)': {'GatorTron': (-54, -3), 'BioClinRoBERTa': (-82, -3), 'PubMedGPT': (-64, -3), 'RoBERTa': (-48, -3), 'T5-Large': (-49, -3), 'Clinical-T5-Base': (-2, -12)},
}



def label_point(df, x, y, ax, task):
    """Label a specific point on a plot.""" 
    for i, point in df.iterrows():
        name = point['Model']
        if name in renamed:
            name = renamed[name]


        offset = (5, -3)
        if name in new_positions[task]:
            offset = new_positions[task][name]

        ax.annotate(name, xy = (point[x], point[y]),
                    xytext=offset, textcoords="offset points")


def scatter_text_jonas(data):
    colors = sns.color_palette()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    
    # Do the first one 
    x, hue = 'Log Total FLOPs', 'Model Type'
    ys = ['MedNLI (Accuracy)', 'RadQA (F1)', 'CLIP (Macro)']
    vals = set(df[hue].values)
    
    for graph_i, y in enumerate(ys):
        for i, v in enumerate(vals):
            label = None
            if y == 'CLIP (Macro)':
                label = 'Clinical' if 'Clinical' in v else 'Non-Clinical'


            if 'T5' in v:
                sns.regplot(x=x, 
                            y=y, 
                            data=df[df[hue] == v], 
                            color=colors[1 if 'Clinical' in v else 0],  
                            marker='o', 
                            ax=ax.flat[graph_i], 
                            scatter_kws={"s": 25, "alpha": 1}, 
                            ci=None,
                            label=label) 
                            #label='Clinical' if 'Clinical' in v else 'General')

            else:
                sns.scatterplot(x=x,
                                y=y,
                                data=df[df[hue] == v],
                                color=colors[1 if 'Clinical' in v else 0],
                                marker='o',
                                ax=ax.flat[graph_i],
                                ci=None, 
                                label=label)
                                #label='Clinical' if 'Clinical' in v else 'General')

        label_point(df, x, y, ax.flat[graph_i], task=y)
        if y == 'CLIP (Macro)':
            ax.flat[graph_i].get_legend().remove()

        ax.flat[graph_i].spines['right'].set_visible(False)
        ax.flat[graph_i].spines['top'].set_visible(False)
        ax.flat[graph_i].set_ylabel(y, fontsize=12)
        ax.flat[graph_i].set_xlabel('Log Total FLOPs', fontsize=12)
   
    #plt.legend()
    #plt.tight_layout()
    #plt.legend(bbox_to_anchor=(1.04, 1), loc='upper right', borderaxespad=0)
    h, l = ax.flat[graph_i].get_legend_handles_labels()
    plt.legend(handles=[item for item in h[2:]], labels= [item for item in l[2:]]) #, bbox_to_anchor=(1.04, 1), loc="upper left")


    plt.savefig('visualize_flops_legend_inside.pdf',bbox_inches='tight')
    #plt.show()
    

def scatter_text_clip_only(data):
    """Scatter plot with country codes on the x y coordinates
       Based on this answer: https://stackoverflow.com/a/54789170/2641825"""

    colors = sns.color_palette()
    fig, ax = plt.subplots() # 1, 2, figsize=(10, 5), sharex=True) #, sharey=True)

    # Do the first one 
    x, y, hue = 'Log Total FLOPs', 'CLIP (Macro)', 'Clinical Model'
    vals = set(df[hue].values)
    for i, v in enumerate(vals):
        sns.regplot(x=x, y=y, data=df[df[hue] == v], color=colors[i],
                    scatter_kws={"s": 25, "alpha": 1}, ci=None, label='Clinical Model' if v else 'General Model')

        label_point(df, x, y, ax)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylabel('CLIP (Macro F1)')

    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    plt.savefig('clip_only.pdf',bbox_inches='tight')



def scatter_text(data):
    """Scatter plot with country codes on the x y coordinates
       Based on this answer: https://stackoverflow.com/a/54789170/2641825"""
    colors = sns.color_palette()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True) #, sharey=True)

    # Do the first one 
    x, y, hue = 'Log Total FLOPs', 'CLIP (Macro)', 'Clinical Model'
    vals = set(df[hue].values)
    for i, v in enumerate(vals):
        sns.regplot(x=x, y=y, data=df[df[hue] == v], ax=axes.flat[0], color=colors[i],
                    scatter_kws={"s": 25, "alpha": 1}, ci=None, label='Clinical Model' if v else 'General Model')
        
        label_point(df, x, y, axes.flat[0])
    
    # Do the second one
    x, y, hue = 'Log Total FLOPs', 'MedNLI (Accuracy)', 'Clinical Model'
    vals = set(df[hue].values)
    for i, v in enumerate(vals):
        sns.regplot(x=x, y=y, data=df[df[hue] == v], ax=axes.flat[1], color=colors[i],
                    scatter_kws={"s": 25, "alpha": 1}, ci=None, label='Clinical Model' if v else 'General/Bio Model')

        label_point(df, x, y, axes.flat[1])

    # 'Log Total FLOPs', 'CLIP (Macro)', 'Clinical Model', 'Model'
    axes.flat[0].spines['right'].set_visible(False)
    axes.flat[0].spines['top'].set_visible(False)
    axes.flat[0].set_ylabel('CLIP (Macro F1)')

    axes.flat[1].spines['right'].set_visible(False)
    axes.flat[1].spines['top'].set_visible(False)
    #axes.flat[1].spines['left'].set_visible(False)
    #axes.flat[1].yaxis.set_visible(False) # same for y axis.
    #plt.ylim([0.55, 1])
    #axes.flat[0].set_ylabel('Performance')

    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    #plt.legend(bbox_to_anchor=(1.04, 1), loc="lower left")
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.savefig('testing123.png', dpi=500, bbox_inches='tight')

def graph_columns(df, column_x: str, column_y: str, hue: str, fig_name: str, folder: str = 'flop_figs/'):
    sns.set_style('whitegrid')
    sns.lmplot(x=column_x, y =column_y, data =df, hue = hue, markers =['o', 'v'], ci=None)
    plt.savefig(folder + fig_name, dpi=1000)
    #plt.show()

def load_data():
    ignore_list = ['PubMed GPT', 'Clinical-T5-Sci']
    df = pd.read_csv('data/NUM_FLOPS.csv')
    df = df[~(df['Model'].isin(ignore_list))]
    
    df['In Domain'] = [model_name in IS_CLINICAL or model_name in IS_BIOMEDICAL for model_name in df.Model]
    df['Clinical Model'] = [model_name in IS_CLINICAL for model_name in df.Model]
    df['Log Total FLOPs'] = np.log(df['Total FLOPS'])
    return df

def load_scratch_data_only():
    df = pd.read_csv('data/NUM_FLOPS.csv')
    #df = pd.read_csv('data/FLOPS_w_extra.csv')
    ignore_list = ['ClinicalBERT', 'ClinicalLongFormer', 'Clinical-T5-Sci', 'Clinical-T5-Base', 'SciFive', 'SciFive Large']
    #ignore_list.append('PubMed GPT')
    
    df = df[~(df['Model'].isin(ignore_list))]
    
    df['In Domain'] = [model_name in IS_CLINICAL or model_name in IS_BIOMEDICAL for model_name in df.Model]
    df['Clinical Model'] = [model_name in IS_CLINICAL for model_name in df.Model]
    df['Log Total FLOPs'] = np.log(df['Total FLOPS'])
    return df


def load_scratch_data_more_complex():
    df = pd.read_csv('data/NUM_FLOPS.csv')
    ignore_list = ['ClinicalBERT', 'ClinicalLongFormer', 'Clinical-T5-Sci', 'Clinical-T5-Base', 'SciFive', 'SciFive Large']
    df = df[~(df['Model'].isin(ignore_list))]

    # 'Clinical-T5-Scratch', 'BioClinRoBERTa', 'GatorTron', 'Clinical-T5-Large', 'BioClinRoBERTa-base']
    mapper = {
              'Clinical-T5-Scratch': 'Clinical T5', 
              'T5-Base': 'T5', 
              'T5-Large': 'T5', 
              'T5-3B': 'T5', 
              'PubMed GPT': 'General Model', 
              'RoBERTa': 'General Model',
              'Clinical-T5-Large': 'Clinical T5',
              'GatorTron': 'Clinical Model',
              'BioClinRoBERTa': 'Clinical Model'
              }

    df['Log Total FLOPs'] = np.log(df['Total FLOPS'])
    df['Model Type'] = [mapper[x] for x in df['Model'].values]
    return df


if __name__ == '__main__':
    df = load_data()

    """
    # Graphing the simple ones. 
    graph_columns(df, 'Log Total FLOPs', 'MedNLI (Accuracy)', 'In Domain', 'mednli_log_flops_biomed_clin.png')
    graph_columns(df, 'Log Total FLOPs', 'MedNLI (Accuracy)', 'Clinical Model', 'mednli_log_flops_clin_only.png')
    graph_columns(df, 'Log Total FLOPs', 'RadQA (F1)', 'In Domain', 'radqa_log_flops_biomed_clin.png')
    graph_columns(df, 'Log Total FLOPs', 'RadQA (F1)', 'Clinical Model', 'radqa_log_flops_clin_only.png')
    graph_columns(df, 'Log Total FLOPs', 'CLIP (Micro)', 'In Domain', 'clip_micro_log_flops_biomed_clin.png')
    graph_columns(df, 'Log Total FLOPs', 'CLIP (Micro)', 'Clinical Model', 'clip_micro_log_flops_clin_only.png')
    graph_columns(df, 'Log Total FLOPs', 'CLIP (Macro)', 'In Domain', 'clip_maacro_log_flops_biomed_clin.png')
    graph_columns(df, 'Log Total FLOPs', 'CLIP (Macro)', 'Clinical Model', 'clip_macro_log_flops_clin_only.png')
    """
   
    """
    df = load_scratch_data_only()
    graph_columns(df, 'Log Total FLOPs', 'MedNLI (Accuracy)', 'In Domain', 'mednli_log_flops_biomed_clin.png', 'filtered_flops/')
    graph_columns(df, 'Log Total FLOPs', 'MedNLI (Accuracy)', 'Clinical Model', 'mednli_log_flops_clin_only.png', 'filtered_flops/')
    graph_columns(df, 'Log Total FLOPs', 'RadQA (Exact)', 'In Domain', 'radqa_exact_log_flops_biomed_clin.png', 'filtered_flops/')
    graph_columns(df, 'Log Total FLOPs', 'RadQA (Exact)', 'Clinical Model', 'radqa_exact_log_flops_clin_only.png', 'filtered_flops/')
    graph_columns(df, 'Log Total FLOPs', 'RadQA (F1)', 'In Domain', 'radqa_log_flops_biomed_clin.png', 'filtered_flops/')
    graph_columns(df, 'Log Total FLOPs', 'RadQA (F1)', 'Clinical Model', 'radqa_log_flops_clin_only.png', 'filtered_flops/')
    graph_columns(df, 'Log Total FLOPs', 'CLIP (Micro)', 'In Domain', 'clip_micro_log_flops_biomed_clin.png', 'filtered_flops/')
    graph_columns(df, 'Log Total FLOPs', 'CLIP (Micro)', 'Clinical Model', 'clip_micro_log_flops_clin_only.png', 'filtered_flops/')
    graph_columns(df, 'Log Total FLOPs', 'CLIP (Macro)', 'In Domain', 'clip_maacro_log_flops_biomed_clin.png', 'filtered_flops/')
    graph_columns(df, 'Log Total FLOPs', 'CLIP (Macro)', 'Clinical Model', 'clip_macro_log_flops_clin_only.png', 'filtered_flops/')
    """


    #scatter_text(x, y, hue text_column, data, title, xlabel, ylabel)
    #df = load_scratch_data_only()
    #scatter_text(df)
    #scatter_text_clip_only(df)

    df = load_scratch_data_more_complex()
    scatter_text_jonas(df)
    #scatter_text('Log Total FLOPs', 'CLIP (Macro)', 'Clinical Model', 'Model', df, '', '', '')
