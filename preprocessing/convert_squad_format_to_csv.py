import json
import pandas as pd

def convert_squad_to_csv_fmt(jf: dict, out_f: str):
    """Convert file to csv format. """
    new_data = {'id': [], 'title': [], 'context': [], 'question': [], 'answers': []}    

    data = jf['data']
    for note in data:
        title = note['title']
        for para in note['paragraphs']:
            context = para['context']

            for qas in para['qas']:
                new_data['id'].append(qas['id'])
                new_data['title'].append(title)
                new_data['context'].append(context)
                new_data['question'].append(qas['question'])
                
                if qas['is_impossible']:
                    new_data['answers'].append({'text': [], 'answer_start': []})
                    continue 
                
    
                to_add = {'text': [], 'answer_start': []}
                for a in qas['answers']:
                    to_add['text'].append(a['text'])
                    to_add['answer_start'].append(a['answer_start'])
    
                new_data['answers'].append(to_add)
    
    csv = pd.DataFrame(new_data)
    csv.to_csv(out_f, index=False)

if __name__ == '__main__':
    pairs = [('data/ids_to_ignore/radqa_ids/radqa_train.json', 'data/radqa_data/train.csv'), 
             ('data/ids_to_ignore/radqa_ids/radqa_dev.json', 'data/radqa_data/dev.csv'), 
             ('data/ids_to_ignore/radqa_ids/radqa_test.json', 'data/radqa_data/test.csv')]
    
    for p in pairs:
        jf = json.load(open(p[0]))
        convert_squad_to_csv_fmt(jf, p[1])

    
