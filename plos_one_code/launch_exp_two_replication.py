import os

### EDIT HERE WITH YOUR PATH_TO_FOLDER
data_folder = '/data/tu_bruera/neuroscience/family_lexicon_eeg'
assert os.path.exists(data_folder)

message = lambda item : 'python3 main.py --analysis {} --input_target_model {} --semantic_category_one {} --semantic_category_two {} --data_folder {}'.format(item[0], item[1], item[2], item[3], item[4])

models = [
          'xlm-roberta-large',
          'w2v_sentence',
          #'word_length',
          #'orthography',
          ]

categories = [
              'place', 
              'person', 
              'all',
              ]
categories_two = [
                  'familiar', 
                  'famous', 
                  'all',
                  ]
plots = [
         ' ', 
         ' --plot'
         ]

analyses = [
            #'time_resolved', 
            'searchlight',
            ]

already_done = list()

for analysis in analyses:
    for model in models:
        for cat in categories:
            for category_two in categories_two:
                comb = sorted([model, analysis, cat, category_two])
                if 'all' in comb and len(set(comb))>3:
                    continue
                if comb in already_done:
                    continue
                    #pass
                else:
                    already_done.append(comb)

                current_message = message([analysis, model, cat, category_two, data_folder])
                for plot in plots:
                    os.system('{}{}'.format(current_message, plot))
                    #os.system('{}{} --debugging'.format(current_message, plot))
                if analysis == 'searchlight':
                    os.system('{} --comparison --plot'.format(current_message))
