import os

data_folder = '../family_lexicon_eeg/'
data_folder = '/import/cogsci/andrea/dataset/neuroscience/family_lexicon_eeg/'

message = lambda item : 'python3 main.py '\
                        '--analysis {} '\
                        '--mapping_model {} '\
                        '--input_target_model {} '\
                        '--semantic_category_one {} '\
                        '--semantic_category_two {} '\
                        '--data_folder {} '\
                        '{}'.format(
                                    item[0], 
                                    item[1], 
                                    item[2], 
                                    item[3], 
                                    item[4], 
                                    data_folder, 
                                    correc
                                    )

models = [
          'coarse_category',
          'famous_familiar',
          'word_length',
          'orthography',
          ]

mappings = [
            'rsa'
            ]
corrections = [
               ' --corrected', 
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
            'time_resolved', 
            'searchlight',
            ]

### regular
already_done = list()

for analysis in analyses:
    for model in models:
        for mapping in mappings:
            for correc in corrections:
                for cat in categories:
                    for category_two in categories_two:
                        comb = sorted([cat, category_two])
                        if comb in already_done:
                            pass
                        else:
                            already_done.append(comb)

                        current_message = message([analysis, mapping, model, cat, category_two])
                        for plot in plots:
                            os.system('{}{}'.format(current_message, plot))
                            #os.system('{}{} --debugging'.format(current_message, plot))
## comparisons
already_done = list()

for analysis in analyses:
    for model in models:
        for mapping in mappings:
            for correc in corrections:
                for cat in categories:
                    for category_two in categories_two:
                        comb = sorted([cat, category_two])
                        if comb in already_done:
                            pass
                        else:
                            already_done.append(comb)

                        current_message = message([analysis, mapping, model, cat, category_two, correc])
                        if analysis == 'searchlight' and cat == category_two and cat == 'all':
                            os.system('{} --comparison --plot'.format(current_message))
