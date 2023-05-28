import os

message = lambda item : 'python3 main.py --analysis {} --mapping_model {} --mapping_direction encoding --input_target_model {} --experiment_id two --temporal_resolution 5 --semantic_category_one {} --semantic_category_two {} --data_kind erp --data_folder /import/cogsci/andrea/dataset/neuroscience/family_lexicon_eeg/ --searchlight_spatial_radius large_distance --searchlight_temporal_radius large --language {} --evaluation_method pairwise --average 24{}'.format(item[0], item[1], item[2], item[3], item[4], item[5], item[6])

models = [
          'coarse_category',
          'word_length',
          'orthography',
          'famous_familiar',
          ]

languages = [
             'it', 
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

analyses = ['time_resolved', 'searchlight']

already_done = list()

for analysis in analyses:
    for model in models:
        for lang in languages:
            for mapping in mappings:
                for correc in corrections:
                    for cat in categories:
                        for category_two in categories_two:
                            comb = sorted([cat, category_two])
                            if comb in already_done:
                                #continue
                                pass
                            else:
                                already_done.append(comb)

                            current_message = message([analysis, mapping, model, cat, category_two, lang, correc])
                            for plot in plots:
                                os.system('{}{}'.format(current_message, plot))
                                #os.system('{}{} --debugging'.format(current_message, plot))
                            os.system('{}{} --comparison'.format(current_message, plot))
