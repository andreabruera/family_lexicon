import os

message = lambda item : 'python3 main.py --analysis {} --mapping_model {} --mapping_direction {} --input_target_model {} --experiment_id two --temporal_resolution 5 --semantic_category_one {} --semantic_category_two {} --data_kind erp --data_folder /import/cogsci/andrea/dataset/neuroscience/family_lexicon_eeg/ --searchlight_spatial_radius large_distance --searchlight_temporal_radius large --language {} --evaluation_method {} --average 24{}'.format(item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8])

lang_agnostic = [
          'famous_familiar',
          'coarse_category',
          'fine_category',
          'orthography',
          'familiarity',
          'imageability',
          'word_length',
          'individuals',
          'gender',
          'occupation',
          'place_type',
          'location',
          ]
models = [
          'famous_familiar',
          'coarse_category',
          ]

analyses = [
            'time_resolved',
            'searchlight',
            ]

languages = [
             'it', 
             ]
mappings = [
            #'ridge', 
            'rsa'
            #'support_vector',
            ]
corrections = [
               ' --corrected', 
               #''
               ]
categories = [
              'all',
              'place', 
              'person', 
              ]
categories_two = [
                  'all',
                  'familiar', 
                  'famous', 
                  ]
plots = [
         ' ', 
         ' --plot'
         ]

eval_methods = [
                'pairwise',
                #'correlation',
                ]

directions = [
              #'encoding',
              'decoding',
              ]

already_done = list()

for direction in directions:
    for analysis in analyses:
        for eval_method in eval_methods:
            for model in models:
                for lang in languages:
                    if ('IT' in model or 'xlm' in model) and lang == 'en':
                        continue
                    if ('BERT' in model or 'gpt2' in model) and lang == 'it':
                        continue
                    if model in lang_agnostic and lang == 'en':
                        continue
                    for mapping in mappings:
                        for correc in corrections:
                            for cat in categories:
                                for category_two in categories_two:
                                    #if cat == category_two:
                                    #    continue
                                    comb = sorted([cat, category_two])
                                    if comb in already_done:
                                        #continue
                                        pass
                                    else:
                                        already_done.append(comb)

                                    current_message = message([analysis, mapping, direction, model, cat, category_two, lang, eval_method, correc])
                                    print(current_message)
                                    for plot in plots:
                                        os.system('{}{}'.format(current_message, plot))
