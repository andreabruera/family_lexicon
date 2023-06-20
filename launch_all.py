import os

#message = lambda item : 'python3 main.py --analysis searchlight --mapping_model {} --mapping_direction decoding --input_target_model {} --experiment_id two --temporal_resolution 5 --semantic_category_one {} --semantic_category_two {} --data_kind erp --data_folder /import/cogsci/andrea/dataset/neuroscience/family_lexicon_eeg/ --searchlight_spatial_radius large_distance --searchlight_temporal_radius large --language {} --evaluation_method pairwise --average 24{}'.format(item[0], item[1], item[2], item[3], item[4], item[5])
message = lambda item : 'python3 main.py --analysis searchlight --mapping_model {} --mapping_direction decoding --input_target_model {} --experiment_id one --temporal_resolution 5 --semantic_category_one {} --semantic_category_two {} --data_kind erp --data_folder /import/cogsci/andrea/dataset/neuroscience/exploring_individual_entities_eeg --searchlight_spatial_radius large_distance --searchlight_temporal_radius large --language {} --evaluation_method pairwise --average 24{}'.format(item[0], item[1], item[2], item[3], item[4], item[5])

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
          #'transe',
          ]
models = [
          #'coarse_category',
          #'fine_category',
          'w2v',
          #'xlm-roberta-large_individuals',
          #'wikipedia2vec',
          #'transe',
          #'BERT_large_individuals',
          #'word_length',
          #'log_frequency',
          #'orthography',

          #'wikipedia2vec_sentence_individuals',
          #'affective_individuals',
          #'w2v_sentence_individuals',
          #'perceptual_individuals',
          #'valence_individuals',
          #'arousal_individuals',
          #'concreteness_individuals',
          #'imageability_individuals',
          #'ITGPT2_individuals',
          #'xlm-roberta-large_all',
          #'xlm-roberta-large_one',
          #'gpt2-large_individuals',
          #'ITGPT2_random',
          #'xlm-roberta-large_random',
          #'BERT_large_random',
          #'gpt2-large_random',
          #'ITGPT2_model_300-500ms',
          #'xlm-roberta-large_model_300-500ms',
          #'BERT_large_model_300-500ms',
          #'gpt2-large_model_300-500ms',
          #'sentence_lengths',
          #'famous_familiar',
          #'individuals',
          #'frequency',
          #'familiarity',
          #'imageability',
          #'gender',
          #'occupation',
          #'place_type',
          #'location',
          ]

languages = [
             'it', 
             #'en'
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
                  #'familiar', 
                  #'famous', 
                  #'all',
                  'individual',
                  #'category',
                  ]
#categories_two = [
#                  'all',
#                  'individual',
#                  ]
plots = [
         ' ', 
         ' --plot'
         ]

already_done = list()

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

                        current_message = message([mapping, model, cat, category_two, lang, correc])
                        for plot in plots:
                            os.system('{}{}'.format(current_message, plot))
                            #os.system('{}{} --cores_usage min'.format(current_message, plot))
                            #os.system('{}{} --comparison'.format(current_message, plot))
                            #os.system('{}{} --debugging'.format(current_message, plot))
