import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


untranscirbed_pairs = [1002, 23024, 29030, 31032, 33034, 61062, 71072, 73074, 83084, 87088, 89090, 93094, 103104, 105106, 113114, 115116, 125126, 129130, 133134, 147148, 149150, 151152, 155156, 157158]

def measure_distance(X, Y):
   X = set(X.split())
   Y = set(Y.split())
   # form a set containing keywords of both strings 
   rvector = X.union(Y) 
   l1 = [1 if w in X else 0 for w in rvector]
   l2 = [1 if w in Y else 0 for w in rvector]
   return dot(l1, l2)/(norm(l1)*norm(l2))

    
def measure_distance_per_word(a_words, b_words):
   a_words = a_words.split()
   b_words = b_words.split()
   return np.max([measure_distance(a, b) for a in a_words for b in b_words])

def word2vec_similarity_spacy(a, b):
    return a.similarity(b)
def word2vec_similarity_between_word(gen_model, a_words, b_words):
    return np.mean([gen_model.similarity(a, b) for a in a_words for b in b_words])
def word2vec_similarity_between_word_key(gen_model, a_words, b_words):
   try:
      return np.mean([gen_model.similarity(a, b) for a in a_words for b in b_words])
   except Exception as e:
      print(e)
      return 0
   #  return np.max([gen_model.similarity(a, b) for a in a_words for b in b_words])
 
 
def get_pre_post_names(naming_task, fribble_specific_exp_info):
   # rename Name_a and Name_b to pre_name_a and pre_name_b. Name_a_lemmas and Name_b_lemmas to pre_name_a_lemmas and pre_name_b_lemmas
   pre_names = naming_task[naming_task['Session'] == 'pre'][['Name_a', 'Name_b', 'Name_a_lemmas', 'Name_b_lemmas', 'Pair', 'Fribble_nr']]
   post_names = naming_task[naming_task['Session'] == 'post'][['Name_a', 'Name_b', 'Name_a_lemmas', 'Name_b_lemmas','Pair', 'Fribble_nr']]
   pre_names = pre_names.rename(columns={'Name_a':'pre_name_a', 'Name_b':'pre_name_b', 'Name_a_lemmas':'pre_name_a_lemmas', 'Name_b_lemmas':'pre_name_b_lemmas'})
   post_names = post_names.rename(columns={'Name_a':'post_name_a', 'Name_b':'post_name_b', 'Name_a_lemmas':'post_name_a_lemmas', 'Name_b_lemmas':'post_name_b_lemmas'})
   pre_post_names = pre_names
   pre_post_names['post_name_a_lemmas'] = post_names['post_name_a_lemmas'].to_list()
   pre_post_names['post_name_b_lemmas'] = post_names['post_name_b_lemmas'].to_list()
   pre_post_names['post_name_a'] = post_names['post_name_a'].to_list()
   pre_post_names['post_name_b'] = post_names['post_name_b'].to_list()
   # make all the names lowercase
   pre_post_names['pre_name_a'] = pre_post_names['pre_name_a'].apply(lambda x: x.lower())
   pre_post_names['pre_name_b'] = pre_post_names['pre_name_b'].apply(lambda x: x.lower())
   pre_post_names['post_name_a'] = pre_post_names['post_name_a'].apply(lambda x: x.lower())
   pre_post_names['post_name_b'] = pre_post_names['post_name_b'].apply(lambda x: x.lower())
   pre_post_names['pre_name_a_lemmas'] = pre_post_names['pre_name_a_lemmas'].apply(lambda x: x.lower())
   pre_post_names['pre_name_b_lemmas'] = pre_post_names['pre_name_b_lemmas'].apply(lambda x: x.lower())
   pre_post_names['post_name_a_lemmas'] = pre_post_names['post_name_a_lemmas'].apply(lambda x: x.lower())
   pre_post_names['post_name_b_lemmas'] = pre_post_names['post_name_b_lemmas'].apply(lambda x: x.lower())
   pre_post_names = pre_post_names[pre_post_names['Pair'].isin(fribble_specific_exp_info['int_pair'])]
   # return if there are pre and post names for the untranscribed pairs
      # for each pair and fribble, get the lists of A_freq and B_freq for each label
   fribble_specific_exp_info['A_freq'] = fribble_specific_exp_info.apply(lambda x: x['speakers'].count('A'), axis=1)
   fribble_specific_exp_info['B_freq'] = fribble_specific_exp_info.apply(lambda x: x['speakers'].count('B'), axis=1)
   speakers_count_per_fribble_and_pair = fribble_specific_exp_info.groupby(['target_fribble', 'int_pair']).agg({'A_freq': list, 'B_freq': list}).reset_index()

   # convert target_fribble and pair to int
   import re
   speakers_count_per_fribble_and_pair['target_fribble'] = speakers_count_per_fribble_and_pair['target_fribble'].astype(int)
   # take the integer part of the pairs names

   pre_post_names = pre_post_names.merge(speakers_count_per_fribble_and_pair[['target_fribble', 'int_pair', 'A_freq', 'B_freq']],
                                       how='left',
                                       left_on=['Fribble_nr', 'Pair'],
                                       right_on=['target_fribble', 'int_pair'])
   # convert each list of A_freq and B_freq to a numpy array
   pre_post_names['A_freq'] = pre_post_names['A_freq'].apply(lambda x: np.array(x))
   pre_post_names['B_freq'] = pre_post_names['B_freq'].apply(lambda x: np.array(x))
   return pre_post_names

def measure_distances_between_names(pre_post_names, fribble_specific_exp_info):


   pre_post_names['pre_lemmas_lexical_similarity'] = pre_post_names.apply(lambda x: measure_distance(x.pre_name_a_lemmas, x.pre_name_b_lemmas), axis=1)
   pre_post_names['post_lemmas_lexical_similarity'] = pre_post_names.apply(lambda x: measure_distance(x.post_name_a_lemmas, x.post_name_b_lemmas), axis=1)
   pre_post_names['lemmas_lexical_similarity_increase'] = pre_post_names['post_lemmas_lexical_similarity'] - pre_post_names['pre_lemmas_lexical_similarity']
   
      # merge pre and post names
   # calculate the distance between pre names lemmas and post names lemmas, and add it to the pre_post_names dataframe
   pre_post_names['pre_names_lexical_similarity'] = pre_post_names.apply(lambda x: measure_distance(x.pre_name_a, x.pre_name_b), axis=1)
   pre_post_names['post_names_lexical_similarity'] = pre_post_names.apply(lambda x: measure_distance(x.post_name_a, x.post_name_b), axis=1)
   # get the difference between the pre and post distances
   pre_post_names['names_lexical_similarity_increase'] = pre_post_names['post_names_lexical_similarity'] - pre_post_names['pre_names_lexical_similarity']

      # count the number of rows where pre_post_names['pre_name_a_lemmas'] is empty
   # if pre_name_a_lemmas is empty, then use the pre_name_a
   pre_post_names['pre_name_a_lemmas'] = pre_post_names.apply(lambda x: x['pre_name_a'] if x['pre_name_a_lemmas'] == '' else x['pre_name_a_lemmas'], axis=1)
   pre_post_names['pre_name_b_lemmas'] = pre_post_names.apply(lambda x: x['pre_name_b'] if x['pre_name_b_lemmas'] == '' else x['pre_name_b_lemmas'], axis=1)
   pre_post_names['post_name_a_lemmas'] = pre_post_names.apply(lambda x: x['post_name_a'] if x['post_name_a_lemmas'] == '' else x['post_name_a_lemmas'], axis=1)
   pre_post_names['post_name_b_lemmas'] = pre_post_names.apply(lambda x: x['post_name_b'] if x['post_name_b_lemmas'] == '' else x['post_name_b_lemmas'], axis=1)
   
   
   # pre_post_names['lemmas_w2v_similarity_increase'] = pre_post_names['post_lemmas_w2v_similarity'] - pre_post_names['pre_lemmas_w2v_similarity']

   pre_post_names['pre_names'] = pre_post_names.apply(lambda x: x['pre_name_a'].strip() + ' '+ x['pre_name_b'].strip(), axis=1)
   pre_post_names['post_names'] = pre_post_names.apply(lambda x: x['post_name_a'].strip() + ' '+ x['post_name_b'].strip(), axis=1)
  
   pre_post_names['distance_between_pre_post_names'] = pre_post_names.apply(lambda x: measure_distance(x.pre_name_a_lemmas, x.post_name_a_lemmas), axis=1) + pre_post_names.apply(lambda x: measure_distance(x.pre_name_b_lemmas, x.post_name_b_lemmas), axis=1)
   # formulate markers for each of names 
   pre_name_markers_similarity = []
   post_name_markers_similarity = []
   for i in range(len(pre_post_names)):
      all_expressions = fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == pre_post_names.iloc[i]['Pair']) & (fribble_specific_exp_info['target_fribble'] == pre_post_names.iloc[i]['Fribble_nr'])]['shared expressions'].to_list()
      all_expressions = ' '.join(set([item for sublist in all_expressions for item in sublist]))
      # print(all_expressions)
      pre_name_a_marker = pre_post_names.iloc[i]['pre_name_a_lemmas']
      for word in pre_post_names.iloc[i]['pre_name_a_lemmas'].split():
         if word in all_expressions:
               pre_name_a_marker = pre_name_a_marker.replace(word, 'shared')
      pre_name_b_marker = pre_post_names.iloc[i]['pre_name_b_lemmas']
      for word in pre_post_names.iloc[i]['pre_name_b_lemmas'].split():
         if word in all_expressions:
            pre_name_b_marker = pre_name_b_marker.replace(word, 'shared')
      post_name_a_marker = pre_post_names.iloc[i]['post_name_a_lemmas']
      for word in pre_post_names.iloc[i]['post_name_a_lemmas'].split():
         if word in all_expressions:
            post_name_a_marker = post_name_a_marker.replace(word, 'shared')
      post_name_b_marker = pre_post_names.iloc[i]['post_name_b_lemmas']
      # print('post_name_b_lemmas plz', post_name_b_marker)
      for word in pre_post_names.iloc[i]['post_name_b_lemmas'].split():
         if word in all_expressions:
            # print('word shared', word)
            post_name_b_marker = post_name_b_marker.replace(word, 'shared')
      # print('post_name_b_marker plz again', post_name_b_marker)
      pre_name_markers_similarity.append(measure_distance(pre_name_a_marker, pre_name_b_marker))
      post_name_markers_similarity.append(measure_distance(post_name_a_marker, post_name_b_marker))
   
   pre_post_names['pre names similarity given shared exps'] = pre_name_markers_similarity
   pre_post_names['post names similarity given shared exps'] = post_name_markers_similarity
   pre_post_names['names similarity increase given shared exps'] = pre_post_names['post names similarity given shared exps'] - pre_post_names['pre names similarity given shared exps']
   
   # pre_post_names['names_bert_increase'] = pre_post_names['post_names_mean_bert'] - pre_post_names['pre_names_mean_bert']
   pre_post_names['lexical_similarity_increase'] = pre_post_names['post_names_lexical_similarity'] - pre_post_names['pre_names_lexical_similarity']
   pre_post_names['Increase in Lemmas Similarity (Cosine Distance)'] = pre_post_names['post_lemmas_lexical_similarity'] - pre_post_names['pre_lemmas_lexical_similarity']

   return pre_post_names


def calculate_features_for_each_fribble_per_pair(pre_post_names, fribble_specific_exp_info):
   # add the average number of rounds to the pre_post_names dataframe
   # convert target_fribble to int
   fribble_specific_exp_info['#shared_expressions'] = fribble_specific_exp_info['shared expressions'].apply(lambda x: len(x))
   pre_post_names['total number of shared exp'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['#shared_expressions'].sum(), axis=1)

   # pre_post_names['average number of shared exp'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['#shared expressions'].mean(), axis=1)
   pre_post_names['average freq of shared exp'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['freq'].mean(), axis=1)
   pre_post_names['max freq of shared exp'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['freq'].max(), axis=1)
   pre_post_names['total freq shared exp'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['freq'].sum(), axis=1)
   fribble_specific_exp_info['first_turn'] = fribble_specific_exp_info['turns'].apply(lambda x: float(x[0]))
   fribble_specific_exp_info['last_turn'] = fribble_specific_exp_info['turns'].apply(lambda x: float(x[-1]))
   pre_post_names['avg number of rounds'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['number of rounds'].mean(), axis=1)
   pre_post_names['max rounds'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['number of rounds'].max(), axis=1)
   pre_post_names['turns_to_emerge'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['turns_to_emerge'].mean(), axis=1)
   pre_post_names['establishment_ts'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['establishment_ts'].mean(), axis=1)
   pre_post_names['last_round'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['last_round'].mean(), axis=1)
   pre_post_names['first_round'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['first_round'].mean(), axis=1)
   pre_post_names['spanning_time'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['spanning_time'].mean(), axis=1)
   pre_post_names['first_turn'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['first_turn'].mean(), axis=1)
   pre_post_names['last_turn'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['last_turn'].mean(), axis=1)

   pre_post_names['number of core labels'] = pre_post_names[['Pair', 'Fribble_nr']].apply(lambda x: fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == x.Pair) & (fribble_specific_exp_info['target_fribble'] == x.Fribble_nr)]['label'].count(), axis=1)

   # remove pre_post_names with nan in shared_exp_freq
   pre_post_names[pre_post_names['Pair'].isin(untranscirbed_pairs)]['average freq of shared exp'].isna().sum()
   
   
   pre_post_names['freq'] = pre_post_names.apply(lambda x: x['A_freq'] + x['B_freq'], axis=1)

   pre_post_names['cumulative_expression_score'] = pre_post_names.apply(lambda x: np.sum(x['A_freq']) + np.sum(x['B_freq']), axis=1)
   pre_post_names['symmetry_score'] = pre_post_names.apply(lambda x: np.abs(x['A_freq'] - x['B_freq']).sum(), axis=1)

   pre_post_names['m_symmetry_score'] = pre_post_names.apply(lambda x: np.sum((np.abs(x['A_freq'] - x['B_freq'])+0.001) * x['cumulative_expression_score'] * x['freq']), axis=1)


   pre_post_names['symmetry_index'] = pre_post_names.apply(lambda x: (np.abs(x['A_freq'] - x['B_freq']) / (x['A_freq'] + x['B_freq'] + np.finfo(float).eps)).mean(), axis=1)
   pre_post_names['speaker_dominance'] = pre_post_names.apply(lambda x: (np.abs(x['A_freq'] - x['B_freq']) / np.maximum(x['A_freq'], x['B_freq'] + np.finfo(float).eps)).mean(), axis=1)
   pre_post_names['pseudo_synchrony_score'] = pre_post_names.apply(lambda x: np.abs(np.cumsum(x['A_freq']) - np.cumsum(x['B_freq'])).sum(), axis=1)
   pre_post_names['weighted_coordination_score'] = pre_post_names.apply(lambda x: np.sum(np.minimum(x['A_freq'], x['B_freq']) * ((x['symmetry_index'] + x['speaker_dominance']) / 2)), axis=1)
   # pre_post_names['total_num_labels'] = pre_post_names.apply(lambda x: len(x['A_freq']), axis=1)
   pre_post_names['max_coordination_score'] = pre_post_names.apply(lambda x: np.max(np.maximum(x['A_freq'], x['B_freq'])), axis=1)
   pre_post_names['max_symmetry_score'] = pre_post_names.apply(lambda x: np.min(np.abs(x['A_freq'] - x['B_freq'])), axis=1)
   pre_post_names['delta_freq'] = pre_post_names.apply(lambda x: np.mean(np.sum(x['A_freq']/np.sum(x['A_freq']) + x['B_freq']/np.sum(x['B_freq']))), axis=1)
   pre_post_names['average_freq'] = pre_post_names.apply(lambda x: np.max(np.max(x['A_freq']) + np.max(x['B_freq'])), axis=1)

    
   return pre_post_names

   # pre_post_names['freq'] = pre_post_names.apply(lambda x: x['A_freq'] + x['B_freq'], axis=1)

   # pre_post_names['cumulative_expression_score'] = pre_post_names.apply(lambda x: np.sum(x['A_freq']) + np.sum(x['B_freq']), axis=1)
   # pre_post_names['symmetry_score'] = pre_post_names.apply(lambda x: np.abs(x['A_freq'] - x['B_freq']).sum(), axis=1)

   # pre_post_names['m_symmetry_score'] = pre_post_names.apply(lambda x: np.sum((np.abs(x['A_freq'] - x['B_freq'])+0.001) * x['cumulative_expression_score'] * x['freq']), axis=1)


   # pre_post_names['symmetry_index'] = pre_post_names.apply(lambda x: (np.abs(x['A_freq'] - x['B_freq']) / (x['A_freq'] + x['B_freq'] + np.finfo(float).eps)).mean(), axis=1)
   # pre_post_names['speaker_dominance'] = pre_post_names.apply(lambda x: (np.abs(x['A_freq'] - x['B_freq']) / np.maximum(x['A_freq'], x['B_freq'] + np.finfo(float).eps)).mean(), axis=1)
   # pre_post_names['pseudo_synchrony_score'] = pre_post_names.apply(lambda x: np.abs(np.cumsum(x['A_freq']) - np.cumsum(x['B_freq'])).sum(), axis=1)
   # pre_post_names['weighted_coordination_score'] = pre_post_names.apply(lambda x: np.sum(np.minimum(x['A_freq'], x['B_freq']) * ((x['symmetry_index'] + x['speaker_dominance']) / 2)), axis=1)




   return pre_post_names

def calculate_marker_distance_per_exp(fribble_specific_exp_info):
   # for each row in fribble specific exp info, check if each word in post_name_a_lemmas is in the list of shared expressions
   # if yes, add 1 to the counter
   # if no, add 0 to the counter
   pre_markers_similarity = []
   post_markers_similarity = []
   for i in range(len(fribble_specific_exp_info)):
      all_expressions = fribble_specific_exp_info[(fribble_specific_exp_info['pair'] == fribble_specific_exp_info.iloc[i]['pair']) & (fribble_specific_exp_info['target_fribble'] == fribble_specific_exp_info.iloc[i]['target_fribble'])]['shared expressions'].to_list()
      all_expressions = ' '.join(set([item for sublist in all_expressions for item in sublist]))

      pre_name_a_marker = fribble_specific_exp_info.iloc[i]['pre_Name_a_lemmas']
      post_name_a_marker = fribble_specific_exp_info.iloc[i]['post_Name_a_lemmas']
      pre_name_b_marker = fribble_specific_exp_info.iloc[i]['pre_Name_b_lemmas']
      post_name_b_marker = fribble_specific_exp_info.iloc[i]['post_Name_b_lemmas']

      for word in fribble_specific_exp_info.iloc[i]['pre_Name_a_lemmas'].split():
         if word in all_expressions:
            pre_name_a_marker = pre_name_a_marker.replace(word, 'found')
      for word in fribble_specific_exp_info.iloc[i]['post_Name_a_lemmas'].split():
         if word in all_expressions:
            post_name_a_marker = post_name_a_marker.replace(word, 'found')
      for word in fribble_specific_exp_info.iloc[i]['pre_Name_b_lemmas'].split():
         if word in all_expressions:
            pre_name_b_marker = pre_name_b_marker.replace(word, 'found')
      for word in fribble_specific_exp_info.iloc[i]['post_Name_b_lemmas'].split():
         if word in all_expressions:
            # print('word: ', word)
            # replace the word in post_name_b_marker with 'found'
            post_name_b_marker = post_name_b_marker.replace(word, 'found')
      pre_markers_similarity.append(measure_distance(pre_name_a_marker, pre_name_b_marker))
      post_markers_similarity.append(measure_distance(post_name_a_marker, post_name_b_marker))
   fribble_specific_exp_info['pre_markers_similarity'] = pre_markers_similarity
   fribble_specific_exp_info['post_markers_similarity'] = post_markers_similarity
   return fribble_specific_exp_info
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel

def plot_real_vs_pseudo_pairs_scores(pre_post_names, distance_measure, measure_name):
   # Filter real and pseudo pairs
   real_pairs = pre_post_names[pre_post_names['Pairs'] == 'Real']
   pseudo_pairs = pre_post_names[pre_post_names['Pairs'] == 'Pseudo']

   # Calculate mean and standard deviation for each group
   real_mean = real_pairs[distance_measure].mean()
   real_std = real_pairs[distance_measure].std()
   pseudo_mean = pseudo_pairs[distance_measure].mean() * -1
   pseudo_std = pseudo_pairs[distance_measure].std()

   # Print the mean and std for each group
   print(f"Mean for Real: {real_mean:.2f}, std: {real_std:.2f}")
   print(f"Mean for Pseudo: {pseudo_mean:.2f}, std: {pseudo_std:.2f}")

   # Set color palette
   sns.set_palette(sns.color_palette("Set2"))

   # Plotting
   ax = sns.violinplot(y=distance_measure, x='Pairs', data=pre_post_names)
   ax.set_xlabel('Pairs', fontsize=12, fontweight='bold')
   ax.set_ylabel('Score difference', fontsize=12, fontweight='bold')
   plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
   plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")

   # ax.axvline(pseudo_mean, color='blue', linestyle='--')
   ax.annotate(f"Mean: {pseudo_mean:.2f}", xy=(pseudo_mean, 0), xytext=(13, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color='black'), color='black', fontsize=12, fontweight='bold')
   # Plot mean lines and annotate
   # ax.axvline(real_mean, color='red', linestyle='--')
   ax.annotate(f"Mean: {real_mean:.2f}", xy=(1, real_mean), xytext=(-95, 35), textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color='black'), color='black', fontsize=12, fontweight='bold')

   # Perform t-test
   t, p = ttest_rel(real_pairs[distance_measure], pseudo_pairs[distance_measure], alternative='greater')
   print(f"t-test for {measure_name}: t = {t}, p = {p}")

def plot_similarity_scores(pre_post_names):
   naming_similarities = pd.DataFrame(
      [('Post', x) for x in pre_post_names['post_lemmas_lexical_similarity']]
      + [('Pre', x) for x in pre_post_names['pre_lemmas_lexical_similarity']],
      columns=['Names', 'score'])

   # Calculate mean for each group
   group_means = naming_similarities.groupby('Names')['score'].mean()

   ax = sns.histplot(x='score', hue='Names', discrete=False, data=naming_similarities, kde=True, bins=10)
   for (group, mean) in group_means.items():
      ax.axvline(mean, color='red', linestyle='--')
      ax.annotate(f"{group} Mean: {mean:.2f}", xy=(mean, 4), rotation=90, color='green', fontsize=14, fontweight='bold')
      # annotate only the x axis
      print(f"Mean for {group}: {mean:.2f}")
   # make x-ticks and y-ticks bold
   ax.set_xlabel('Score', fontsize=12, fontweight='bold')
   ax.set_ylabel('Count', fontsize=12, fontweight='bold')
   plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
   plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
   # fig.tight_layout(w_pad=1, h_pad=1)

   t, p = ttest_rel(pre_post_names['post_lemmas_lexical_similarity'], pre_post_names['pre_lemmas_lexical_similarity'], alternative='greater')
   print(f"t-test for semantic lemmas wv2 similarity: t = {t}, p = {p}")


   ax.legend(fontsize=14, loc='best')
 
   

def plot_correlations_between_features_and_similarity_scores(df):
    '''
    Function to plot graphs with seaborn lineplot.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to plot.
    '''

    # Define the properties for all plots
    plots_properties = [
        {"x": "max rounds", "y1": "post_names_lexical_similarity", "y2": "names_lexical_similarity_increase", "xlabel": 'Maximum number of rounds', "ylabel": 'Max lexical similarity score'},
        {"x": "max freq of shared exp", "y1": "post_names_lexical_similarity", "y2": "names_lexical_similarity_increase", "xlabel": 'Maximum frequency of shared experiences', "ylabel": 'Lexical similarity score'}
    ]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=False)

    # Iterate over the plots properties
    for i, plot_prop in enumerate(plots_properties):
        sns.lineplot(x=plot_prop["x"], y=plot_prop["y1"], data=df, label='Similarity of post names', ax=ax[i])
        sns.lineplot(x=plot_prop["x"], y=plot_prop["y2"], data=df, label='Similarity increase', ax=ax[i])
        ax[i].set_xlabel(plot_prop["xlabel"])
        ax[i].set_ylabel(plot_prop["ylabel"])

    plt.show()

    

def plot_correlations_between_symmetry_and_similarity_scores(df):
    '''
    Function to plot graphs with seaborn lineplot.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to plot.
    '''

    # Define the properties for all plots
    plots_properties = [
        {"x": "symmetry_score", "y1": "post_names_lexical_similarity", "y2": "names_lexical_similarity_increase", "xlabel": 'Symmetry score', "ylabel": 'lexical similarity score'},
        {"x": "symmetry_score", "y1": "post_names_mean_bert", "y2": "mean_bert_diff", "xlabel": 'Symmetry score', "ylabel": 'BERTje based score'}
    ]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=False)

    # Iterate over the plots properties
    for i, plot_prop in enumerate(plots_properties):
        sns.lineplot(x=plot_prop["x"], y=plot_prop["y1"], data=df, label='Similarity of post names', ax=ax[i])
        sns.lineplot(x=plot_prop["x"], y=plot_prop["y2"], data=df, label='Similarity increase', ax=ax[i])
        ax[i].set_xlabel(plot_prop["xlabel"])
        ax[i].set_ylabel(plot_prop["ylabel"])

    plt.show()
    
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def plot_with_density(data, x_feature, y_feature, title):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot
    sns.scatterplot(data=data, x=x_feature, y=y_feature, 
                    ax=ax, palette='Set2', legend='full', s=100, alpha=0.5, edgecolor='black')

    # Regression line
    sns.regplot(data=data, x=x_feature, y=y_feature, 
                ax=ax, scatter=False, color='black')

    # Calculate and display the Spearman correlation
    corr, p_value = spearmanr(data[x_feature], data[y_feature])
    corr_text = f'Spearman Correlation: {corr:.2f} and P-value: {p_value:.2e}'

    # Adjust text position to avoid overlapping
    text_x = data[x_feature].max() * 0.05
    text_y = data[y_feature].max() * 0.95

    # Add a density plot for x_feature on the top
    ax2 = ax.twiny()
    sns.kdeplot(data[x_feature], ax=ax2, color='blue', alpha=0.5, vertical=False)
    ax2.set_xlabel('Density', fontsize=14, weight='bold')
    ax2.grid(False)

    # Add a density plot for y_feature on the right
    ax3 = ax.twinx()
    sns.kdeplot(data[y_feature], ax=ax3, color='green', alpha=0.5, vertical=True)
    ax3.set_ylabel('Density', fontsize=14, weight='bold')
    ax3.grid(False)

    # Set title with different styles
    plt.title(title, fontsize=20)
    plt.text(0.5, 1.07, corr_text, fontsize=12, color='red', ha='center', va='bottom', transform=ax.transAxes, weight='bold')

    ax.set_xlabel(x_feature, fontsize=14, weight='bold')
    ax.set_ylabel(y_feature, fontsize=14, weight='bold')

    plt.show()

# Example usage


