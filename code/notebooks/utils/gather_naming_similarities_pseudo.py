
import numpy as np
from numpy import dot
from numpy.linalg import norm
import seaborn as sns
import matplotlib.pyplot as plt
   
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
 
 
def get_pre_post_pseudo_names(naming_task, fribble_specific_exp_info):
   # rename Name_a and Name_b to pre_name_a and pre_name_b. Name_a_lemmas and Name_b_lemmas to pre_name_a_lemmas and pre_name_b_lemmas
   pre_names = naming_task[naming_task['Session'] == 'pre'][['Name_a', 'Name_b', 'Name_a_lemmas', 'Name_b_lemmas', 'Pair', 'Fribble_nr']]
   post_names = naming_task[naming_task['Session'] == 'post'][['Name_a', 'Name_b', 'Name_a_lemmas', 'Name_b_lemmas', 'Pair', 'Fribble_nr']]
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
   # define cos as the cosine similarity measure from pytorch
   # remove the pairs that are not transcribed
   pre_post_names = pre_post_names[~pre_post_names['Pair'].isin(untranscirbed_pairs)]
   pseudo_pairs = fribble_specific_exp_info['pair'].unique()
   pseudo_pairs_speaker_A = [int(pair.replace('pair', '').split('_')[0]) for pair in pseudo_pairs]
   pseudo_pairs_speaker_B = [int(pair.replace('pair', '').split('_')[-1]) for pair in pseudo_pairs]
   # remove pair 19
   pre_post_names = pre_post_names[pre_post_names['Pair'] != 19]
   pseudo_pairs_pre_post_names = pre_post_names.copy()
   # columns to replace:
   columns_to_replace = ['pre_name_b', 'pre_name_b_lemmas', 'post_name_b_lemmas', 'post_name_b']
   for i, row in pseudo_pairs_pre_post_names.iterrows():
      original_pair = row['Pair']
      fribble = row['Fribble_nr']
      # index of the pair in pseudo_pairs
      if original_pair in untranscirbed_pairs:
         continue
      idx = pseudo_pairs_speaker_A.index(original_pair)
      speaker_b = pseudo_pairs_speaker_B[idx]
      # get the row of the speaker B
      rows_of_speaker_b = pre_post_names[(pre_post_names['Pair'] == speaker_b) & (pre_post_names['Fribble_nr'] == fribble)]
      # replace the columns
      for col in columns_to_replace:
         pseudo_pairs_pre_post_names.at[i, col] = rows_of_speaker_b[col].to_list()[0] 
      pseudo_pairs_pre_post_names.at[i, 'Pair'] = str(original_pair) + '_and_' + str(speaker_b)
      # convert pseudo pairs into int pairs
   # select the rows of pseudo speaker B
   pseudo_pairs_pre_post_names['Pair'] = pseudo_pairs_pre_post_names['Pair'].apply(lambda x: int(x.replace('_and_', '')))
   # reset the index
   pseudo_pairs_pre_post_names = pseudo_pairs_pre_post_names.reset_index(drop=True)
   # copy the pre_post_names from the pseudo pairs
   pre_post_names = pseudo_pairs_pre_post_names.copy()
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
   pre_post_names['distance_between_pre_post_names'] = pre_post_names.apply(lambda x: measure_distance(x.pre_name_a_lemmas, x.post_name_a_lemmas), axis=1) + pre_post_names.apply(lambda x: measure_distance(x.pre_name_b_lemmas, x.post_name_b_lemmas), axis=1)


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

