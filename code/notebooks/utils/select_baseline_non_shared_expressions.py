
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from statannotations.Annotator import Annotator
import secrets

secrets_generator = secrets.SystemRandom()


def all_func_words(pos_seq, pos_func_words):
    tokenized_pos_seq = pos_seq.split(' ')
    return all(word in pos_func_words for word in tokenized_pos_seq)


def annonate_time_based_exps(time_based_exp_info, pos_func_words, function_words, interaction_words, frequent_words):
   time_based_exp_info['from_ts'] = time_based_exp_info['from_ts'].astype(float) 
   time_based_exp_info['to_ts'] = time_based_exp_info['to_ts'].astype(float)
   time_based_exp_info['duration'] = time_based_exp_info['to_ts'] - time_based_exp_info['from_ts']

   filtered_exps = np.logical_not([all_func_words(pos_seq, pos_func_words) for idx, pos_seq in enumerate(time_based_exp_info['pos_seq'].to_list())])
   # filtered_offset_results = offset_results[exp_with_only_func_words
   filtered_exps = np.logical_and(filtered_exps, np.logical_not([all_func_words(exp, function_words) for idx, exp in enumerate(time_based_exp_info['exp'].to_list())]))
   # filtered_offset_results = filtered_offset_results[exp_with_only_func_words]

   filtered_exps = np.logical_and(filtered_exps, np.logical_not([all_func_words(exp, frequent_words) for idx, exp in enumerate(time_based_exp_info['exp'].to_list())]))
   # data['count'] = data['count']/data['count'].sum()
   filtered_exps = np.logical_and(filtered_exps, np.logical_not([all_func_words(exp, interaction_words) for idx, exp in enumerate(time_based_exp_info['exp'].to_list())]))

   time_based_exp_info['data'] = ['']*len(time_based_exp_info)
   time_based_exp_info['data'][np.logical_not(filtered_exps)] = 'function words'
   time_based_exp_info['data'][filtered_exps] = 'non-function words'


   # keep only the pairs with annotations
   content_time_based_exp_info = time_based_exp_info[time_based_exp_info['data'] == 'non-function words']
   # pairs_with_annotatations = list(pairs_annotations.keys())
   # content_time_based_exp_info = content_time_based_exp_info[content_time_based_exp_info['pair'].isin(pairs_with_annotatations)]
   # keep only the exps from rounds 1 and 2
   content_time_based_exp_info['type'] = 'shared'
   # rename the columns rounds to round
   content_time_based_exp_info.rename(columns={'rounds': 'round'}, inplace=True)
   content_time_based_exp_info.rename(columns={'speakers': 'speaker'}, inplace=True)

   return time_based_exp_info, content_time_based_exp_info

def secure_random_int(min_value, max_value):
    """Generate a cryptographically secure random integer between min_value and max_value (inclusive)."""
    range_size = max_value - min_value + 1
    num_bits = range_size.bit_length()
    mask = (1 << num_bits) - 1

    while True:
        rand_value = secrets.randbits(num_bits) & mask
        if rand_value < range_size:
            break

    return rand_value + min_value
 
# select a random sample of 3 subsequent words from the expression, but the words must be in the same order as in the expression
def get_random_sample_of_words_from_exp(exp, num_words):
   exp_words = exp.split(' ')
   if len(exp_words) < num_words:
      return exp
   # generate number using secure random
   random_start_idx = secure_random_int(0, len(exp_words) - num_words)
   # random_start_idx = random.sample(range(0, len(exp_words) - num_words + 1), 1)[0]
   # random_start_idx = random.randint(0, len(exp_words) - num_words)
   return ' '.join(exp_words[random_start_idx:random_start_idx+num_words])


def get_non_shared_expressions(time_based_exp_info, turns_info, pos_func_words, function_words, interaction_words, frequent_words, get_turn_from_to_ts, pairs_without_audio_recordings):
   data = time_based_exp_info[['length', 'data']]
   shared_exps_lens = data['length'].to_list()
   mle_rate_shared_exps = np.mean(shared_exps_lens)
   # fig, ax = plt.subplots(1, 2, sharex=True, figsize=(12, 4))

   p = 1 / mle_rate_shared_exps

   shared_exp_df_geometric = pd.DataFrame(
   [('shared exps length', x) for x in shared_exps_lens]
   + [('non-shared exps length', x) for x in st.geom.rvs(p, size=int(12*len(shared_exps_lens)))],
   columns=['data', 'length']
   )
   turn_lemmas_pair = {'turn_lemmas_with_pos': [], 'pair': [], 'round': [], 'turn': [], 'speaker': []}
   time_based_exp_info['turns'] = time_based_exp_info['turns'].astype(int)
   time_based_exp_info['rounds'] = time_based_exp_info['rounds'].astype(int)
   pairs = time_based_exp_info['pair'].unique()
   for pair in pairs:
      for turn in turns_info[pair]:
         turn_lemmas_pair['turn_lemmas_with_pos'].append(turn.lemmas_with_pos)
         turn_lemmas_pair['pair'].append(pair)
         turn_lemmas_pair['round'].append(turn.round)
         turn_lemmas_pair['turn'].append(turn.ID)
         turn_lemmas_pair['speaker'].append(turn.speaker)
   turn_lemmas_pair = pd.DataFrame(turn_lemmas_pair)
   all_non_shared_constructions_info = []
   for geometric_sample in shared_exp_df_geometric[shared_exp_df_geometric['data'] == 'non-shared exps length']['length'].to_list():
      # choose a random index from turn_lemmas_pair
      # random_index = random.randint(0, len(turn_lemmas_pair)-1)
      # select random_index row using  secure random
      random_index = secure_random_int(0, len(turn_lemmas_pair)-1)
      # print(f"random_index: {random_index}")
      turn_lemmas_with_pos = turn_lemmas_pair.iloc[random_index]['turn_lemmas_with_pos']
      # print(f"turn_lemmas_with_pos: {turn_lemmas_with_pos}")
      turn_id = turn_lemmas_pair.iloc[random_index]['turn']
      speaker = turn_lemmas_pair.iloc[random_index]['speaker']
      pair = turn_lemmas_pair.iloc[random_index]['pair']
      round = turn_lemmas_pair.iloc[random_index]['round']
      if len(turn_lemmas_with_pos) < geometric_sample:
         sampled_turn_lemmas_with_pos = turn_lemmas_with_pos
      else:
         sampled_turn_lemmas_with_pos = get_random_sample_of_words_from_exp(turn_lemmas_with_pos, geometric_sample)
      if '?' in sampled_turn_lemmas_with_pos:
         continue
      overlapping_exp_info = time_based_exp_info[(time_based_exp_info['pair'] == pair) & (time_based_exp_info['turns']) & (time_based_exp_info['exp with pos'].str.contains(sampled_turn_lemmas_with_pos))]
      if len(overlapping_exp_info) > 0:
         # print('overlapping expression found')
         # print(overlapping_exp_info.iloc[0]['exp with pos'])
         # print(sampled_turn_lemmas_with_pos)
         continue
      # get the pos_seq from the sampled sequence
      pos_seq = ' '.join([lemma_pos.split('#')[1] for lemma_pos in sampled_turn_lemmas_with_pos.split()])
      # get the surface form from the sampled sequence
      surface_form = ' '.join([lemma_pos.split('#')[0] for lemma_pos in sampled_turn_lemmas_with_pos.split()])
      # print(surface_form)
      non_shared_constructions_info_dict = {'exp': surface_form, 'turn': turn_id, 'length': len(surface_form), 'turns': turn_id, 'rounds': [round], 'pos_seq': pos_seq, 'exp with pos': sampled_turn_lemmas_with_pos, 'round': round, 'speaker': speaker, 'pair': pair,'estab_turn': turn_id, 'estab_round': round, 'estab_speaker': speaker, 'first_round': round, 'last_round': round, 'type': 'not-shared', 'data': 'geometric-sample'}
      all_non_shared_constructions_info.append(non_shared_constructions_info_dict)
   # convert the list of dictionaries to a dataframe and explode the turns column
   non_shared_constructions_info = pd.DataFrame(all_non_shared_constructions_info)
   non_shared_constructions_info = non_shared_constructions_info.explode('turns')
   # add the from_ts and to_ts columns to the non_shared_constructions_info
   non_shared_constructions_info['to_ts'] = non_shared_constructions_info.apply(lambda row: get_turn_from_to_ts(row, turns_info, 'to_ts', pairs_without_audio_recordings), axis=1) # type: ignore
   non_shared_constructions_info['from_ts'] = non_shared_constructions_info.apply(lambda row: get_turn_from_to_ts(row, turns_info, 'from_ts', pairs_without_audio_recordings), axis=1) # type: ignore
   non_shared_constructions_info['data'] = 'geometric-sample'
   non_shared_constructions_info['length'] = non_shared_constructions_info['exp'].str.split().str.len()
   non_shared_constructions_info['duration'] = non_shared_constructions_info['to_ts'] - non_shared_constructions_info['from_ts']

   filtered_exps = np.logical_not([all_func_words(pos_seq, pos_func_words) for idx, pos_seq in enumerate(non_shared_constructions_info['pos_seq'].to_list())])
   # filtered_offset_results = offset_results[exp_with_only_func_words
   filtered_exps = np.logical_and(filtered_exps, np.logical_not([all_func_words(exp, function_words) for idx, exp in enumerate(non_shared_constructions_info['exp'].to_list())]))
   # filtered_offset_results = filtered_offset_results[exp_with_only_func_words]

   filtered_exps = np.logical_and(filtered_exps, np.logical_not([all_func_words(exp, frequent_words) for idx, exp in enumerate(non_shared_constructions_info['exp'].to_list())]))
   # data['count'] = data['count']/data['count'].sum()
   filtered_exps = np.logical_and(filtered_exps, np.logical_not([all_func_words(exp, interaction_words) for idx, exp in enumerate(non_shared_constructions_info['exp'].to_list())]))
   non_shared_constructions_info['data'] = ['']*len(non_shared_constructions_info)
   non_shared_constructions_info['data'][np.logical_not(filtered_exps)] = 'function words'
   non_shared_constructions_info['data'][filtered_exps] = 'non-function words'

   # select only the non-function words
   non_shared_constructions_info = non_shared_constructions_info[non_shared_constructions_info['data'] == 'non-function words']
   non_shared_constructions_info.reset_index(drop=True, inplace=True)
   non_shared_constructions_info['type'] = 'not-shared'
   # remove any duplicates in exp column
   non_shared_constructions_info.drop_duplicates(subset=['exp', 'from_ts'], inplace=True)
   non_shared_constructions_info.reset_index(drop=True, inplace=True)
   # remove any row with exp column value containing 'ja'
   # non_shared_constructions_info = non_shared_constructions_info[~non_shared_constructions_info['pos_seq'].str.contains('INTJ')]
   # non_shared_constructions_info = non_shared_constructions_info[~non_shared_constructions_info['pos_seq'].str.contains('SPACE')]
   # non_shared_constructions_info = non_shared_constructions_info[~non_shared_constructions_info['pos_seq'].str.contains('PUNCT')]
   # non_shared_constructions_info = non_shared_constructions_info[~non_shared_constructions_info['pos_seq'].str.contains('SYM')]
   # non_shared_constructions_info.reset_index(drop=True, inplace=True)
   return non_shared_constructions_info

def get_shared_non_shared_expressions(content_time_based_exp_info, non_shared_constructions_info):
   all_shared_non_shared_exp = pd.concat([content_time_based_exp_info[['exp', 'pos_seq', 'length', 'type', 'pair', 'speaker', 'round', 'from_ts', 'to_ts', 'duration']], non_shared_constructions_info[['exp', 'pos_seq', 'length', 'type', 'pair', 'speaker', 'round', 'from_ts', 'to_ts', 'duration']]], ignore_index=True)
   return all_shared_non_shared_exp

def get_balanced_non_shared_shared_expressions(content_time_based_exp_info, non_shared_constructions_info):
   # get random indices (with the length of content_based_time_based_exp) from the non_shared_constructions_info dataframe, but each time the indices should be different
   lengths_of_shared_exps = set(content_time_based_exp_info['length'].unique())

   # lengths_of_shared_exps = [1]
   list_of_non_shared_exp_dataframes = []
   list_of_shared_exp_dataframes = []
   open_class = ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB', 'ADP', 'DET']
   # open_class = ['NOUN']
   from sklearn.preprocessing import KBinsDiscretizer
   non_shared_constructions_info = non_shared_constructions_info[non_shared_constructions_info['duration'].notna()]
   # create discretizer
   kbins = KBinsDiscretizer(n_bins=30, strategy='uniform', encode='ordinal')
   content_time_based_exp_info['duration_bins'] = kbins.fit_transform(np.array(content_time_based_exp_info['duration']).reshape(-1,1))
   non_shared_constructions_info['duration_bins'] = kbins.transform(np.array(non_shared_constructions_info['duration']).reshape(-1,1))
   durations_of_shared_exps = set(content_time_based_exp_info['duration_bins'].unique())
   rounds = set(content_time_based_exp_info['round'].unique())
   pairs = set(content_time_based_exp_info['pair'].unique())
   for pair in pairs:
      for this_round in rounds:
         for length in lengths_of_shared_exps:
            tags_counts = {}
            for duration in durations_of_shared_exps:
               non_shared_constructions_info_with_length = non_shared_constructions_info[(non_shared_constructions_info['duration_bins'] == duration) & (non_shared_constructions_info['length'] == length) & (non_shared_constructions_info['round'] == this_round) & (non_shared_constructions_info['pair'] == pair)]
               content_time_based_exp_info_length = content_time_based_exp_info[(content_time_based_exp_info['duration_bins'] == duration) & (content_time_based_exp_info['length'] == length) & (content_time_based_exp_info['round'] == this_round) & (content_time_based_exp_info['pair'] == pair)]
               print('duration: ', duration)
               print('length: ', length)
               for pos_tag in open_class:
                     # get the number of rows with the length of the shared expressions
                     tags_counts[pos_tag] = content_time_based_exp_info_length['pos_seq'].str.contains(pos_tag).sum()
               normalized_tags_counts = {k: v / sum(tags_counts.values()) for k, v in tags_counts.items()}
               total_num_exps_given_tags = sum(tags_counts.values())
               for pos_tag in open_class:
                  if len(content_time_based_exp_info_length) == 0 or tags_counts[pos_tag] == 0:
                     continue
                  if total_num_exps_given_tags == len(content_time_based_exp_info_length):
                     print('total_num_exps_given_tags == len(content_time_based_exp_info_length)')
                     num_exps_with_length_and_tag = tags_counts[pos_tag]
                  else:
                     # get the number of exp given the probability of the tag
                     tags_count = normalized_tags_counts[pos_tag]
                     num_exps_with_length_and_tag = round(tags_count * len(content_time_based_exp_info_length))
                  # get the rows with the length of the shared expressions from the non_shared_constructions_info dataframe, with the same pos_tag
                  non_shared_constructions_info_with_length_and_tag = non_shared_constructions_info_with_length[non_shared_constructions_info_with_length['pos_seq'].str.contains(pos_tag)]
                  # get the rows with the length of the shared expressions from the non_shared_constructions_info dataframe
                  print('num_exps_with_length is {} and tag is {}: '.format(num_exps_with_length_and_tag, pos_tag))
                  print('len(non_shared_constructions_info_with_length_and_tag): ', len(non_shared_constructions_info_with_length_and_tag))
                  # get the rows with the length of the shared expressions from the non_shared_constructions_info dataframe
                  if len(non_shared_constructions_info_with_length_and_tag) < num_exps_with_length_and_tag:
                     num_exps_with_length_and_tag = len(non_shared_constructions_info_with_length_and_tag)
                  list_of_shared_exp_dataframes.append(content_time_based_exp_info_length.sample(n=num_exps_with_length_and_tag, replace=False))
                  # get the number of rows with the length of the non-shared expressions
                  content_non_shared_constructions_info_with_length_and_tag = non_shared_constructions_info_with_length_and_tag.sample(n=num_exps_with_length_and_tag, replace=False)
                  list_of_non_shared_exp_dataframes.append(content_non_shared_constructions_info_with_length_and_tag) 
   content_non_shared_constructions_info = pd.concat(list_of_non_shared_exp_dataframes, ignore_index=True)
   content_shared_constructions_info = pd.concat(list_of_shared_exp_dataframes, ignore_index=True)
   # select the same number of shared expressions as the number of non-shared expressions
   num_non_shared_exps = len(content_non_shared_constructions_info)
   # content_time_based_exp_info_sampled = content_time_based_exp_info.sample(n=num_non_shared_exps, replace=False)
   all_content_based_exp_info = pd.concat([content_shared_constructions_info[['exp', 'pos_seq', 'length', 'type', 'pair', 'speaker', 'round', 'from_ts', 'to_ts', 'duration']], content_non_shared_constructions_info[['exp', 'pos_seq', 'length', 'type', 'pair', 'speaker', 'round', 'from_ts', 'to_ts', 'duration']]], ignore_index=True)
   # print statistics about the length and duration of the shared and non-shared expressions
   return all_content_based_exp_info

def calculate_overlap(start1, end1, start2, end2):
   # Find the earliest and latest start and end times
   earliest_start = min(start1, start2)
   latest_end = max(end1, end2)
   latest_start = max(start1, start2)
   earliest_end = min(end1, end2)

    # Calculate the overlap, if any
   overlap = 0
   if latest_start < earliest_end:
      overlap = (earliest_end - latest_start)

   # Calculate the total duration
   total_duration = (end1-start1)

   # Calculate the percentage overlap
   if total_duration > 0:
      percentage_overlap = (overlap / total_duration) * 100
   else:
      percentage_overlap = 0

   return percentage_overlap
def get_maximum_index_overlap_from_a_list(overlap_times):
   return overlap_times.index(max(overlap_times))

def calculate_overlaps_between_expressions_and_gestures(all_shared_non_shared_exp,iconic_gestures_info):
   all_shared_non_shared_exp['overlap with gestures'] = [0]*len(all_shared_non_shared_exp)
   for index, row in all_shared_non_shared_exp.iterrows():
      pair = row['pair']
      round = row['round']
      speaker = row['speaker']
      pair_iconic_gestures_info = iconic_gestures_info[(iconic_gestures_info['pair'] == pair) & (iconic_gestures_info['speaker'] == speaker)]
      # calculate the overlap between the exp and the gestures
      overlaps = pair_iconic_gestures_info.apply(lambda gesture_row: calculate_overlap(row['from_ts'], row['to_ts'], gesture_row['from_ts'], gesture_row['to_ts']), axis=1) # type: ignore
      # get the maximum overlap
      max_overlap = max(overlaps)
      all_shared_non_shared_exp['overlap with gestures'][index] = max_overlap
   return all_shared_non_shared_exp

def plot_dists_of_shared_non_shared_lengths_duration_overlap(content_time_based_exp_info, all_non_shared_constructions_info, iconic_gestures_info, select_balanced=False, round=1, remove_punct_intj_sym_space=True):
   all_non_shared_constructions_info['round'] = pd.to_numeric(all_non_shared_constructions_info['round'])
   content_time_based_exp_info['round'] = pd.to_numeric(content_time_based_exp_info['round'])
   if remove_punct_intj_sym_space:
      non_shared_constructions_info = all_non_shared_constructions_info[~all_non_shared_constructions_info['pos_seq'].str.contains('INTJ')]
      non_shared_constructions_info = non_shared_constructions_info[~non_shared_constructions_info['pos_seq'].str.contains('SPACE')]
      non_shared_constructions_info = non_shared_constructions_info[~non_shared_constructions_info['pos_seq'].str.contains('PUNCT')]
      non_shared_constructions_info = non_shared_constructions_info[~non_shared_constructions_info['pos_seq'].str.contains('SYM')]
   else:
      non_shared_constructions_info = all_non_shared_constructions_info
   from utils.select_baseline_non_shared_expressions import get_balanced_non_shared_shared_expressions

   if select_balanced:
      all_shared_non_shared_exp = get_balanced_non_shared_shared_expressions(content_time_based_exp_info, non_shared_constructions_info)
   else:
      all_shared_non_shared_exp = get_shared_non_shared_expressions(content_time_based_exp_info, non_shared_constructions_info)
   # convert the round column to int
   # select only the expressions from the round
   # pairs_to_exclude = ['pair10', 'pair12', 'pair21']
   # all_shared_non_shared_exp = all_shared_non_shared_exp[~all_shared_non_shared_exp['pair'].isin(pairs_to_exclude)] 
   
   if round != 0:
      all_shared_non_shared_exp = all_shared_non_shared_exp[all_shared_non_shared_exp['round'] == round]
      
   all_shared_non_shared_exp = calculate_overlaps_between_expressions_and_gestures(all_shared_non_shared_exp,iconic_gestures_info)

   fig, ax = plt.subplots(1, 3, sharex=True, figsize=(18, 6))
   # covert all the columns to numeric
   all_shared_non_shared_exp['length'] = pd.to_numeric(all_shared_non_shared_exp['length'])
   all_shared_non_shared_exp['duration'] = pd.to_numeric(all_shared_non_shared_exp['duration'])
   all_shared_non_shared_exp['duration'] = all_shared_non_shared_exp['duration'].fillna(0)
   _ = sns.histplot(x='length', hue='type', discrete=True, data=all_shared_non_shared_exp, ax=ax[0])
   _ = sns.violinplot(x='length', y='type', discrete=True, data=all_shared_non_shared_exp, ax=ax[1])
   _ = sns.histplot(x='duration', hue='type', discrete=False, data=all_shared_non_shared_exp, ax=ax[2])
   # _ = sns.violinplot(x='duration', y='type', discrete=False, data=all_shared_non_shared_exp, ax=ax[1,1])
   # rename the x axis
   ax[0].set_xlabel('length (number of words)')
   ax[1].set_xlabel('length (number of words)')
   ax[2].set_xlabel('duration (in seconds)')

   

   # in all_content_based_exp_info, add a column with random numbers between 0 and 10, which will be used for statistical testing
   # replace nan with 0, for all the columns
   all_shared_non_shared_exp = all_shared_non_shared_exp.fillna(0)
   # plot distribution of the length & duration & overlap of the shared and non-shared expressions according to the random column
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   rx1 = sns.barplot(x='type', y='length', data=all_shared_non_shared_exp, ax=axes[0])
   rx2 = sns.barplot(x='type', y='duration', data=all_shared_non_shared_exp, ax=axes[1])
   rx3 = sns.barplot(x='type', y='overlap with gestures', data=all_shared_non_shared_exp, ax=axes[2])

   x = 'type'
   y = 'length'
   pairs = [('not-shared', 'shared')]
   order = np.unique(all_shared_non_shared_exp['type']).tolist()
   hue = 'type'
   annotator = Annotator(rx1, pairs, data=all_shared_non_shared_exp, x=x, y=y, order=order, verbose=False, log=False, )
   annotator.configure(test='t-test_welch', text_format='star', loc='outside')
   annotator.apply_and_annotate()

   x = 'type'
   y = 'duration'
   pairs = [('not-shared', 'shared')]
   order = np.unique(all_shared_non_shared_exp['type']).tolist()
   hue = 'type'
   annotator = Annotator(rx2, pairs, data=all_shared_non_shared_exp, x=x, y=y, order=order, verbose=False, log=False, )
   annotator.configure(test='t-test_welch', text_format='star', loc='outside')
   annotator.apply_and_annotate()

   x = 'type'
   y = 'overlap with gestures'
   pairs = [('not-shared', 'shared')]
   order = np.unique(all_shared_non_shared_exp['type']).tolist()
   hue = 'type'
   annotator = Annotator(rx3, pairs, data=all_shared_non_shared_exp, x=x, y=y, order=order, verbose=False, log=False, )
   annotator.configure(test='t-test_welch', text_format='star', loc='outside')
   annotator.apply_and_annotate()

   # rename x axis of each plot
   rx1.set(xlabel='type of expression')
   rx2.set(xlabel='type of expression')
   rx3.set(xlabel='type of expression')
   # rename y axis of each plot
   rx1.set(ylabel='length of expression (number of words)')
   rx2.set(ylabel='duration of expression (seconds)')
   rx3.set(ylabel='percentage of overlap with gestures')
   return all_shared_non_shared_exp
