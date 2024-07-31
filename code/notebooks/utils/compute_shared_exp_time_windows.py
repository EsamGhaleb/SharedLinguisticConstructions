import pandas as pd
import numpy as np
import re
from collections import defaultdict

def get_turn_from_to_ts(exp_row, turns_info, time_stamp, no_audio_pairs=[]):
   pair = exp_row['pair']
   if pair in no_audio_pairs:
      return None
   turn= int(exp_row['turns'])
   time_stamps = []
   if time_stamp == 'from_ts':
         index = 0
   else: 
      index = -1
   longest_match = exp_row['exp']
   sequence = turns_info[exp_row['pair']][turn].lemmas_sequence
   matches_counter = 0
   matches_list = np.zeros(len(sequence.split(' ')), dtype=int)
   for token_idx, token in enumerate(sequence.split(' ')): 
      token = token.replace('?', '\?') 
      a_match = re.finditer(token, longest_match)
      for a_sub_match in a_match: 
            # print('re matches', a_sub_match)
            if (a_sub_match.group(0) == token) & (a_sub_match.group(0) in longest_match.split(' ')):
               matches_list[token_idx] = 1
               matches_counter += 1
               break
   matches_list = ''.join(matches_list.astype(str))
   matched_groups = defaultdict()
   ones = re.compile(r'1+')
   for idx, m in enumerate(ones.finditer(matches_list)):
      indexes = np.array(list(range(m.start(), m.end())))
      matched_groups[idx] = indexes
   sorted_indexes = sorted(matched_groups, key=lambda k: len(matched_groups[k]), reverse=True)
   if len(matched_groups) >1 :
      if (len(matched_groups[sorted_indexes[0]]) == len(matched_groups[sorted_indexes[1]])):
         time_stamps.append(getattr(turns_info[exp_row['pair']][turn].utterance_speech[matched_groups[sorted_indexes[0]][index]], time_stamp))
         time_stamps.append(getattr(turns_info[exp_row['pair']][turn].utterance_speech[matched_groups[sorted_indexes[1]][index]], time_stamp))
   # print(pair, turn)
   time_stamps.append(getattr(turns_info[exp_row['pair']][turn].utterance_speech[matched_groups[sorted_indexes[0]][index]], time_stamp))
   return time_stamps[0]
def get_from_to_ts(exp_row, turns_info, time_stamp, no_audio_pairs):
   pair = exp_row['pair']
   if pair in no_audio_pairs:
      return None
   turns = exp_row['turns']
   time_stamps = []
   if time_stamp == 'from_ts':
         index = 0
   else: 
      index = -1
   for turn in turns:
      longest_match = exp_row['exp']
      sequence = turns_info[exp_row['pair']][turn].lemmas_sequence
      matches_counter = 0
      matches_list = np.zeros(len(sequence.split(' ')), dtype=int)
      for token_idx, token in enumerate(sequence.split(' ')): 
         token = token.replace('?', '\?') 
         a_match = re.finditer(token, longest_match)
         for a_sub_match in a_match: 
               # print('re matches', a_sub_match)
               if (a_sub_match.group(0) == token) & (a_sub_match.group(0) in longest_match.split(' ')):
                  matches_list[token_idx] = 1
                  matches_counter += 1
                  break
      matches_list = ''.join(matches_list.astype(str))
      # print('matches_list', matches_list)
      # print(exp_row['exp'])
      matched_groups = defaultdict()
      ones = re.compile(r'1+')
      for idx, m in enumerate(ones.finditer(matches_list)):
         indexes = np.array(list(range(m.start(), m.end())))
         matched_groups[idx] = indexes
      sorted_indexes = sorted(matched_groups, key=lambda k: len(matched_groups[k]), reverse=True)
      if len(matched_groups) >1 :
         if (len(matched_groups[sorted_indexes[0]]) == len(matched_groups[sorted_indexes[1]])):
            time_stamps.append(getattr(turns_info[exp_row['pair']][turn].utterance_speech[matched_groups[sorted_indexes[0]][index]], time_stamp))
            time_stamps.append(getattr(turns_info[exp_row['pair']][turn].utterance_speech[matched_groups[sorted_indexes[1]][index]], time_stamp))
    
      time_stamps.append(getattr(turns_info[exp_row['pair']][turn].utterance_speech[matched_groups[sorted_indexes[0]][index]], time_stamp))
   return time_stamps