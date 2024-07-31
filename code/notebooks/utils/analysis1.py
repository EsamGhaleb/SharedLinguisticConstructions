from utils.similarity_measures import measure_distance
from utils.similarity_measures import prepare_per_speaker_data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# speaker_turn_round = prepare_per_speaker_data(pre_post_names, fribble_specific_exp_info, turns_info)
# speaker_turn_round['speaker_names_similarity_increase'] = speaker_turn_round['exp_post_names_lexical_similarity'] - speaker_turn_round['exp_pre_names_lexical_similarity']
# speaker_turn_round['speaker_name_change'] = speaker_turn_round.apply(lambda x: measure_distance(x.pre_lemmas, x.post_lemmas), axis=1)

def percentage_over_overlap(utterance, shared_exp):
   utterance = set(utterance.split())
   shared_exp = set(shared_exp.split())
   num_shared_exp = len(utterance.intersection(shared_exp))
   return num_shared_exp/len(utterance)

def similarity_of_names_with_utterances(turns_info, speaker_turn_round, pre_post_names, pseudo_or_real='Real', filter_fuction_words_utterances=False, function_words=None, stop_words=None, pos_func_words=None, pos_stop_words=None):
   turns_utterances_names = []
   for pair in turns_info.keys():
      if pseudo_or_real == 'Real':
            turn_pair_id = int(pair.replace('pair', ''))
      else:
         speaker_A_pair = str(int(pair.replace('pair', '').split('_')[0]))
         speaker_B_pair = str(int(pair.replace('pair', '').split('_')[-1]))
         turn_pair_id = speaker_A_pair + speaker_B_pair
         turn_pair_id = int(turn_pair_id)
      pairs_turns = speaker_turn_round[speaker_turn_round['int_pair'] == turn_pair_id]['turns'].to_list()
      for turn in turns_info[pair]:
         pos_sequence = turn.pos_sequence.strip()
         if filter_fuction_words_utterances:
            if all_func_words(pos_sequence, pos_func_words) or all_func_words(turn.lemmas_sequence, function_words) or all_func_words(turn.lemmas_sequence, stop_words):
               # print('skipping this turn')
               continue
            if 'NOUN' not in pos_sequence:
               # print('skipping this turn')
               continue

         turn_fribble = int(turn.target)
         speaker = turn.speaker.lower()
         naming_row = pre_post_names[(pre_post_names['Pair'] == turn_pair_id) & (pre_post_names['Fribble_nr'] == turn_fribble)]

         contains_shared_exp_yes = 'Contains shared exp' 
         overlap_with_shared_expression = 0
         if turn.ID not in pairs_turns:
            contains_shared_exp_yes = 'Does not contain shared exp'
            overlap_with_shared_expression = 0
         else:
            # calculate the overlap between the utterance and shared expressions
            utterance = turn.lemmas_sequence
            shared_expressions = speaker_turn_round[(speaker_turn_round['int_pair'] == turn_pair_id) & (speaker_turn_round['turns'] == turn.ID)]['exp'].to_list()
            shared_expressions = ' '.join(shared_expressions)
            overlap_with_shared_expression = percentage_over_overlap(utterance, shared_expressions)
            # calculate just the percentage of overlap
            
         turns_utterances_names.append({'pre_name': naming_row['pre_name_{}_lemmas'.format(speaker)].values[0], 'post_name': naming_row['post_name_{}_lemmas'.format(speaker)].values[0], 'turn_ID': turn.ID, 'pair': pair, 'speaker': speaker, 'utterance': turn.lemmas_sequence, 'round': turn.round, 'target': turn.target, 'contains_shared_exp': contains_shared_exp_yes, 'percentage_overlap_with_shared_expression': overlap_with_shared_expression})
         # break
   turns_utterances_names = pd.DataFrame(turns_utterances_names)
   turns_utterances_names['exp_pre_name_utterance'] = turns_utterances_names.apply(lambda x: measure_distance(x.pre_name, x.utterance), axis=1)
   turns_utterances_names['exp_post_name_utterance'] = turns_utterances_names.apply(lambda x: measure_distance(x.post_name, x.utterance), axis=1)
   return turns_utterances_names

def plot_percentage_of_shared_constructions_over_rounds(real_turns_utterances_names, pseudo_turns_utterances_names):
        # for each speaker, calculate the percentage of utterances that contain shared expressions per round
        real_turns_utterances_names['utterance_contains_shared_exp_percentage'] = real_turns_utterances_names.groupby(['pair', 'speaker', 'round'])['contains_shared_exp'].transform(lambda x: x[x == 'Contains shared exp'].count()/x.count())
        # group by pair, speaker and round and calculate the mean of the percentage of utterances that contain shared expressions
        pseudo_turns_utterances_names['utterance_contains_shared_exp_percentage'] = pseudo_turns_utterances_names.groupby(['pair', 'speaker', 'round'])['contains_shared_exp'].transform(lambda x: x[x == 'Contains shared exp'].count()/x.count())

        # Set the size of the plot
        plt.figure(figsize=(9, 5))
        real_data = real_turns_utterances_names.groupby(['pair', 'speaker', 'round'])['utterance_contains_shared_exp_percentage'].mean().reset_index()
        pseudo_data = pseudo_turns_utterances_names.groupby(['pair', 'speaker', 'round'])['utterance_contains_shared_exp_percentage'].mean().reset_index()
        # Plotting lines for real turns
        sns.lineplot(data=real_data, x='round', y='utterance_contains_shared_exp_percentage', 
                label='Real-Pairs', color='blue', marker='o', linestyle='-')

        # Plotting lines for pseudo turns
        sns.lineplot(data=pseudo_data, x='round', y='utterance_contains_shared_exp_percentage', 
                label='Pseudo-Pairs', color='green', marker='x', linestyle='--')


        # Add title and axis names
        # plt.title('Percentage of Utterances with Shared Expressions', fontsize=15)
        # remove the x Label
        plt.xlabel('')
        # plt.xlabel('Round', fontsize=12, fontweight='bold')
        plt.ylabel('Percentage', fontsize=12, fontweight='bold')
        # make the legend font bold
        plt.legend(title_fontsize='16', fontsize='16', loc='best', title='Pair Type')
        # make the x and y ticks bold
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        # replace X ticks with the following: R1, R2, R3, R4, R5, R6
        plt.xticks(np.arange(0, 6), ['R1', 'R2', 'R3', 'R4', 'R5', 'R6'])
        legend = plt.legend()
        for text in legend.get_texts():
                text.set_fontweight('bold')
                text.set_fontsize('14')
        # Display the plot
        plt.show()