from difflib import SequenceMatcher
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from IPython.display import Image
from IPython.display import display
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import entropy
import math 
from matplotlib.ticker import MaxNLocator
from utils.read_labeled_and_pos_target_data import assert_match_betwee_shared_exp_and_actual_utterances
from utils.compute_shared_exp_time_windows import get_from_to_ts

def calculate_entropy(targets):
    bins=np.arange(0, 17)
    max_entropy = math.log(len(bins)-1,2)
    exp_entropy = entropy(np.histogram(np.array(targets)-1, bins=bins)[0], base=2)/max_entropy
    return exp_entropy

def create_exp_info_row(expression, length, targets, target_fribbles, num_targets, exp_entropy, speakers, freq, free_freq, priming, spanning_rounds, spanning_time, duration_to_emerge, turns_to_emerge, rounds_to_emerge, turns, rounds, establishment_round, establishment_turn,  first_round, last_round, initiators, shared_exp_pos_seq, shared_exp_pos, pair, dataset, from_ts, to_ts, establishment_ts, shared_expressions):
    return { 'exp': expression, 'length': length, 'fribbles': targets, 'target_fribbles': target_fribbles,  '#fribbles': num_targets, 'fribbles entropy': exp_entropy, 'speakers': speakers, 'freq': freq, 'free. freq': free_freq, 'priming': priming, 'spanning_rounds': spanning_rounds, 'spanning_time': spanning_time, 'duration_to_emerge': duration_to_emerge, 'turns_to_emerge':turns_to_emerge, 'rounds_to_emerge': rounds_to_emerge, 'turns': turns, 'rounds': rounds, 'estab_round': establishment_round, 'estab_turn': establishment_turn, 'first_round': first_round, 'last_round': last_round, 'initiator': initiators, 'pos_seq': shared_exp_pos_seq, 'exp with pos': shared_exp_pos, 'pair': pair, 'dataset': dataset, 'from_ts': from_ts, 'to_ts': to_ts, 'establishment_ts': establishment_ts, 'shared expressions': shared_expressions}

def all_func_words(pos_seq, pos_func_words):
    tokenized_pos_seq = pos_seq.split(' ')
    return all(word in pos_func_words for word in tokenized_pos_seq)



def check_turns(idx, turns, all_turns, target_pair_exp, target_pair_fribbles):
    # turn_repeated = [set(turn).isdisjoint(sub_turn) for sub_turn in all_turns]
    turn_repeated =  np.bool8([set(sub_turns).issubset(turns) for sub_turns in all_turns])
    turn_repeated[idx] = False
    contain_expression =  np.bool8([target_pair_exp[idx] in exp_idx for exp_idx in target_pair_exp])
    contain_expression[idx] = False
    # contain_similar_fribbles_with_same_num = np.bool8([np.all(np.unique(target_pair_fribbles[idx]) == np.unique(fribbles)) for fribbles in target_pair_fribbles ])
    turn_repeated = np.logical_and(turn_repeated, contain_expression)
    # turn_repeated = np.logical_and(contain_similar_fribbles_with_same_num, turn_repeated)
    return turn_repeated

def filter_expressions_based_on_turns_overlap(target_pair_fribble_exp_info):
    target_pair_turns = target_pair_fribble_exp_info['turns'].to_numpy()
    target_pair_exp = target_pair_fribble_exp_info['exp'].to_numpy()
    target_pair_fribbles = target_pair_fribble_exp_info['fribbles'].to_numpy()
    filtered_exp = np.zeros(target_pair_exp.shape[0], dtype=bool)
    for idx, turns in reversed(list(enumerate(target_pair_turns))):
        states = check_turns(idx, turns, target_pair_turns, target_pair_exp, target_pair_fribbles)
        filtered_exp = np.logical_or(filtered_exp, states)
    filtered_exp = np.logical_not(filtered_exp)
    return target_pair_fribble_exp_info[filtered_exp]

def remove_consecutive_duplicate_words(sentence):
  # Split the sentence into words
  words = sentence.split()

  # Initialize a result list
  result = []

  # Keep track of the previous word
  previous_word = ''

  # Iterate over the words in the sentence
  for idx, word in enumerate(words):

    # If the current word is not the same as the previous word,
    # append it to the result list
    if idx == 0:
        result.append(word)
        previous_word = word
        continue
    if word.split('#')[0] != previous_word.split('#')[0]:
      result.append(word)

    # Update the previous word
    previous_word = word

  # Join the result list into a sentence and return it
  return ' '.join(result)

def merge_expressions_based_on_content_words(orig_sequence, orig_target_sequence, function_words, pos_func_words):
    # remove duplicates words and keep the order
    orig_sequence = remove_consecutive_duplicate_words(orig_sequence)
    orig_target_sequence = remove_consecutive_duplicate_words(orig_target_sequence)
    orig_target_sequence = remove_consecutive_duplicate_words(orig_target_sequence)
    # print('------------------')
    merge = {}
    # print('orig_sequence', orig_sequence)
    # print('orig_target_sequence', orig_target_sequence)
    # TODO: check with raquel the exclusion of tokens that contain only numbers
    pos_func_words = pos_func_words+['NUM']
    ones = re.compile(r'1+')

    sequence = ' '.join([word_pos.split('#')[0] for word_pos in orig_sequence.split(' ')])
    sequence_words = np.array(sequence.split(' '))

    target_sequence = ' '.join([word_pos.split('#')[0] for word_pos in orig_target_sequence.split(' ')])
    sequence_pos = np.array([word_pos.split('#')[1] for word_pos in orig_sequence.split(' ')])
    target_sequence_words = [word_pos.split('#')[0] for word_pos in orig_target_sequence.split(' ')]
    merge['merge'] = False
    # initialize SequenceMatcher object with
    # input string
    seqMatch = SequenceMatcher(None,sequence, target_sequence)

    # find match of longest sub-string
    # output will be like Match(a=0, b=0, size=5)
    match = seqMatch.find_longest_match(0, len(sequence), 0, len(target_sequence))
    # print longest substring
    matches_counter = 0
    matches_list = np.zeros(len(sequence.split(' ')), dtype=int)
    tokens = sequence.split(' ')
    matched_groups = defaultdict()

    if (match.size >= 1):
        longest_match = sequence[match.a: match.a + match.size]
        for idx, token in enumerate(tokens):
            for to_idx in range(idx, len(tokens)):
                if ' '.join(tokens[idx:to_idx+1]) in longest_match:
                    matched_groups[idx] = np.array(range(idx, to_idx+1))
                    matches_counter += 1
                else:
                    break
    if matches_counter == 0:
        return merge
    sorted_indexes = sorted(matched_groups, key=lambda k: len(matched_groups[k]), reverse=True)
    matching_indexes = None
    for index in sorted_indexes:
        pos_seq = sequence_pos[matched_groups[index]]
        if not all(pos in pos_func_words for pos in pos_seq):
            matching_indexes = matched_groups[index]
            break
    if matching_indexes is None:
        return merge

    merge['merge'] = True
    merge['merge_info'] = {}
    merge['merge_info']['exp with pos'] = ' '.join([sequence_words[idx]+'#'+sequence_pos[idx] for idx in matching_indexes])
    merge['merge_info']['pos_seq'] = ' '.join([sequence_pos[idx] for idx in matching_indexes])
    merge['merge_info']['exp'] = ' '.join([sequence_words[idx] for idx in matching_indexes])
 
    return merge


#  Merge two expressions if the have overlapping tokens (which are have content words)
def merge_identical_expressions(turns_info, exp_info, target_pair):
    first_row = True
    found_duplicates = False
    merged = np.full((len(exp_info),1), False, dtype=bool)
    # print(merged)
    merge_counter = 0
    # for idx, this_fribble in enumerate(target_pair_fribbles):
    for idx in exp_info[::-1].index:
        # for exp_to_merge_idx, fribbles_to_merge in enumerate(target_pair_fribbles):
        for exp_to_merge_idx in exp_info.index:
            # contain_similar_fribbles_with_same_num = np.any(np.unique(this_fribble) == np.unique(fribbles_to_merge))
            if ((not merged[idx]) or (not merged[exp_to_merge_idx])) & (exp_info['exp'].iloc[idx] == exp_info['exp'].iloc[exp_to_merge_idx]) & (idx != exp_to_merge_idx): # 
                merge_counter +=1
                found_duplicates = True
                exp = exp_info['exp'].iloc[idx]
                merged_turns = list(exp_info['turns'].iloc[idx]) + list(exp_info['turns'].iloc[exp_to_merge_idx])
                merged_turns.sort()
                merged_turns = np.unique(merged_turns)

                rounds = [int(turns_info[target_pair][turn].round) for turn in merged_turns]

                length = len(exp.split(' '))
                targets =  [int(turns_info[target_pair][turn].target) for turn in merged_turns]
                num_targets = len(np.unique(targets))
                fribbles_entropy =  calculate_entropy(targets) #target_pair_fribble_exp_info.iloc[idx]['fribbles entropy']  
                speakers = [turns_info[target_pair][turn].speaker for turn in merged_turns]
                freq = len(merged_turns)
                free_freq = min(exp_info['free. freq'].iloc[idx], exp_info['free. freq'].iloc[exp_to_merge_idx])
                spanning_rounds = int(rounds[-1]) - int(rounds[0])
                spanning_time = turns_info[target_pair][merged_turns[-1]].from_ts - turns_info[target_pair][merged_turns[0]].to_ts
                establishment_round = min(exp_info['estab_round'].iloc[idx], exp_info['estab_round'].iloc[exp_to_merge_idx])
                last_round = turns_info[target_pair][merged_turns[-1]].round
                first_round = turns_info[target_pair][merged_turns[0]].round
                initiator = turns_info[target_pair][merged_turns[0]].speaker
                establishment_turn = exp_info['estab_turn'].iloc[idx]

                priming = 0
                initiator = turns_info[target_pair][merged_turns[0]].speaker
                dataset = exp_info['dataset'].iloc[idx]
                for speaker_idx, speaker in enumerate(speakers):
                    if speaker == initiator:
                        priming += 1
                    else:
                        establishment_turn = merged_turns[speaker_idx]
                        break
                duration_to_emerge = turns_info[target_pair][establishment_turn].from_ts - turns_info[target_pair][merged_turns[0]].from_ts
                rounds_to_emerge = int(turns_info[target_pair][establishment_turn].round) - int(turns_info[target_pair][merged_turns[0]].round)
                turns_to_emerge = establishment_turn - merged_turns[0]

                pos_seq = exp_info['pos_seq'].iloc[idx]
                exp_with_pos = exp_info['exp with pos'].iloc[idx]
                from_ts = [turns_info[target_pair][a_turn].from_ts for a_turn in merged_turns]
                to_ts = [turns_info[target_pair][a_turn].to_ts for a_turn in merged_turns]
                establishment_ts = turns_info[target_pair][establishment_turn].from_ts
                shared_expressions = exp_info['shared expressions'].iloc[idx] #+ exp_info['shared expressions'].iloc[exp_to_merge_idx]

                target_fribbles = np.unique(targets)
                new_df = create_exp_info_row(exp, length, targets, target_fribbles, num_targets, fribbles_entropy, speakers, freq, free_freq, priming, spanning_rounds, spanning_time, duration_to_emerge, turns_to_emerge, rounds_to_emerge, merged_turns, rounds, establishment_round, establishment_turn, first_round, last_round, initiator, pos_seq, exp_with_pos, target_pair, dataset, from_ts, to_ts, establishment_ts, shared_expressions)
                new_df = pd.Series(new_df).to_frame().T
                merged[idx] = True
                merged[exp_to_merge_idx] = True
                if first_row:
                    merged_exp = new_df
                    first_row = False
                else:
                    merged_exp = pd.concat([merged_exp, new_df])
    if found_duplicates:
        not_merged = np.logical_not(merged)
        # print(merged_exp)
        merged_exp = pd.concat([merged_exp, exp_info[not_merged]])
        merged_exp.reset_index(inplace=True, drop=True)
        return found_duplicates, merged_exp
    else:
        return found_duplicates,exp_info 

#  Merge two expressions if the have overlapping tokens (which are have content words)
def filter_expressions_based_on_nouns_overlap(turns_info, exp_info, target_pair, target_fribble, function_words, pos_func_words):
    target_fribble = int(target_fribble)
    first_row = True
    found_duplicates = False
    merged = np.full((len(exp_info),1), False, dtype=bool)
    # print(merged)
    merge_counter = 0
    # for idx, this_fribble in enumerate(target_pair_fribbles):
    #TODO change iterrows to iterations over indexes
    for idx in exp_info[::-1].index:
        this_fribble = exp_info['fribbles'].iloc[idx]
        if not (target_fribble in this_fribble):
            continue
        # for exp_to_merge_idx, fribbles_to_merge in enumerate(target_pair_fribbles):
        for exp_to_merge_idx in exp_info.index:
            fribbles_to_merge = exp_info['fribbles'].iloc[exp_to_merge_idx]
            if not (target_fribble in fribbles_to_merge):
                # print('not in fribbles to merge', target_fribble, fribbles_to_merge)
                continue
      
            if not bool(set(exp_info['exp'].iloc[idx].split(' ')) & set(exp_info['exp'].iloc[exp_to_merge_idx].split(' '))):
                continue 
            # contain_similar_fribbles_with_same_num = np.any(np.unique(this_fribble) == np.unique(fribbles_to_merge))
            if (not merged[idx]) or (not merged[exp_to_merge_idx]): # check if two expressions are used for the same fribble
                if (idx == exp_to_merge_idx) or (len(exp_info['exp'].iloc[idx]) > len(exp_info['exp'].iloc[exp_to_merge_idx])):
                    continue
                merge = merge_expressions_based_on_content_words(exp_info['exp with pos'].iloc[idx],exp_info['exp with pos'].iloc[exp_to_merge_idx], function_words, pos_func_words)
                new_df = {}
                if merge['merge']:
                    # print('idx to merge:', exp_to_merge_idx)
                    merge_counter +=1
                    found_duplicates = True
                    exp= merge['merge_info']['exp']
                    
                    merged_turns = list(exp_info['turns'].iloc[idx]) + list(exp_info['turns'].iloc[exp_to_merge_idx])
                    merged_turns.sort()
                    merged_turns = np.unique(merged_turns)

                    rounds = [int(turns_info[target_pair][turn].round) for turn in merged_turns]

                    length = len(exp.split(' '))
                    targets =  [int(turns_info[target_pair][turn].target) for turn in merged_turns]
                    num_targets = len(np.unique(targets))
                    fribbles_entropy =  calculate_entropy(targets) #target_pair_fribble_exp_info.iloc[idx]['fribbles entropy']  
                    speakers = [turns_info[target_pair][turn].speaker for turn in merged_turns]
                    freq = len(merged_turns)
                    free_freq = min(exp_info['free. freq'].iloc[idx], exp_info['free. freq'].iloc[exp_to_merge_idx])
                    spanning_rounds = int(rounds[-1]) - int(rounds[0])
                    spanning_time = turns_info[target_pair][merged_turns[-1]].from_ts - turns_info[target_pair][merged_turns[0]].to_ts
                    establishment_round = min(exp_info['estab_round'].iloc[idx], exp_info['estab_round'].iloc[exp_to_merge_idx])
                    last_round = turns_info[target_pair][merged_turns[-1]].round
                    first_round = turns_info[target_pair][merged_turns[0]].round
                    initiator = turns_info[target_pair][merged_turns[0]].speaker
                    establishment_turn = exp_info['estab_turn'].iloc[idx]
                    priming = 0
                    initiator = turns_info[target_pair][merged_turns[0]].speaker
                    dataset = exp_info['dataset'].iloc[idx]
                    for speaker_idx, speaker in enumerate(speakers):
                        if speaker == initiator:
                            priming += 1
                        else:
                            establishment_turn = merged_turns[speaker_idx]
                            break
                    duration_to_emerge = turns_info[target_pair][establishment_turn].from_ts - turns_info[target_pair][merged_turns[0]].from_ts
                    rounds_to_emerge = int(turns_info[target_pair][establishment_turn].round) - int(turns_info[target_pair][merged_turns[0]].round)
                    turns_to_emerge = establishment_turn - merged_turns[0]

                    pos_seq = merge['merge_info']['pos_seq']
                    exp_with_pos = merge['merge_info']['exp with pos']
                    target_fribbles = np.unique(targets)
                    from_ts = [turns_info[target_pair][a_turn].from_ts for a_turn in merged_turns]
                    to_ts = [turns_info[target_pair][a_turn].to_ts for a_turn in merged_turns]
                    establishment_ts = turns_info[target_pair][establishment_turn].from_ts
                    # print('merged', exp_info)
                    # print('merged', exp_info['shared expressions'].iloc[idx], exp_info['shared expressions'].iloc[exp_to_merge_idx])
                    shared_expressions = exp_info['shared expressions'].iloc[idx] + exp_info['shared expressions'].iloc[exp_to_merge_idx]
                    shared_expressions = list(set(list(shared_expressions)))

                    # print('shared expressions', shared_expressions)


                    new_df = create_exp_info_row(exp, length, targets, target_fribbles, num_targets, fribbles_entropy, speakers, freq, free_freq, priming, spanning_rounds, spanning_time, duration_to_emerge, turns_to_emerge, rounds_to_emerge, merged_turns, rounds, establishment_round, establishment_turn, first_round, last_round, initiator, pos_seq, exp_with_pos, target_pair, dataset, from_ts, to_ts, establishment_ts, shared_expressions)
                    new_df = pd.Series(new_df).to_frame().T
                    merged[idx] = True
                    merged[exp_to_merge_idx] = True
                    if first_row:
                        merged_exp = new_df
                        first_row = False
                    else:
                        merged_exp = pd.concat([merged_exp, new_df])
    if found_duplicates:
     #    if merge_counter >4:
     #      print(merged_exp)
        # print(merged)
        not_merged = np.logical_not(merged)
        # print(merged_exp)
        merged_exp = pd.concat([merged_exp, exp_info[not_merged]])
        merged_exp.reset_index(inplace=True, drop=True)
        return found_duplicates, merged_exp
    else:
        return found_duplicates,exp_info 

def display_shared_exp_freq_for_target_fribble(linked_expressions, turns_info, target_fribble='09', target_pair='pair04', title_addition='', title_of_legend=''):
    linked_expressions.reset_index(inplace=True, drop=True)

    # check speaker role (director or matcher)
    # check if the speakers are the same as the directors, return D if they are the same, M if they are different
    linked_expressions['speaker role'] = linked_expressions.apply(lambda row: ["D" if turns_info[row['pair']][turn].speaker == turns_info[row['pair']][turn].director else "M" for turn in row['turns']], axis=1) # type: ignore
    linked_expressions['initiator role'] = linked_expressions.apply(lambda row: [ "D" if turns_info[row['pair']][row['turns'][0]].speaker == turns_info[row['pair']][row['turns'][0]].director else "M"], axis=1) # type: ignore
    linked_expressions['establisher role'] = linked_expressions.apply(lambda row: ["D" if turns_info[row['pair']][row['estab_turn']].speaker == turns_info[row['pair']][row['estab_turn']].director else "M"], axis=1) # type: ignore
    linked_expressions = linked_expressions.rename(columns={'exp': 'lexical core'})

    linked_expressions_speaker_rold = linked_expressions[['lexical core', 'shared expressions', 'speakers', 'speaker role', 'initiator role', 'establisher role', 'rounds', 'estab_round']]
    # display(linked_expressions_speaker_rold)

    # rename exp to label 
    # show all the content of cells in the dataframe
    pd.set_option('display.max_colwidth', None)
    # rename shared expressions to shared constructions
    new_linked_expressions = linked_expressions
    new_linked_expressions = new_linked_expressions.rename(columns={'shared expressions': 'shared constructions'})
    new_linked_expressions = new_linked_expressions.rename(columns={'lexical core': 'shared construction types'})
    display(new_linked_expressions[['shared construction types', 'shared constructions', 'freq', 'speakers', 'speaker role', 'initiator role', 'establisher role', 'rounds']])
    linked_expressions = linked_expressions.explode('rounds').groupby(['lexical core', 'rounds', 'estab_round']).count().reset_index().sort_values(by=['lexical core', 'rounds', 'estab_round']).rename(columns={'length': 'occurrences'})[['lexical core', 'rounds', 'estab_round', 'occurrences']]
    linked_expressions_estab_rounds = linked_expressions.groupby(['lexical core', 'estab_round'])['rounds'].count().reset_index().sort_values(by=['lexical core', 'estab_round']).sort_values(by=['lexical core'], ascending=False)    
    linked_expressions = linked_expressions.rename(columns={'lexical core': title_of_legend})
    ax = sns.histplot(linked_expressions, x='rounds', hue=title_of_legend, weights='occurrences',
                multiple='stack', palette='colorblind', shrink=.8, discrete=True)

    ax.figure.set_size_inches(7,5)

    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    for idx, c in enumerate(ax.containers):
        # Optional: if the segment is small or 0, customize the labels
        estab_round = int(linked_expressions_estab_rounds.iloc[idx]['estab_round'])
        label = linked_expressions_estab_rounds.iloc[idx]['lexical core']
        # get speaker and speaker role from linked_expressions_speaker_rold
        speaker_role = linked_expressions_speaker_rold[(linked_expressions_speaker_rold['lexical core'] == label) &(linked_expressions_speaker_rold['estab_round'] == estab_round)]['speaker role']
        # display(speaker_role)
        # print('label: ', label, 'estab_round: ', estab_round)
        speaker = linked_expressions_speaker_rold[(linked_expressions_speaker_rold['lexical core'] == label) & (linked_expressions_speaker_rold['estab_round'] == estab_round)]['speakers']
        # display(speaker)
        # print(estab_round)
        # print('idx is ', idx)
        speaker_role = speaker_role.to_list()[0]
        speaker = speaker.to_list()[0]
        # print('current_speaker_role: ', current_speaker_role, 'current_speaker: ', current_speaker)
        labels = []
        idx_select = 0
        for idx_c, v in enumerate(c):
            # print('idx_c: ', idx_c, 'v.get_height(): ', v.get_height())
            if (v.get_height() > 0):
                label_to_add = ''
                for v_h in range(int(v.get_height())):
                    # print('idx_select: ', idx_select)
                    # print('v_h: ', v_h)
                    label_to_add += speaker_role[idx_select]+ '-'+speaker[idx_select]+'\n'
                    idx_select += 1
                # print('label_to_add: ', label_to_add)
                labels.append(label_to_add)
            else:
                labels.append('')
        ax.bar_label(c, labels=labels, label_type='center', color='black', fontsize=10)
    plt.show()
    plt.close()


def link_all_pairs_shared_expressions_per_num_fribbles(turns_info, exp_info, function_words, pos_func_words, num_fribbles=1):
    all_pairs = set(exp_info['pair'].to_list())
    fribbles = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16']
    # all_linked_exps = pd.DataFrame()
    all_linked_exps_first = True
    for target_pair in all_pairs:
        for target_fribble in fribbles:
            target_pair_fribble_exp_info = exp_info[(exp_info['pair']==target_pair) & (exp_info['#fribbles']==num_fribbles) & np.array([int(target_fribble) in fribbles for fribbles in exp_info['target_fribbles'].to_list()])]
            if target_pair_fribble_exp_info.shape[0] == 0:
                # print('Num fribblbes {}, No shared expressions for pair {} and fribble {}'.format(num_fribbles, target_pair, target_fribble))
                continue

            target_pair_fribble_exp_info.reset_index(inplace=True, drop=True)
            found_duplicates = True
            while found_duplicates:
                found_duplicates, target_pair_fribble_exp_info = filter_expressions_based_on_nouns_overlap(turns_info, target_pair_fribble_exp_info, target_pair, target_fribble, function_words, pos_func_words)
            target_pair_fribble_exp_info.reset_index(inplace=True, drop=True)
            target_pair_fribble_exp_info['target_fribble'] = [target_fribble] * target_pair_fribble_exp_info.shape[0]

            if all_linked_exps_first:
                all_linked_exps = target_pair_fribble_exp_info
                all_linked_exps_first = False
            else:
                all_linked_exps = pd.concat([all_linked_exps, target_pair_fribble_exp_info], axis=0)
    return all_linked_exps.reset_index(drop=True)


def link_identical_shared_expressions_based_on_pair(turns_info, exp_info):
    all_pairs = set(exp_info['pair'].to_list())
    # all_linked_exps = pd.DataFrame()
    all_linked_exps_first = True
    for target_pair in all_pairs:
        # print('Pair: {}'.format(target_pair))
        target_pair_fribble_exp_info = exp_info[exp_info['pair']==target_pair]
        target_pair_fribble_exp_info.reset_index(inplace=True, drop=True)
        # target_pair_fribble_exp_info = filter_expressions_based_on_turns_overlap(target_pair_fribble_exp_info)
        # target_pair_fribble_exp_info.reset_index(inplace=True, drop=True)
        found_duplicates = True
        while found_duplicates:
            found_duplicates, target_pair_fribble_exp_info = merge_identical_expressions(turns_info, target_pair_fribble_exp_info, target_pair)
        target_pair_fribble_exp_info.reset_index(inplace=True, drop=True)

        if all_linked_exps_first:
            all_linked_exps = target_pair_fribble_exp_info
            all_linked_exps_first = False
        else:
            all_linked_exps = pd.concat([all_linked_exps, target_pair_fribble_exp_info], axis=0)
    return all_linked_exps.reset_index(drop=True)




def link_shared_expressions_fribble_based(turns_info, exp_info, function_words, pos_func_words, num_fribbles=1, target_fribble='09', target_pair='pair04', fribbles_path='', videos_path='', save_clips=False):

    display(Image(filename=fribbles_path.format(target_fribble)))
    target_pair_fribble_exp_info = exp_info[(exp_info['pair']==target_pair) & (exp_info['#fribbles']<=num_fribbles) & np.array([int(target_fribble) in fribbles for fribbles in exp_info['fribbles'].to_list()])]
    # print('#' * 36 + ' shared expressions before linking ' + '#' * 36)
    target_pair_fribble_exp_info.reset_index(inplace=True, drop=True)
    title_of_legend = 'All shared expressions'
    display_shared_exp_freq_for_target_fribble(target_pair_fribble_exp_info, turns_info, target_fribble=target_fribble, target_pair=target_pair, title_addition='Before linking expressions', title_of_legend=title_of_legend)

    target_pair_fribble_exp_info.reset_index(inplace=True, drop=True)

    # display(target_pair_fribble_exp_info[['exp', 'shared expressions', 'freq', 'turns', 'rounds', 'speakers']])

    found_duplicates = True
    while found_duplicates:
        found_duplicates, target_pair_fribble_exp_info = filter_expressions_based_on_nouns_overlap(turns_info, target_pair_fribble_exp_info, target_pair, target_fribble, function_words, pos_func_words)
    print('#' * 36 + ' shared expressions after linking ' + '#' * 36)
    title_of_legend = 'Common lexical cores of \n the shared expressions'
    display_shared_exp_freq_for_target_fribble(target_pair_fribble_exp_info, turns_info, target_fribble=target_fribble, target_pair=target_pair, title_addition='After linking expressions', title_of_legend=title_of_legend)
    return target_pair_fribble_exp_info

def link_shared_expressions_final_algo(turns_info, exp_info, function_words, pos_func_words):
    return link_all_pairs_shared_expressions_per_num_fribbles(turns_info, exp_info, function_words, pos_func_words)

def final_linking_algo_individually(turns_info, exp_info, shared_constructions_info, function_words, pos_func_words, frequent_words, pseudo_pairs=False):
   # select exps where #fribbles is only 1
   assert_match_betwee_shared_exp_and_actual_utterances(turns_info, shared_constructions_info)
   pairs_without_audio_recordings = ['011012', '097098', '137138', '127128', '153154', 'pair19'] # these pairs have no audio recordings
   # get word-level time windows for each lemma in each shared expression
   exp_info['to_ts'] = exp_info.apply(lambda row: get_from_to_ts(row, turns_info, 'to_ts', pairs_without_audio_recordings), axis=1) # type: ignore
   exp_info['from_ts'] = exp_info.apply(lambda row: get_from_to_ts(row, turns_info, 'from_ts', pairs_without_audio_recordings), axis=1) # type: ignore
   # Here, we assert that all detected shared expressions are in the original lemmatized utterances, with matching turn numbers
   linked_exp_info = link_shared_expressions_final_algo(turns_info, exp_info, function_words, pos_func_words)
   # rename exp to label
   linked_exp_info = linked_exp_info.rename(columns={'exp': 'label'})
   linked_exp_info['exp'] = linked_exp_info['label']
   linked_exp_info['#shared expressions'] = linked_exp_info['shared expressions'].apply(lambda x: len(x))
   for row_idx, row in linked_exp_info.iterrows():
      pair = row['pair']
      turns = row['turns']
      for turn in turns:
         assert row['label'] in turns_info[pair][int(turn)].lemmas_sequence
         
   fribble_specific_exp_info = linked_exp_info[linked_exp_info['#fribbles'] == 1]
   fribble_specific_exp_info['target_fribble'] = fribble_specific_exp_info.apply(lambda row: row['target_fribbles'][0], axis=1) # type: ignore
   fribble_specific_exp_info['expressions_set'] = fribble_specific_exp_info.apply(lambda x: ' '.join(set(' '.join(x['shared expressions']).split())), axis=1) # make a set of shared expressions
   fribble_specific_exp_info['number of rounds'] = fribble_specific_exp_info.apply(lambda x: len(np.unique(x['rounds'])), axis=1)
   import re
   all_function_words = list(function_words) + list(frequent_words)
   fribble_specific_exp_info['content_words_exp'] = fribble_specific_exp_info.apply(lambda row: ' '.join([word for word in set(row['expressions_set'].split()) if word not in all_function_words]), axis=1) # type: ignore
   # remove if content_words_exp has empty strings
   fribble_specific_exp_info = fribble_specific_exp_info[fribble_specific_exp_info['content_words_exp'] != '']
   
   
   if not pseudo_pairs:
      fribble_specific_exp_info['int_pair'] = fribble_specific_exp_info.apply(lambda x: int(re.findall(r'\d+', x['pair'])[0]), axis=1)
   else:
      fribble_specific_exp_info['pair'].apply(lambda x: x.replace('pair', '').replace('_and_', '')).astype(int)
      fribble_specific_exp_info['speaker_A_pair'] = fribble_specific_exp_info['pair'].apply(lambda x: int(x.replace('pair', '').split('_')[0]))
      fribble_specific_exp_info['speaker_B_pair'] = fribble_specific_exp_info['pair'].apply(lambda x: int(x.replace('pair', '').split('_')[-1]))
      # put the number next to each other
      fribble_specific_exp_info['int_pair'] = fribble_specific_exp_info['speaker_A_pair'].astype(str) + fribble_specific_exp_info['speaker_B_pair'].astype(str)
      fribble_specific_exp_info['int_pair'] = fribble_specific_exp_info['int_pair'].astype(int)
      # remove the speaker_A_pair and speaker_B_pair columns
      fribble_specific_exp_info = fribble_specific_exp_info.drop(columns=['speaker_A_pair', 'speaker_B_pair'])
   
   return fribble_specific_exp_info