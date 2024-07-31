import re
from collections import defaultdict
import glob
from speach import elan
import csv
import pickle
import spacy

import numpy as np
from whisperx.audio import load_audio
import whisperx
from utils.align_speech_time_wise import get_audio, align
import torch
import os

from utils.data_classes import Tier, Turn, Utterance, Gesture 

def get_gestures_info(all_gestures, dialogue, participant_ID, g_hand, from_ts, to_ts, verbose=False):
    gesture_n = participant_ID+'_'+g_hand+'H_gesture'
    # gestures_index = [idx for idx,val in enumerate(dialogue[gesture_n]['from_ts']) if val >= from_ts and val <= to_ts]
    gestures_index = [idx for idx,gesture in enumerate(dialogue[gesture_n]) if gesture.from_ts >= from_ts and gesture.from_ts <= to_ts]

    if gestures_index:
        if verbose:
            print('Utterance from {} to {} :'.format(from_ts, to_ts))
            print('Gesture/s found! they are as follows:')
        gestures_index = np.array(gestures_index)
        for gesture_index in gestures_index:
            g_to_ts = dialogue[gesture_n][gesture_index].to_ts
            g_from_ts = dialogue[gesture_n][gesture_index].from_ts
            is_gesture = dialogue[gesture_n][gesture_index].value
            if is_gesture == 'non-gesture':
                print('Non-gesture found!')
            
            # gesture should usually come with type, referent, and comment
            g_comment = None
            g_type = None
            g_referent = None
            #get the comment/referent/type from their tiers
            g_type_index = [idx for idx, gesture_type in enumerate(dialogue[gesture_n+'_type']) if gesture_type.from_ts == g_from_ts]
            if g_type_index:
                assert dialogue[gesture_n+'_type'][g_type_index[0]].to_ts == g_to_ts
                g_type = dialogue[gesture_n+'_type'][g_type_index[0]].value

            g_comment_index = [idx for idx, gesture_comment in enumerate(dialogue[gesture_n+'_comment']) if gesture_comment.from_ts == g_from_ts]
            if g_comment_index:
                assert dialogue[gesture_n+'_comment'][g_comment_index[0]].to_ts == g_to_ts
                g_comment = dialogue[gesture_n+'_comment'][g_comment_index[0]].value
            g_referent_index = [idx for idx, gesture_referent in enumerate(dialogue[gesture_n+'_referent']) if gesture_referent.from_ts == g_from_ts]

            if g_referent_index:
                assert dialogue[gesture_n+'_referent'][g_referent_index[0]].to_ts == g_to_ts
                g_referent = dialogue[gesture_n+'_referent'][g_referent_index[0]] .value
            if verbose:
                print('Gesture is for the {}'.format(participant_ID+g_hand))
                print('Gesture is from {} to {}'.format(g_from_ts, g_to_ts))
                print('Is it gesture?: {}'.format(is_gesture))
                print('Gesture type is: {}'.format(g_type))
                print('Gesture referent is: {}'.format(g_referent))
                print('Gesture comment is: {}'.format(g_comment))

            this_gesture = Gesture(is_gesture, g_from_ts, g_to_ts, g_type, g_referent, g_comment, g_hand)
            all_gestures.append(this_gesture)
    return all_gestures

def get_turn_info(dialogue, tier, trial_index, turn_ID, dataset = 'small', orig_dialogue={}, verbose=False, is_sudo=False):
    if verbose:
        message = 'new turn'
        x = message.center(20, "-")
        print(x)

    utterance = tier.value
    duration = tier.duration
    from_ts = tier.from_ts
    to_ts = tier.to_ts
    speaker = tier.tier_name.split('_')[0]
   
    gestures = []
    if dataset == 'small':
        gestures = get_gestures_info(gestures, dialogue, 'A', 'R', from_ts, to_ts, verbose=verbose)
        gestures = get_gestures_info(gestures, dialogue, 'A', 'L', from_ts, to_ts, verbose=verbose)
        gestures = get_gestures_info(gestures, dialogue, 'B', 'R', from_ts, to_ts, verbose=verbose)
        gestures = get_gestures_info(gestures, dialogue, 'B', 'L', from_ts, to_ts, verbose=verbose)
    if is_sudo:
        trial = orig_dialogue['trial'][trial_index].value
    else:   
        trial = dialogue['trial'][trial_index].value
    if dataset == 'large' and trial.split('_')[-1] == 'loc': # if the trial is a location trial, we do not use it as we are now focusing on the referential trials
        if verbose:
            print(utterance)
            print ('Location trial, skipping: {}'.format(trial))
        return None
    if is_sudo:
        target = orig_dialogue['target'][trial_index].value
        accuracy = orig_dialogue['accuracy'][trial_index].value
        correct_answer = orig_dialogue['correct_answer'][trial_index].value
        given_answer = orig_dialogue['given_answer'][trial_index].value
        director = orig_dialogue['director'][trial_index].value
    else:
        target = dialogue['target'][trial_index].value
        accuracy = dialogue['accuracy'][trial_index].value
        correct_answer = dialogue['correct_answer'][trial_index].value
        given_answer = dialogue['given_answer'][trial_index].value
        director = dialogue['director'][trial_index].value

    if verbose:
        print()
        print('Utterance from {} to {} :'.format(from_ts, to_ts))
        print('Utterance {}'.format(utterance))
        if is_sudo:
            print('Trial from {} to {} :'.format(orig_dialogue['trial'][trial_index].from_ts, orig_dialogue['trial'][trial_index].to_ts))
        else:
            print('Trial from {} to {} :'.format(dialogue['trial'][trial_index].from_ts, dialogue['trial'][trial_index].to_ts))
        print('Trial from {} to {} :'.format(dialogue['trial'][trial_index].from_ts, dialogue['trial'][trial_index].to_ts))
        print('Speaker {}'.format(speaker))
        print('Director {}'.format(director))
        print('Trial {}'.format(trial))
        print('Target {}'.format(target))
        print('Accuracy {}'.format(accuracy))
        print('Given answer {}'.format(given_answer))
        print('Correct answer {}'.format(correct_answer))
    round = trial.split('.')[0]
    if dataset == 'large':
        trial = trial.split('.')[1].split('_')[0]
    elif dataset == 'small':
        trial = trial.split('.')[1]
    this_turn = Turn(speaker, turn_ID, turn_ID, utterance, gestures, duration,
    from_ts, to_ts, round, trial, target, director, correct_answer, given_answer, accuracy, dataset)
    return this_turn



def build_dialogue_dict(eaf, verbose=False):
    complete_dialogue_tiers = defaultdict(list)
    for tier in eaf:
        if verbose:
            print(f"{tier.ID} | Participant: {tier.participant} | Type: {tier.type_ref}")
        for ann in tier:
            _from_ts = ann.from_ts.sec
            _to_ts = ann.to_ts.sec
            _duration = ann.duration
            if tier.ID in ['B_po', 'A_po']:
                tier_name = 'speakers'
            else:
                tier_name = tier.ID
            # rows.append((tier.ID, tier.participant, _from_ts, _to_ts, _duration, ann.value))            
            this_tier = Tier(tier_name=tier.ID, from_ts=_from_ts, to_ts=_to_ts, duration=_duration, value=ann.value)
            complete_dialogue_tiers[tier_name].append(this_tier)
            
            if verbose:
                print(f"{ann.ID.rjust(4, ' ')}. [{ann.from_ts} :: {ann.to_ts}] {ann.text}")
                print(tier.ID+ ': value '+ann.value)
                print(tier.ID+ ': text '+ann.text)
    return complete_dialogue_tiers

def normalize_utterance(utterance, dataset_name, dataset, verbose=False):
    print('Before: ', utterance)
    # utterance = utterance.replace(dataset[dataset_name]['repairs_marker'], "")
    for key, value in dataset[dataset_name]['dutch_numbers'].items():
        utterance = utterance.replace(key, value) 
    for key, value in dataset[dataset_name]['dutch_capital_letters'].items():
        utterance = utterance.replace(key, value)
    if dataset[dataset_name]['Non-verbal_sounds_marker'] in utterance:
        if verbose:
            print('Before: ', utterance)
        for expression in dataset[dataset_name]['Non-verbal_sounds_and_target_mentions']: # this is special for both datasets
            utterance = utterance.replace(expression, '')
        if verbose:
            print('After: ', utterance)
    if '(' in utterance or ')' in utterance:
        if verbose:
            print('Before: ', utterance)
        for inaduable_speech_marker in dataset[dataset_name]['inaudiable_speech_markers']:
            utterance = utterance.replace(inaduable_speech_marker, '')
        if verbose:
            print('After: ', utterance)
    if dataset[dataset_name]['targets_marker'] in utterance:
        if verbose:
            print('Before: ', utterance)
        utterance = utterance.replace(dataset[dataset_name]['targets_marker'], '')
        if verbose:
            print('After: ', utterance)
    if '\\' in utterance:
        if verbose:
            print('Before: ', utterance)
        utterance = utterance.replace("\\", '')
        if verbose:
            print('After: ', utterance)
    if '-' in utterance and dataset_name == 'large':
        if verbose:
            print('Before: ', utterance)
        utterance = utterance.replace("-", '')
        if verbose:
            print('After: ', utterance)
    if '/' in utterance:
        if verbose:
            print('Before: ', utterance)
        utterance = utterance.replace("/", ' ')
        if verbose:
            print('After: ', utterance)
    
    utterance = " ".join([word.replace('=-', 'IS') if len(word.split('?')[-1])==0 else ' '.join(word.split('?')).replace('=-', 'IS') for word in utterance.split()])
    utterance = utterance.strip()
    if utterance != '':
        if utterance[-1] in ['`', '\'', '!', '.', ',', ';', ':', '?']:
            utterance = utterance[:-1]
    # utterance = " ".join(utterance.split())
    print('After: ', utterance)

    return utterance
def clean_and_pos_tag_a_turn(pair_name, turn, dataset_name, dataset, nlp, verbose=False):
    utterance = turn.utterance_speech
    include_word = np.ones(len(utterance), dtype=bool)
    for idx, word_level in enumerate(utterance):
        word = word_level.word
        if word == '' or word == ' ':
            include_word[idx] = False
            continue
        if word[-1] == dataset[dataset_name]['repairs_marker']: # Self-repairs: e.g., “aan de zijkant twee ku- kubussen” --> this especial for the large dataset
            if verbose:
                print(' Self-repairs: e.g., “aan de zijkant twee ku- kubussen”')
                print(utterance)
            include_word[idx] = False
            continue
        if word[-1] == '?' and len(word) > 1:
            if word[-2] == dataset[dataset_name]['repairs_marker']: # Self-repairs: e.g., “aan de zijkant twee ku- kubussen” --> this especial for the large dataset
                if verbose:
                    print(' Self-repairs: e.g., “aan de zijkant twee ku- kubussen”')
                    print(utterance)
                include_word[idx] = False
                continue
        if word in dataset[dataset_name]['interaction_words']:
            include_word[idx] = False
            continue # remove interactional markers
        if word[0] == dataset[dataset_name]['Non-verbal_sounds_marker']:
            if verbose:
                print(utterance)
            include_word[idx] = False
            continue # remove encoded laughs
        if (len(word) > 1) & (word[-1] == dataset[dataset_name]['Non-verbal_sounds_marker']):
            if verbose:
                print(utterance)
            include_word[idx] = False
            continue
    utterance = [word_level for idx, word_level in enumerate(utterance) if include_word[idx]]
    if len(utterance) == 0:
        return None
    processed_utterance = ' '.join([word_level.word for word_level in utterance])
    pos_utterance = nlp(str(processed_utterance))
    lemmas_with_pos = ''
    pos_sequence = ''
    lemmas_sequence = ''
    text_lemma_pos = ''
    utterance_idx = 0
    final_utterance = []
    special_cases = ['-', '`', '\'', '!', '.', ',', ';', ':', '?', '=', '(', ')', '[', ']', '{', '}', '/', '\\']
    for idx, token in enumerate(pos_utterance):
        lemmas_sequence += token.lemma_+' '
        pos_sequence += token.pos_+' '
        lemmas_with_pos += token.lemma_+'#'+token.pos_+' '
        text_lemma_pos += token.text+'_'+token.lemma_+'#'+token.pos_+' '
        print(pair_name)
        print(token.text, '---',utterance[utterance_idx].word)
        final_utterance.append(Utterance(token.text, utterance[utterance_idx].from_ts, utterance[utterance_idx].to_ts, token.lemma_, token.pos_, token.lemma_+'#'+token.pos_))

        if (utterance[utterance_idx].word[-1] in special_cases) and (not token.text in special_cases):
            assert token.text == utterance[utterance_idx].word[:-1]
            continue 
        else:
            if token.text in special_cases:
                assert token.text == utterance[utterance_idx].word[-1]
            else:
                assert token.text == utterance[utterance_idx].word
            utterance_idx += 1
    turn.set_pos_sequence(pos_sequence)
    turn.set_lemmas_with_pos(lemmas_with_pos)
    turn.set_lemmas_sequence(lemmas_sequence)
    turn.set_text_lemma_pos(text_lemma_pos)
    turn.set_utterance_speech(final_utterance)
    return turn
 
def get_non_vocal_sounds_and_targets_mentions(dataset_name, dataset):
    non_vocal_sounds_and_targets_mentions = []
    for filepath in glob.iglob(dataset[dataset_name]["dataset_path"]):
        eaf = elan.read_eaf(filepath)
        complete_dialogue_tiers = build_dialogue_dict(eaf, verbose=False)
        complete_dialogue_tiers['speakers'].sort(key=lambda x: x.from_ts, reverse=False)
        for tier in complete_dialogue_tiers['speakers']:
            utterance = tier.value
            if dataset_name == 'small':
                matching = re.findall(r"\#.*?\#", " ".join(utterance.strip().split()))
                # target_matching = re.findall(r"\*.*?\*", " ".join(utterance.strip().split()))
            else:
                matching = re.findall(r"\*.*?\*", " ".join(utterance.strip().split()))
                # target_matching = re.findall(r"\#.*?\#", " ".join(utterance.strip().split()))
            if len(matching) > 0:
                for match in matching:
                    non_vocal_sounds_and_targets_mentions.append(match)
            elif dataset[dataset_name]['Non-verbal_sounds_marker'] in utterance:
                print(utterance)
            # if len(target_matching) > 0:
            #     for match in target_matching:
            #         non_vocal_sounds_and_targets_mentions.append(match)
            # elif dataset[dataset_name]['targets_marker'] in utterance:
            #     print(utterance)

    if dataset_name == 'large':
        return set(non_vocal_sounds_and_targets_mentions+ ["*sounds of the microphonesmack one's lips*", "*smack one's lipsbreath laugh*", "*smackinhale*", "*smack one's lipsbreath laugh*", "*coughsmack one's lips*", "*smack one's lipsbreath laugh*", "*smackinhale*", "*inhalesmack*", "*swallowsmack*", "*airthudairthudairthudairthudairthud*", "*stuttersmack one's lips*", "stutterstutter", "*stuttersmack one's lips*", "*smack one's lipsthinking sound*", "*airthudairthudairthudairthudairthud*", "*click of the computer", "*tongueclick* *tongueclick* *tongueclick* *tongueclick* *tongueclick*", "*click of the computer*"])
    return set(non_vocal_sounds_and_targets_mentions)

def get_interactional_words():
    interaction_words = []
    interaction_words_path = '/Users/esamghaleb/Documents/ResearchData/CABB Large Dataset/Results/Word_counts/1_preprocessing/input/interaction_words.txt'
    with open(interaction_words_path) as f:
        lines = f.readlines()
    for word in lines:
        word = word.split("\n")[0]
        interaction_words.append(word)
    return interaction_words

def prepare_and_write_turns(dataset_name, dataset, nlp, verbose=False):
    for filepath in glob.iglob(dataset[dataset_name]["dataset_path"]):
        pair_name  = re.findall(dataset[dataset_name]['pair_name_pattern'], filepath)[0]
        label_file = dataset[dataset_name]["output_path"]+pair_name.split('.')[0]+'.tsv'
        object_file =  dataset[dataset_name]["output_path"]+pair_name.split('.')[0]+'.pickle'
        eaf = elan.read_eaf(filepath)
        complete_dialogue_tiers = build_dialogue_dict(eaf, verbose=verbose)
        Turns = []
        trial_index = 0
        #sort the dialogue tiers based on the start time
        trail_to_ts = np.array([tier.to_ts for tier in complete_dialogue_tiers['trial']])
        trail_from_ts = np.array([tier.from_ts for tier in complete_dialogue_tiers['trial']])

        complete_dialogue_tiers['speakers'].sort(key=lambda x: x.from_ts, reverse=verbose)
        with open(label_file, 'w', encoding='utf8', newline='') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
            for turn_ID, tier in enumerate(complete_dialogue_tiers['speakers']):
                from_ts = tier.from_ts
                to_ts = tier.to_ts
                if trial_index >= dataset[dataset_name]['max_trials']:
                    print('Turn is out of the trials range')# the number of trials depends on the dataset
                    break
                try: # I noticed that there are some turns that are out of the trials range, e.g., these turns are between two trails. 
                    real_trail_index = np.where((trail_from_ts <= from_ts) & (trail_to_ts >= to_ts))
                    trial_index = real_trail_index[0][0]
                except:
                    print('Turn is overlapping two trials')
                this_turn = get_turn_info(complete_dialogue_tiers, tier, trial_index, turn_ID, dataset_name, verbose=verbose)
                if this_turn is not None:
                    this_turn = clean_and_pos_tag_a_turn(pair_name, this_turn, dataset_name, dataset, nlp, verbose=verbose)
                    if this_turn is not None:
                        this_turn_ID = len(Turns)
                        this_turn.set_ID(this_turn_ID)
                        this_turn.set_target_turn(this_turn_ID)
                        Turns.append(this_turn)
                        # write the turns to a tsv file
                        tsv_writer.writerow([this_turn.speaker+': ', this_turn.lemmas_sequence])
                        print(this_turn.speaker+': ', this_turn.lemmas_sequence)
        with open(object_file, 'wb') as handle:
            pickle.dump(Turns, handle, protocol=pickle.HIGHEST_PROTOCOL)

def prepare_and_write_target_trials_per_pair(dataset_name, dataset, nlp, alignment_model, alignment_metadata, device, verbose=False, aling_speech=False):
    non_used_pairs = []
    all_pairs = []
    for filepath in glob.iglob(dataset[dataset_name]["dataset_path"]):
        all_pairs.append(re.findall(dataset[dataset_name]['pair_name_pattern'], filepath)[0])
    # make pairs of pairs
    all_pairs = np.array(all_pairs)
    paired_pairs = {}
    # assert that each pair is not repeated
    for pair in all_pairs:
        # choose a random pair that is not the same as the current pair
        paired_pairs[pair] = np.random.choice(all_pairs[all_pairs != pair])
    for filepath in glob.iglob(dataset[dataset_name]["dataset_path"]):
        targets_turns = defaultdict(list)
        pair_name  = re.findall(dataset[dataset_name]['pair_name_pattern'], filepath)[0]
        sudo_pair_name = paired_pairs[pair_name]
        # check if AB_audio_path exists
        object_file =  dataset[dataset_name]["output_path"]+pair_name.split('.')[0]+'_and_'+sudo_pair_name.split('.')[0]+'.pickle'
        print('Pair name: ', pair_name)
        print('Sudo pair name: ', sudo_pair_name)
        eaf_pair_name = elan.read_eaf(filepath)
        eaf_sudo_pair_name = elan.read_eaf(filepath.replace(pair_name, sudo_pair_name))
        complete_dialogue_tiers = build_dialogue_dict(eaf_pair_name, verbose=verbose)
        complete_dialogue_tiers_sudo = build_dialogue_dict(eaf_sudo_pair_name, verbose=verbose)
        Turns = []
        AB_result_aligned = {'segments': [], 'word_segments': []}
        trial_index = 0
        sudo_trail_index = 0
        #sort the dialogue tiers based on the start time
        trail_to_ts = np.array([tier.to_ts for tier in complete_dialogue_tiers['trial']])
        trail_from_ts = np.array([tier.from_ts for tier in complete_dialogue_tiers['trial']])
        sudo_trail_to_ts = np.array([tier.to_ts for tier in complete_dialogue_tiers_sudo['trial']])
        sudo_trail_from_ts = np.array([tier.from_ts for tier in complete_dialogue_tiers_sudo['trial']])
        A_speaker_index = np.array([idx for idx, tier in enumerate(complete_dialogue_tiers['speakers']) if tier.tier_name == 'A_po'])
        complete_dialogue_tiers['speakers'] = [complete_dialogue_tiers['speakers'][idx] for idx in A_speaker_index]
        B_sudo_speaker_index = np.array([idx for idx, tier in enumerate(complete_dialogue_tiers_sudo['speakers']) if tier.tier_name == 'B_po'])
        complete_dialogue_tiers['speakers'] += [complete_dialogue_tiers_sudo['speakers'][idx] for idx in B_sudo_speaker_index]
        complete_dialogue_tiers['speakers'].sort(key=lambda x: x.from_ts, reverse=verbose)
        AB_audios = {}
        if dataset_name =='small':
            A_audio_path = dataset[dataset_name]['audio_path'].format(pair_name, pair_name, 'A')
            B_audio_path = dataset[dataset_name]['audio_path'].format(sudo_pair_name, sudo_pair_name, 'B')
            # AB_audio_path = dataset[dataset_name]['audio_path'].format(pair_name, pair_name, 'AB')
        else:
            A_audio_path = dataset[dataset_name]['audio_path'].format(pair_name, 'A')
            B_audio_path = dataset[dataset_name]['audio_path'].format(sudo_pair_name, 'B')
            # AB_audio_path = dataset[dataset_name]['audio_path'].format(pair_name, 'AB')
        speaker_A_found = False
        speaker_B_found = False
        try:
            AB_audios['A'] = load_audio(A_audio_path)       
            speaker_A_found = True
        except:
            speaker_A_found = False
        try:
            AB_audios['B'] = load_audio(B_audio_path)
            speaker_B_found = True
        except:
            speaker_B_found = False
             
        # AB_audio_path_ass = AB_audio_path.replace('wav', 'ass')
        if speaker_A_found:
            AB_audios['A'] = get_audio(AB_audios['A'], speaker = 'A')
        if speaker_B_found:
            AB_audios['B'] = get_audio(AB_audios['B'], speaker = 'B')
        transcript = []
        # with open(label_file, 'w', encoding='utf8', newline='') as tsv_file:
            # tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        speaker_recording_found = True
        for turn_ID, tier in enumerate(complete_dialogue_tiers['speakers']):
            from_ts = tier.from_ts
            to_ts = tier.to_ts
            if tier.tier_name == 'A_po':
                speaker_recording_found = speaker_A_found
            else:
                speaker_recording_found = speaker_B_found
            if trial_index >= dataset[dataset_name]['max_trials'] or sudo_trail_index >= dataset[dataset_name]['max_trials']:
                # print('Turn is out of the trials range')# the number of trials depends on the dataset
                break
            try: # I noticed that there are some turns that are out of the trials range, e.g., these turns are between two trails. 
                if tier.tier_name == 'A_po':
                    overlaps = np.minimum(trail_to_ts, to_ts) - np.maximum(trail_from_ts, from_ts)
                    overlaps[overlaps < 0] = 0
                    trial_index = np.argmax(overlaps)
                else:
                    overlaps = np.minimum(sudo_trail_to_ts, to_ts) - np.maximum(sudo_trail_from_ts, from_ts)
                    overlaps[overlaps < 0] = 0
                    sudo_trail_index = np.argmax(overlaps)
            except Exception as e:
                print(e)
                print('Turn is overlapping two trials')
            if tier.tier_name == 'A_po':
                # get the corresponding turn from the sudo pair
                this_turn = get_turn_info(complete_dialogue_tiers, tier, trial_index, turn_ID, dataset_name, verbose=verbose)
            else:
                this_turn = get_turn_info(complete_dialogue_tiers_sudo, tier, sudo_trail_index, turn_ID, dataset_name, verbose=verbose)
            if this_turn is not None:
                normalized_utterance = normalize_utterance(this_turn.utterance, dataset_name, dataset, verbose=False)
                if 'oh zo hij is leuk' in normalized_utterance: # wrong transcription, it is longer than the audio segment
                    normalized_utterance = 'oh zo'
                if normalized_utterance != "":
                    print('Turn ID: ', turn_ID)
                    if speaker_recording_found and aling_speech:
                        turns_transcript = {'text': normalized_utterance, 'start': this_turn.from_ts, 'end': this_turn.to_ts, 'speaker': this_turn.speaker}
                        transcript.append(turns_transcript)
                        turn_result_aligned = align([turns_transcript], alignment_model, alignment_metadata, AB_audios, device)
                        word_level = turn_result_aligned['segments'][0]['word-level']
                        AB_result_aligned['segments'].append(turn_result_aligned['segments'][0])
                        AB_result_aligned['word_segments'].append(turn_result_aligned['word_segments'][0])
                        utterance_speech = []
                        for text in word_level:
                            utterance_speech.append(Utterance(text['text'], text['start'], text['end']))
                    else:
                        utterance_speech = []
                        for word in normalized_utterance.split():
                            utterance_speech.append(Utterance(word, this_turn.from_ts, this_turn.to_ts))
                    this_turn.set_utterance_speech(utterance_speech)
                    this_turn = clean_and_pos_tag_a_turn(pair_name, this_turn, dataset_name, dataset, nlp, verbose=verbose)
                    if this_turn is not None:
                        this_turn_ID = len(Turns)
                        this_turn.set_ID(this_turn_ID)
                        target_turn = len(targets_turns[this_turn.target])
                        this_turn.set_target_turn(target_turn)
                        Turns.append(this_turn)
                        targets_turns[this_turn.target].append(this_turn)
        # with open(object_file, 'wb') as handle:
        with open(object_file, 'wb') as handle:
            pickle.dump(Turns, handle, protocol=pickle.HIGHEST_PROTOCOL)
        fribbles = targets_turns.keys()
        for fribble in fribbles:
            label_file = dataset[dataset_name]["output_path"]+pair_name.split('.')[0] + '_and_' + sudo_pair_name.split('.')[0]+"_"+fribble+'.tsv'
            object_file =  dataset[dataset_name]["output_path"]+pair_name.split('.')[0]+ '_and_'+ sudo_pair_name.split('.')[0]+"_"+fribble+'.pickle'
            with open(label_file, 'w', encoding='utf8', newline='') as tsv_file:
                tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
                for turn in targets_turns[fribble]:
                    tsv_writer.writerow([turn.speaker+': ', turn.lemmas_sequence])
            with open(object_file, 'wb') as handle:
                pickle.dump(targets_turns[fribble], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
       
        print("Well done!")
    return non_used_pairs
    # return targets_turns
        #
if __name__ == '__main__':

    device = "cpu" 
    # check if a GPU is available
    if torch.cuda.is_available():
        device = "cuda"
    spacy_model = 'nl_core_news_lg' # options: 'nl_core_news_lg', 'nl_core_news_md', 'nl_core_news_sm'
    # load alignment model and metadata
    alignment_model, alignment_metadata = whisperx.load_align_model("nl", device=device)

    inaudiable_speech_markers = ['(?)', '()', '( )', '(', ')', '(?])', '?]', '[?', ']?']
    nlp = spacy.load(spacy_model)
    dataset ={}
    dataset_name = 'small'
    dataset[dataset_name] = {}

    dataset[dataset_name]['audio_path'] = '/home/eghaleb/data/{}_synced_pp{}.wav'
    dataset[dataset_name]["dataset_path"] = "/home/eghaleb/data/ElanFiles/New_ELAN_files/pair*.eaf"
    dataset[dataset_name]['output_path'] = 'lex_code/dialign/targets_riws_lemma_{}/'.format(spacy_model.split('_')[-1])
    if not os.path.exists(dataset[dataset_name]['output_path']):
        os.makedirs(dataset[dataset_name]['output_path'])
    dataset[dataset_name]['name'] = dataset_name
    dataset[dataset_name]['max_trials'] = 96
    dataset[dataset_name]['pair_name_pattern'] = r"/([\w\d]+)\."

    dataset[dataset_name]['Non-verbal_sounds_matcher'] = r"\#.*?\#"
    dataset[dataset_name]['Non-verbal_sounds_marker'] = '#'
    dataset[dataset_name]['targets_marker'] = '*'
    dataset[dataset_name]['target_matcher'] = r"\*.*?\*"
    dataset[dataset_name]['repairs_marker'] = '-'
    dataset[dataset_name]['inaudiable_speech_markers'] = inaudiable_speech_markers
    dataset[dataset_name]['Non-verbal_sounds_and_target_mentions'] = get_non_vocal_sounds_and_targets_mentions(dataset_name, dataset)
    dataset[dataset_name]['interaction_words'] = get_interactional_words()

    # A dictionary of dutch numbers (between two * character) with their written form, from 1 to 16 
    dataset[dataset_name]['dutch_numbers'] = {'*1*': 'een', '*2*': 'twee', '*3*': 'drie', '*4*': 'vier', '*5*': 'vijf', '*6*': 'zes', '*7*': 'zeven', '*8*': 'acht', '*9*': 'negen', '*10*': 'tien', '*11*': 'elf', '*12*': 'twaalf', '*13*': 'dertien', '*14*': 'veertien', '*15*': 'vijfteen', '*16*': 'zestien'}

    # A dictionary with the first 16 dutch capital letters between two * characters, and the letters without the * characters
    dataset[dataset_name]['dutch_capital_letters'] = {'*A*': 'A', '*B*': 'B', '*C*': 'C', '*D*': 'D', '*E*': 'E', '*F*': 'F', '*G*': 'G', '*H*': 'H', '*I*': 'I', '*J*': 'J', '*K*': 'K', '*L*': 'L', '*M*': 'M', '*N*': 'N', '*O*': 'O', '*P*': 'P'}
    # A list of the first 16 dutch capital letters between two * characters
    dataset[dataset_name]['dutch_capital_letters_list'] = ['*A*', '*B*', '*C*', '*D*', '*E*', '*F*', '*G*', '*H*', '*I*', '*J*', '*K*', '*L*', '*M*', '*N*', '*O*', '*P*']
    

    all_targets_turns = prepare_and_write_target_trials_per_pair(dataset_name, dataset, nlp,alignment_model, alignment_metadata, device, verbose=False)

    dataset_name = 'large'
    dataset[dataset_name] = {}
    dataset[dataset_name]["dataset_path"] = '/home/eghaleb/data/ElanFiles/New_ELAN_files/*complete.eaf'
    dataset[dataset_name]['audio_path'] = '/home/eghaleb/data/{}_synced_pp{}.wav'

    dataset[dataset_name]['output_path'] = 'lex_code/dialign/targets_riws_lemma_{}/'.format(spacy_model.split('_')[-1])
   

    dataset[dataset_name]['name'] = dataset_name
    dataset[dataset_name]['max_trials'] = 192
    dataset[dataset_name]['pair_name_pattern'] = r"/(\d+)_"

    dataset[dataset_name]['Non-verbal_sounds_matcher'] = r"\*.*?\*"
    dataset[dataset_name]['Non-verbal_sounds_marker'] = '*'
    dataset[dataset_name]['targets_marker'] = '#'
    dataset[dataset_name]['target_matcher'] = r"\#.*?\#"
    dataset[dataset_name]['repairs_marker'] = ']'
    dataset[dataset_name]['inaudiable_speech_markers'] = inaudiable_speech_markers
    dataset[dataset_name]['Non-verbal_sounds_and_target_mentions'] = get_non_vocal_sounds_and_targets_mentions(dataset_name, dataset)
    dataset[dataset_name]['interaction_words'] = get_interactional_words()
    dataset[dataset_name]['dutch_capital_letters'] = {'#A#': 'A', '#B#': 'B', '#C#': 'C', '#D#': 'D', '#E#': 'E', '#F#': 'F', '#G#': 'G', '#H#': 'H', '#I#': 'I', '#J#': 'J', '#K#': 'K', '#L#': 'L', '#M#': 'M', '#N#': 'N', '#O#': 'O', '#P#': 'P'}
    # A list of the first 16 dutch capital letters between two # characters
    dataset[dataset_name]['dutch_capital_letters_list'] = ['#A#', '#B#', '#C#', '#D#', '#E#', '#F#', '#G#', '#H#', '#I#', '#J#', '#K#', '#L#', '#M#', '#N#', '#O#', '#P#']
    # A dictionary of dutch numbers (between two # character) with their written form, from 1 to 16
    dataset[dataset_name]['dutch_numbers'] = {'#1#': 'een', '#2#': 'twee', '#3#': 'drie', '#4#': 'vier', '#5#': 'vijf', '#6#': 'zes', '#7#': 'zeven', '#8#': 'acht', '#9#': 'negen', '#10#': 'tien', '#11#': 'elf', '#12#': 'twaalf', '#13#': 'dertien', '#14#': 'veertien', '#15#': 'vijfteen', '#16#': 'zestien'}

    non_used_pairs = prepare_and_write_target_trials_per_pair(dataset_name, dataset, nlp, alignment_model, alignment_metadata, device, verbose=False)
    # print('wait')
    print(non_used_pairs)
