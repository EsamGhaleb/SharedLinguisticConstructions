import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
# from utils.extract_embeddings import get_embeddings
from nltk.metrics.distance import edit_distance
import re
import seaborn as sns
import torch
import spacy
import json


def fix_names(names_to_fix, name):
   for key, value in names_to_fix.items():
      name = name.replace(key, value)
   return name
   
POS_content_words = ['ADJ', 'NOUN', 'PROPN', 'VERB', 'NUM', 'ADV', 'INTJ']
def get_lemmas(names, nlp):
   # use spacy to get the lemmas
   nlp_names = nlp(names)
   spacy_lemmas = " ".join([token.lemma_ for token in nlp_names])
   for token in nlp_names:
      if token.pos_ not in POS_content_words:
         spacy_lemmas = spacy_lemmas.replace(token.lemma_, '')
   spacy_lemmas = " ".join(spacy_lemmas.split())
   return spacy_lemmas

def word2vec_similarity_between_word(gen_model, a_words, b_words):
    return np.mean([gen_model.similarity(a, b) for a in a_words for b in b_words])

def get_naming_task(gen_model=None, nlp=None):
   large_dataset_naming_ouptut_path = '../data/CABB/naming_task/naming_task_lexical_similarity_corrected.txt'
   # read the the data using csv reader
   large_dataset_naming_task = pd.read_csv(large_dataset_naming_ouptut_path, sep=';',  encoding='latin-1')
   large_dataset_naming_ouptut_path = '../data/CABB/naming_task/naming_behstudy_FINAL.csv'
   small_dataset_naming_task = pd.read_csv(large_dataset_naming_ouptut_path, sep=';',  encoding='latin-1')
   small_dataset_naming_task = small_dataset_naming_task.drop(columns=['RT_a', 'RT_b'])
   naming_task = pd.concat([large_dataset_naming_task, small_dataset_naming_task])
   naming_task.reset_index(drop=True, inplace=True)
   # special charachters
   characters_to_remove = [',' , '.' , '!' , '?' , ':' , ';' , '(' , ')' , '-', '\'']
   
   names_to_fix = {'ineenlopende': 'in eenlopende', 'waterspuwen': 'water spuwen', 'waterspuit': 'water spuit', 'sticom': 'sitcom', 'kaasplakje': 'kaas plakje', 'gepixeliseerde': 'gepixeleerde', 'hollevorm': 'hollevorm', 'circelplateau': 'circel plateau', 'yoghurtbak': 'yoghurt bak', 'yoghurtbakje': 'yoghurt bakje', '  ': ' ', 'blankjes': 'blanken', 'diaboloõs': 'diabolo'}

   columns_to_fix = ['Name_a', 'Name_b']
   for column in columns_to_fix:
      naming_task[column] = naming_task.apply(lambda x: fix_names(names_to_fix, x[column]), axis=1)
   
   for index, row in naming_task.iterrows():
      Name_a = row['Name_a']
      Name_b = row['Name_b']
      for character in characters_to_remove:
         Name_a = Name_a.replace(character, '')
         Name_b = Name_b.replace(character, '')
      # make all characters lower case
      Name_a = Name_a.lower().strip()
      Name_b = Name_b.lower().strip()
      naming_task.at[index, 'Name_a'] = Name_a
      naming_task.at[index, 'Name_b'] = Name_b
      
      naming_task.at[index, 'Name_a_lemmas'] = get_lemmas(Name_a, nlp)
      # label_b = nlp(str(Name_b))
      # label_b_lemmas = " ".join([token.lemma_ for token in label_b])
      naming_task.at[index, 'Name_b_lemmas'] = get_lemmas(Name_b, nlp)
   return naming_task

def word_level_edit_distance(X, Y):
   return edit_distance(X.split(), Y.split())

def measure_semantic_cosine_similarity(X, Y, nlp):
   X = nlp(X)
   Y = nlp(Y)
   return X.similarity(Y)
def measure_semantic_cosine_similarity_based_on_BERT_embeddings_between_two_list_of_expressions(X, Y, model, tokenizer, cos):
   X = model.encode(X)
   # convert to pytorch tensor
   X = torch.from_numpy(X)
   Y = model.encode(Y)
   # convert to pytorch tensor
   Y = torch.from_numpy(Y)
   return cos(X, Y).numpy()
   
from numpy import dot
from numpy.linalg import norm
def measure_distance(X, Y):
   X = set(X.split())
   Y = set(Y.split())
   # form a set containing keywords of both strings 
   rvector = X.union(Y) 
   l1 = [1 if w in X else 0 for w in rvector]
   l2 = [1 if w in Y else 0 for w in rvector]
   return dot(l1, l2)/(norm(l1)*norm(l2))

def distance_betwee_shared_exp_and_pre_post_names(naming_task, speaker_based_fribble_specific_exp_info, nlp, gen_model):
   # Load model from HuggingFace Hub
   speaker_based_fribble_specific_exp_info = speaker_based_fribble_specific_exp_info.reset_index(drop=True)
   ID_pattern = r"\d+"
   IDs = []
   pre_name = []
   post_name = []
   for idx in range(0, len(speaker_based_fribble_specific_exp_info)):
      exp_pair_id = int(re.findall(ID_pattern, speaker_based_fribble_specific_exp_info.iloc[idx]['pair'])[0])
      exp_fribble = int(speaker_based_fribble_specific_exp_info.iloc[idx]['fribbles'])
      speaker = speaker_based_fribble_specific_exp_info.iloc[idx]['speakers'].lower()
      pre_naming_row = naming_task[(naming_task['Pair'] == exp_pair_id) & (naming_task['Fribble_nr'] == exp_fribble) & (naming_task['Session'].str.contains('pre'))]
      post_name_row = naming_task[(naming_task['Pair'] == exp_pair_id) & (naming_task['Fribble_nr'] == exp_fribble) & (naming_task['Session'].str.contains('post'))]
      pre_name.append(pre_naming_row['Name_{}_lemmas'.format(speaker)].values[0])
      post_name.append(post_name_row['Name_{}_lemmas'.format(speaker)].values[0])
   # load uknown words json
   with open('unknown_words.json') as json_file:
      unknown_words = json.load(json_file)
   
   speaker_based_fribble_specific_exp_info['pre_name'] = pre_name
   speaker_based_fribble_specific_exp_info['post_name'] = post_name

   speaker_based_fribble_specific_exp_info['exp_pre_name_lexical_similarity'] = speaker_based_fribble_specific_exp_info.apply(lambda x: measure_distance(x.pre_name, x.content_words_exp), axis=1)
   speaker_based_fribble_specific_exp_info['exp_post_name_lexical_similarity'] = speaker_based_fribble_specific_exp_info.apply(lambda x: measure_distance(x.post_name, x.content_words_exp), axis=1)

   
   speaker_based_fribble_specific_exp_info['exp_post_names_similarity_key'] = speaker_based_fribble_specific_exp_info.apply(lambda x: word2vec_similarity_between_keywords(unknown_words, gen_model,x.post_name.strip().split(), x.content_words_exp.strip().split()), axis=1)
   speaker_based_fribble_specific_exp_info['exp_pre_names_similarity_key'] = speaker_based_fribble_specific_exp_info.apply(lambda x: word2vec_similarity_between_keywords(unknown_words, gen_model,x.pre_name.strip().split(), x.content_words_exp.strip().split()), axis=1)
   
   speaker_based_fribble_specific_exp_info['exp_post_names_similarity'] = speaker_based_fribble_specific_exp_info.apply(lambda x: word2vec_similarity_between_expressions(unknown_words, gen_model,x.post_name.strip().split(), x.content_words_exp.strip().split()), axis=1)
   speaker_based_fribble_specific_exp_info['exp_pre_names_similarity'] = speaker_based_fribble_specific_exp_info.apply(lambda x: word2vec_similarity_between_expressions(unknown_words, gen_model,x.pre_name.strip().split(), x.content_words_exp.strip().split()), axis=1)


   return speaker_based_fribble_specific_exp_info

def distance_between_utterances_across_rounds_and_pre_post_names(turns_info, naming_task):
   ID_pattern = r"\d+"
   turns_utterances_names = []
   for pair in turns_info.keys():
      # get the corresponding pre&post names for the target fribble at hand 
      turn_pair_id = int(re.findall(ID_pattern, pair)[0])
      for turn in turns_info[pair]:
         turn_fribble = int(turn.target)
         pre_naming_row = naming_task[(naming_task['Pair'] == turn_pair_id) & (naming_task['Fribble_nr'] == turn_fribble) & (naming_task['Session'].str.contains('pre'))]
         post_name_row = naming_task[(naming_task['Pair'] == turn_pair_id) & (naming_task['Fribble_nr'] == turn_fribble) & (naming_task['Session'].str.contains('post'))]
         speaker = 'a' if turn.speaker == 'A' else 'b'
         turns_utterances_names.append({'pre_name': pre_naming_row['Name_{}_lemmas'.format(speaker)].values[0], 'post_name': post_name_row['Name_{}_lemmas'.format(speaker)].values[0], 'turn_ID': turn.ID, 'pair': pair, 'speaker': speaker, 'utterance': turn.lemmas_sequence, 'round': turn.round, 'target': turn.target})
   turns_utterances_names = pd.DataFrame(turns_utterances_names)
   turns_utterances_names['exp_pre_name_lexical_distance'] = turns_utterances_names.apply(lambda x: measure_distance(x.pre_name, x.utterance), axis=1)
   turns_utterances_names['exp_post_name_lexical_distance'] = turns_utterances_names.apply(lambda x: measure_distance(x.post_name, x.utterance), axis=1)
   turns_utterances_names['exp_pre_name_levenshtein_distance'] = turns_utterances_names.apply(lambda x: edit_distance(x.pre_name.split(), x.utterance.split()), axis=1)
   turns_utterances_names['exp_post_name_levenshtein_distance'] = turns_utterances_names.apply(lambda x: edit_distance(x.post_name.split(), x.utterance.split()), axis=1)
   
   ax = sns.lineplot(data=turns_utterances_names, x="round", y="exp_post_name_lexical_distance", label='Similarity with post names')#, hue="speaker")
   ax = sns.lineplot(data=turns_utterances_names, x="round", y="exp_pre_name_lexical_distance", label='Similarity with pre names')#, hue="speaker")
   sns.set(font_scale=1.2)  
   ax.figure.set_size_inches(9, 5)
   # # Add title and axis names
   # ax.set_yticks([0, 0.1, 0.2])
   ax.set_title("Lexical Cosine similarity between all utterances and pre&post-interaction names", loc='left', fontsize=15)
   ax.set_xlabel('Round', fontsize=15)
   ax.set_ylabel('Lexical cosine similarity - score', fontsize=15)
   return turns_utterances_names
   # # Add legend
# print(turns_info['pair04'][0].target)

def word2vec_similarity_spacy(a, b):
    return a.similarity(b)

def compute_distance_between_two_words(unknown_words, word1, word2, gen_model, new_unknown_words=[]):
    word1 = word1.lower()
    word2 = word2.lower()
    try:
        return gen_model.similarity(word1, word2)
    except Exception as e0:
      try:
          gen_model.similarity(word1, 'hond')
          problem_word = word2
          good_word = word1
      except Exception as e1:
          problem_word = word1
          good_word = word2
      # check if the problem word is in the unknown words mapping
      if not problem_word in unknown_words.keys():
         print(problem_word)
      # print('Error: {}'.format(e))
      try:
         unknown_words_mapping = unknown_words[word1]
         
         return np.max([gen_model.similarity(unknown_word.lower(), word2) for unknown_word in unknown_words_mapping])
      except Exception as e1:
         # print('Error: {}'.format(e))
         try:
            # print('tried to find mapping for word1: {}, but failed'.format(word1))
            unknown_words_mapping = unknown_words[word2]
            return np.max([gen_model.similarity(word1, unknown_word.lower()) for unknown_word in unknown_words_mapping])
         except Exception as e2:
            # print('Error: {}'.format(e))
            # print('both words are unknown')
            try:
               unknown_words_mapping1 = unknown_words[word1]
               unknown_words_mapping2 = unknown_words[word2]
               return np.max([gen_model.similarity(unknown_word1.lower(), unknown_word2.lower()) for unknown_word1 in unknown_words_mapping1 for unknown_word2 in unknown_words_mapping2])
            except Exception as e:
               return 0
def word2vec_similarity_between_keywords(unknown_words, gen_model, a_words, b_words):
    return np.max([ compute_distance_between_two_words(unknown_words, a, b, gen_model) for a in a_words for b in b_words])
def word2vec_similarity_between_expressions(unknown_words, gen_model, a_words, b_words):
      return np.mean([ compute_distance_between_two_words(unknown_words, a, b, gen_model) for a in a_words for b in b_words])

def measure_distance_between_shared_exp_and_keywords_per_num_rounds(naming_task, fribble_specific_exp_info, gen_model):
   # load the mapping from a json file
   with open('unknown_words.json', 'r') as fp:
      unknown_words = json.load(fp)
   fribble_specific_exp_info['int_pair'] = fribble_specific_exp_info['pair'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
   
   fribble_specific_exp_info['pre_name_b'] = fribble_specific_exp_info[['int_pair', 'target_fribble']].apply(lambda x: naming_task[(naming_task['Pair'] == x.int_pair) & (naming_task['Fribble_nr'] == x.target_fribble) & (naming_task['Session'] == 'pre')]['Name_b_lemmas'].values[0], axis=1)

   
   fribble_specific_exp_info['post_name_b'] = fribble_specific_exp_info[['int_pair', 'target_fribble']].apply(lambda x: naming_task[(naming_task['Pair'] == x.int_pair) & (naming_task['Fribble_nr'] == x.target_fribble) & (naming_task['Session'] == 'post')]['Name_b_lemmas'].values[0], axis=1)
   fribble_specific_exp_info['post_name_a'] = fribble_specific_exp_info[['int_pair', 'target_fribble']].apply(lambda x: naming_task[(naming_task['Pair'] == x.int_pair) & (naming_task['Fribble_nr'] == x.target_fribble) & (naming_task['Session'] == 'post')]['Name_a_lemmas'].values[0], axis=1)
   fribble_specific_exp_info['pre_name_a'] = fribble_specific_exp_info[['int_pair', 'target_fribble']].apply(lambda x: naming_task[(naming_task['Pair'] == x.int_pair) & (naming_task['Fribble_nr'] == x.target_fribble) & (naming_task['Session'] == 'pre')]['Name_a_lemmas'].values[0], axis=1)
   
   fribble_specific_exp_info['pre_names_lexical_similarity'] = fribble_specific_exp_info.apply(lambda x: measure_distance(x.pre_name_a, x.pre_name_b), axis=1)
   fribble_specific_exp_info['post_names_lexical_similarity'] = fribble_specific_exp_info.apply(lambda x: measure_distance(x.post_name_a, x.post_name_b), axis=1)

   fribble_specific_exp_info['post_names_similarity_key'] = fribble_specific_exp_info.apply(lambda x: word2vec_similarity_between_keywords(unknown_words, gen_model, x['post_name_a'].strip().split(' '), x['post_name_b'].strip().split(' ')), axis=1)
   fribble_specific_exp_info['pre_names_similarity_key'] = fribble_specific_exp_info.apply(lambda x: word2vec_similarity_between_keywords(unknown_words, gen_model, x['pre_name_a'].strip().split(' '), x['pre_name_b'].strip().split(' ')), axis=1)
   
   fribble_specific_exp_info['exp_post_names_similarity_key'] = fribble_specific_exp_info.apply(lambda x: (word2vec_similarity_between_keywords(unknown_words, gen_model,x.post_name_a.strip().split(), x.content_words_exp.strip().split()) + word2vec_similarity_between_keywords(unknown_words, gen_model, x.post_name_b.strip().split(), x.content_words_exp.strip().split()))/2, axis=1)
   fribble_specific_exp_info['exp_pre_names_similarity_key'] = fribble_specific_exp_info.apply(lambda x: (word2vec_similarity_between_keywords(unknown_words, gen_model,x.pre_name_a.strip().split(), x.content_words_exp.strip().split()) + word2vec_similarity_between_keywords(unknown_words, gen_model, x.pre_name_b.strip().split(), x.content_words_exp.strip().split()))/2, axis=1)

   fribble_specific_exp_info['post_names_similarity'] = fribble_specific_exp_info.apply(lambda x: word2vec_similarity_between_expressions(unknown_words, gen_model, x['post_name_a'].strip().split(' '), x['post_name_b'].strip().split(' ')), axis=1)
   fribble_specific_exp_info['pre_names_similarity'] = fribble_specific_exp_info.apply(lambda x: word2vec_similarity_between_expressions(unknown_words, gen_model, x['pre_name_a'].strip().split(' '), x['pre_name_b'].strip().split(' ')), axis=1)
   
   fribble_specific_exp_info['exp_post_names_similarity'] = fribble_specific_exp_info.apply(lambda x: (word2vec_similarity_between_expressions(unknown_words, gen_model,x.post_name_a.strip().split(), x.content_words_exp.strip().split()) + word2vec_similarity_between_expressions(unknown_words, gen_model, x.post_name_b.strip().split(), x.content_words_exp.strip().split()))/2, axis=1)
   fribble_specific_exp_info['exp_pre_names_similarity'] = fribble_specific_exp_info.apply(lambda x: (word2vec_similarity_between_expressions(unknown_words, gen_model,x.pre_name_a.strip().split(), x.content_words_exp.strip().split()) + word2vec_similarity_between_expressions(unknown_words, gen_model, x.pre_name_b.strip().split(), x.content_words_exp.strip().split()))/2, axis=1)

   return fribble_specific_exp_info, naming_task


def measure_distance_between_shared_exp_and_keywords_per_speaker_for_all_data(naming_task, speaker_based_fribble_specific_exp_info):
   # load the mapping from a json file
   with open('unknown_words.json', 'r') as fp:
      unknown_words = json.load(fp)
   speaker_based_fribble_specific_exp_info = speaker_based_fribble_specific_exp_info.reset_index(drop=True)
   ID_pattern = r"\d+"
   IDs = []
   pre_name = []
   post_name = []
   for idx in range(0, len(speaker_based_fribble_specific_exp_info)): 
      # exp_pair_id = int(re.findall(ID_pattern, speaker_based_fribble_specific_exp_info.iloc[idx]['pair'])[0])
      exp_pair_id = speaker_based_fribble_specific_exp_info.iloc[idx]['int_pair']
      exp_fribble = int(speaker_based_fribble_specific_exp_info.iloc[idx]['fribbles'])
      naming_row = naming_task[(naming_task['Pair'] == exp_pair_id) & (naming_task['Fribble_nr'] == exp_fribble)]
      speaker = speaker_based_fribble_specific_exp_info.iloc[idx]['speakers'].lower()
      
      pre_name.append(naming_row['pre_name_{}_lemmas'.format(speaker)].values[0])
      post_name.append(naming_row['post_name_{}_lemmas'.format(speaker)].values[0])
   
      
   # load uknown words json
   with open('unknown_words.json') as json_file:
      unknown_words = json.load(json_file)
   
   speaker_based_fribble_specific_exp_info['pre_lemmas'] = pre_name
   speaker_based_fribble_specific_exp_info['post_lemmas'] = post_name
   speaker_based_fribble_specific_exp_info['exp_post_names_lexical_similarity'] = speaker_based_fribble_specific_exp_info.apply(lambda x: measure_distance(x.post_lemmas.strip(), x.exp.strip()), axis=1)
   speaker_based_fribble_specific_exp_info['exp_pre_names_lexical_similarity'] = speaker_based_fribble_specific_exp_info.apply(lambda x: measure_distance(x.pre_lemmas.strip(), x.exp.strip()), axis=1)
   return speaker_based_fribble_specific_exp_info
 
def measure_distance_between_shared_exp_and_keywords_per_speaker(naming_task, speaker_based_fribble_specific_exp_info):
   # load the mapping from a json file
   with open('unknown_words.json', 'r') as fp:
      unknown_words = json.load(fp)
   speaker_based_fribble_specific_exp_info = speaker_based_fribble_specific_exp_info.reset_index(drop=True)
   ID_pattern = r"\d+"
   IDs = []
   pre_name = []
   post_name = []
   for idx in range(0, len(speaker_based_fribble_specific_exp_info)): 
      # exp_pair_id = int(re.findall(ID_pattern, speaker_based_fribble_specific_exp_info.iloc[idx]['pair'])[0])
      exp_pair_id = speaker_based_fribble_specific_exp_info.iloc[idx]['int_pair']
      exp_fribble = int(speaker_based_fribble_specific_exp_info.iloc[idx]['fribbles'])
      naming_row = naming_task[(naming_task['Pair'] == exp_pair_id) & (naming_task['Fribble_nr'] == exp_fribble)]
      speaker = speaker_based_fribble_specific_exp_info.iloc[idx]['speakers'].lower()
      
      pre_name.append(naming_row['pre_name_{}_lemmas'.format(speaker)].values[0])
      post_name.append(naming_row['post_name_{}_lemmas'.format(speaker)].values[0])
   
      
   # load uknown words json
   with open('unknown_words.json') as json_file:
      unknown_words = json.load(json_file)
   
   speaker_based_fribble_specific_exp_info['pre_lemmas'] = pre_name
   speaker_based_fribble_specific_exp_info['post_lemmas'] = post_name
   
   speaker_based_fribble_specific_exp_info['exp_post_names_lexical_similarity'] = speaker_based_fribble_specific_exp_info.apply(lambda x: measure_distance(x.post_lemmas.strip(), x.label.strip()), axis=1)
   speaker_based_fribble_specific_exp_info['exp_pre_names_lexical_similarity'] = speaker_based_fribble_specific_exp_info.apply(lambda x: measure_distance(x.pre_lemmas.strip(), x.label.strip()), axis=1)
   
   return speaker_based_fribble_specific_exp_info


def measure_distance_between_shared_exp_and_names_per_num_rounds(fribble_specific_exp_info, nlp, tokenizer, model, cos):
   naming_task = get_naming_task(nlp)
   fribble_specific_exp_info['int_pair'] = fribble_specific_exp_info['pair'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
   fribble_specific_exp_info['pre_Name_a_lemmas'] = fribble_specific_exp_info[['int_pair', 'target_fribble']].apply(lambda x: naming_task[(naming_task['Pair'] == x.int_pair) & (naming_task['Fribble_nr'] == x.target_fribble) & (naming_task['Session'] == 'pre')]['Name_a_lemmas'].values[0], axis=1)
   fribble_specific_exp_info['pre_Name_b_lemmas'] = fribble_specific_exp_info[['int_pair', 'target_fribble']].apply(lambda x: naming_task[(naming_task['Pair'] == x.int_pair) & (naming_task['Fribble_nr'] == x.target_fribble) & (naming_task['Session'] == 'pre')]['Name_b_lemmas'].values[0], axis=1)
   fribble_specific_exp_info['pre_names_distance'] = fribble_specific_exp_info.apply(lambda x: measure_distance(x.pre_Name_a_lemmas, x.pre_Name_b_lemmas), axis=1)
   fribble_specific_exp_info['post_Name_a_lemmas'] = fribble_specific_exp_info[['int_pair', 'target_fribble']].apply(lambda x: naming_task[(naming_task['Pair'] == x.int_pair) & (naming_task['Fribble_nr'] == x.target_fribble) & (naming_task['Session'] == 'post')]['Name_a_lemmas'].values[0], axis=1)
   fribble_specific_exp_info['post_Name_b_lemmas'] = fribble_specific_exp_info[['int_pair', 'target_fribble']].apply(lambda x: naming_task[(naming_task['Pair'] == x.int_pair) & (naming_task['Fribble_nr'] == x.target_fribble) & (naming_task['Session'] == 'post')]['Name_b_lemmas'].values[0], axis=1)
   fribble_specific_exp_info['post_names_similarity'] = fribble_specific_exp_info.apply(lambda x: measure_distance(x.post_Name_a_lemmas, x.post_Name_b_lemmas), axis=1)
   fribble_specific_exp_info['exp_post_name_similarity'] = fribble_specific_exp_info.apply(lambda x: (measure_distance(x.post_Name_a_lemmas, x.label) + measure_distance(x.post_Name_b_lemmas, x.label))/2, axis=1)
   fribble_specific_exp_info['exp_pre_name_similarity'] = fribble_specific_exp_info.apply(lambda x: (measure_distance(x.pre_Name_a_lemmas, x.label) + measure_distance(x.pre_Name_b_lemmas, x.label))/2, axis=1)
   fribble_specific_exp_info['similarity_diff_with_exp'] = fribble_specific_exp_info['exp_post_name_similarity'] - fribble_specific_exp_info['exp_pre_name_similarity']
   fribble_specific_exp_info['post_name_similarity'] = fribble_specific_exp_info.apply(lambda x: measure_distance(x.post_Name_a_lemmas, x.post_Name_b_lemmas), axis=1)
   fribble_specific_exp_info['pre_name_similarity'] = fribble_specific_exp_info.apply(lambda x: measure_distance(x.pre_Name_a_lemmas, x.pre_Name_b_lemmas), axis=1)
   fribble_specific_exp_info['name_similarity_diff'] = fribble_specific_exp_info['post_name_similarity'] - fribble_specific_exp_info['pre_name_similarity']

   fribble_specific_exp_info['post_name_semantic_similarity'] = measure_semantic_cosine_similarity_based_on_BERT_embeddings_between_two_list_of_expressions(fribble_specific_exp_info['post_Name_a_lemmas'].to_list(), fribble_specific_exp_info['post_Name_b_lemmas'].to_list(), model, tokenizer, cos)
   fribble_specific_exp_info['pre_name_semantic_similarity'] = measure_semantic_cosine_similarity_based_on_BERT_embeddings_between_two_list_of_expressions(fribble_specific_exp_info['pre_Name_a_lemmas'].to_list(), fribble_specific_exp_info['pre_Name_b_lemmas'].to_list(), model, tokenizer, cos)
   fribble_specific_exp_info['name_semantic_similarity_diff'] = fribble_specific_exp_info['post_name_semantic_similarity'] - fribble_specific_exp_info['pre_name_semantic_similarity']

   fribble_specific_exp_info['speaker_name_change'] = fribble_specific_exp_info.apply(lambda x: (measure_distance(x.pre_Name_a_lemmas, x.post_Name_a_lemmas) + measure_distance(x.pre_Name_b_lemmas, x.post_Name_b_lemmas))/2, axis=1)
   
   return fribble_specific_exp_info, naming_task
# check if the establisher is the speaker
def speaker_reuse(row, fribble_specific_exp_info, turns_info, all=False):
   if all:
      label = row['exp']
      fribble_specific_exp_info_row = fribble_specific_exp_info[(fribble_specific_exp_info['exp'] == label) & (fribble_specific_exp_info['pair'] == row['pair']) & (fribble_specific_exp_info['target_fribble'] == row['target_fribble'])]

   else:
      label = row['label']
      fribble_specific_exp_info_row = fribble_specific_exp_info[(fribble_specific_exp_info['label'] == label) & (fribble_specific_exp_info['pair'] == row['pair']) & (fribble_specific_exp_info['target_fribble'] == row['target_fribble'])]

      
   # find the corresponding row in fribble_specific_exp_info
   estab_turn = fribble_specific_exp_info_row['estab_turn'].values[0]
   turns = fribble_specific_exp_info_row['turns'].values[0]
   pair = fribble_specific_exp_info_row['pair'].values[0]
   speaker = row['speakers']
   # count the nubmers of turns for a speaker after estab_turn
   speaker_rounds = []
   for turn in turns:
      if turns_info[pair][turn].speaker == speaker:
         speaker_rounds.append(turns_info[pair][turn].round)
   speaker_rounds_counter = np.unique(speaker_rounds).shape[0]
   # calculate speaker frequency
   speaker_turns = []
   for turn in turns:
      if turns_info[pair][turn].speaker == speaker:
         speaker_turns.append(turn)
   number_of_turns = len(speaker_turns)
   # check if the speaker is also the establisher
   if turns_info[pair][estab_turn].speaker == speaker:
      is_establisher = 1
   else:
      is_establisher = 0
   # check if the speaker is also the initiator
   if turns_info[pair][turns[0]].speaker == speaker:
      is_initiator = 1
   else:
      is_initiator = 0
   # select speaker's pre and post name
   # last time the speaker used the shared expressions
   last_round = turns_info[pair][speaker_turns[-1]].round
   first_round = turns_info[pair][speaker_turns[0]].round
   return speaker_rounds_counter, is_establisher, is_initiator, number_of_turns, last_round, first_round
  
  

def prepare_per_speaker_for_all_data(naming_task, fribble_specific_exp_info, turns_info):
   
   # 'label', 'content_words_exp', 'expressions_set', '#shared expressions', 'Pairs'
   fribble_specific_exp_info['A_freq'] = fribble_specific_exp_info.apply(lambda x: x['speakers'].count('A'), axis=1)
   fribble_specific_exp_info['B_freq'] = fribble_specific_exp_info.apply(lambda x: x['speakers'].count('B'), axis=1)
   speaker_based_fribble_specific_exp_info= fribble_specific_exp_info.explode(['speakers', 'rounds', 'turns', 'fribbles'])

   speaker_turn_round = speaker_based_fribble_specific_exp_info[['exp', 'fribbles', 'last_round', 'first_round', 'int_pair', 'pair', 'target_fribble', 'speakers','rounds', 'estab_turn', 'estab_round', 'turns', 'Pairs']]
   # calculate the speaker re-use after estab_turn 
   rounds_establisher_initiator = speaker_turn_round.apply(lambda x: speaker_reuse(x, fribble_specific_exp_info, turns_info, all=True), axis=1)
   # rounds_establisher_initiator is a tuple, select the first column
   speaker_turn_round['Speaker Number of Rounds'] = [x[0] for x in rounds_establisher_initiator]
   speaker_turn_round['is_establisher'] = [x[1] for x in rounds_establisher_initiator]
   speaker_turn_round['is_initiator'] = [x[2] for x in rounds_establisher_initiator]
   speaker_turn_round['Speaker Number of Turns'] = [x[3] for x in rounds_establisher_initiator]
   speaker_turn_round['Speaker Last Round'] = [x[4] for x in rounds_establisher_initiator]
   speaker_turn_round['Speaker First Round'] = [x[5] for x in rounds_establisher_initiator]

   speaker_turn_round = measure_distance_between_shared_exp_and_keywords_per_speaker_for_all_data(naming_task, speaker_turn_round)
 
   return speaker_turn_round 
def prepare_per_speaker_data(naming_task, fribble_specific_exp_info, turns_info):
   fribble_specific_exp_info['A_freq'] = fribble_specific_exp_info.apply(lambda x: x['speakers'].count('A'), axis=1)
   fribble_specific_exp_info['B_freq'] = fribble_specific_exp_info.apply(lambda x: x['speakers'].count('B'), axis=1)
   speaker_based_fribble_specific_exp_info= fribble_specific_exp_info.explode(['speakers', 'rounds', 'turns', 'fribbles'])

   speaker_turn_round = speaker_based_fribble_specific_exp_info[['label', 'content_words_exp', 'fribbles', 'expressions_set', '#shared expressions', 'last_round', 'first_round', 'int_pair', 'pair', 'target_fribble', 'speakers','rounds', 'estab_turn', 'estab_round', 'turns', 'Pairs']].groupby(['label', 'content_words_exp', 'fribbles', 'expressions_set', '#shared expressions' , 'last_round', 'first_round', 'int_pair', 'pair', 'target_fribble', 'speakers', 'estab_turn', 'estab_round', 'Pairs']).agg({'rounds': 'count', 'turns': 'count'}).reset_index().sort_values(by=['label', 'target_fribble', 'speakers', 'estab_turn', 'estab_round'])
   # import Counter
   
   # calculate the speaker re-use after estab_turn 
   rounds_establisher_initiator = speaker_turn_round.apply(lambda x: speaker_reuse(x, fribble_specific_exp_info, turns_info), axis=1)
   # rounds_establisher_initiator is a tuple, select the first column
   speaker_turn_round['Speaker Number of Rounds'] = [x[0] for x in rounds_establisher_initiator]
   speaker_turn_round['is_establisher'] = [x[1] for x in rounds_establisher_initiator]
   speaker_turn_round['is_initiator'] = [x[2] for x in rounds_establisher_initiator]
   speaker_turn_round['Speaker Number of Turns'] = [x[3] for x in rounds_establisher_initiator]
   speaker_turn_round['Speaker Last Round'] = [x[4] for x in rounds_establisher_initiator]
   speaker_turn_round['Speaker First Round'] = [x[5] for x in rounds_establisher_initiator]

   speaker_turn_round = measure_distance_between_shared_exp_and_keywords_per_speaker(naming_task, speaker_turn_round)
 
   return speaker_turn_round


def measure_distance_between_shared_exp_and_names(speaker_based_fribble_specific_exp_info):
   # fig, axs = plt.subplots(nrows=1, figsize=(9, 10))
   # fig.tight_layout(pad=3.0)
   ax1 = sns.lineplot(data=speaker_based_fribble_specific_exp_info, x="rounds", y="exp_post_name_distance", label='Similarity with post names')
   ax1 = sns.lineplot(data=speaker_based_fribble_specific_exp_info, x="rounds", y="exp_pre_name_distance", label='Similarity with pre names')
   sns.set(font_scale=1.5)  
   # # Add title and axis names
   # ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
   ax1.set_title("Cosine lexical similarity between shared expressions over rounds and pre&post interaction names", loc='left', fontsize=15)
   ax1.set_xlabel('Rounds where expressions are used', fontsize=15)
   ax1.set_ylabel('Lexical cosine similarity - score', fontsize=15)


if __name__ == "__main__":
   nlp = spacy.load("nl_core_news_lg")
   naming_task = get_naming_task(nlp=nlp)
  