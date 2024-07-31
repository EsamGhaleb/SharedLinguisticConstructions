def calculate_main_simple_feats(pre_post_names, fribble_specific_exp_info):
   # recency feature
   # convert last_round to int
   fribble_specific_exp_info['last_round'] = fribble_specific_exp_info['last_round'].astype(int)
   # convert freq to int
   fribble_specific_exp_info['freq'] = fribble_specific_exp_info['freq'].astype(int)
   # convert number of rounds to int
   fribble_specific_exp_info['number of rounds'] = fribble_specific_exp_info['number of rounds'].astype(int)
   for i, row in pre_post_names.iterrows():
      this_pair_int = row['Pair']
      this_fribble_int = row['Fribble_nr']
      # print(this_fribble_int, this_pair_int)
      # get the features of this fribble in this pair
      this_fribble_features = fribble_specific_exp_info[(fribble_specific_exp_info['int_pair'] == this_pair_int) & (fribble_specific_exp_info['target_fribble'] == this_fribble_int)][['number of rounds', 'estab_round', 'freq', 'last_round']]
      if this_fribble_features.shape[0] == 0:
         continue
      # print(this_fribble_features)
      # get the prominent exp with the highest number of rounds
      max_row = this_fribble_features[this_fribble_features['number of rounds'] == this_fribble_features['number of rounds'].max()] 
      pre_post_names.loc[i, 'Last Round'] = max_row['last_round'].values[0]
      pre_post_names.loc[i, 'Number of Rounds'] = max_row['number of rounds'].values[0]
      pre_post_names.loc[i, 'Frequency'] = max_row['freq'].values[0]
      pre_post_names.loc[i, 'Establishment Round'] = max_row['estab_round'].values[0]
      # get average features
      pre_post_names.loc[i, 'Average Last Round'] = this_fribble_features['last_round'].mean()
      pre_post_names.loc[i, 'Average Establishment Round'] = this_fribble_features['estab_round'].mean()
      pre_post_names.loc[i, 'Average Number of Rounds'] = this_fribble_features['number of rounds'].mean()
      pre_post_names.loc[i, 'Average Frequency'] = this_fribble_features['freq'].mean()
             
   return pre_post_names