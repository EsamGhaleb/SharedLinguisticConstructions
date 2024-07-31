import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_similarity_of_shared_constructions_to_pre_and_post_names(real_speaker_turn_round, real_turns_utterances_names, pseudo_turns_utterances_names, Pairs='Real'):
    # for each speaker, calculate the percentage of utterances that contain shared expressions per round
    real_turns_utterances_names['utterance_contains_shared_exp_percentage'] = real_turns_utterances_names.groupby(['pair', 'speaker', 'round'])['contains_shared_exp'].transform(lambda x: x[x == 'Contains shared exp'].count()/x.count())
    # group by pair, speaker and round and calculate the mean of the percentage of utterances that contain shared expressions
    pseudo_turns_utterances_names['utterance_contains_shared_exp_percentage'] = pseudo_turns_utterances_names.groupby(['pair', 'speaker', 'round'])['contains_shared_exp'].transform(lambda x: x[x == 'Contains shared exp'].count()/x.count())

    # concatenate the two dataframes
    real_turns_utterances_names['Pairs'] = 'Real'
    pseudo_turns_utterances_names['Pairs'] = 'Pseudo'

    turns_utterances_names = pd.concat([real_turns_utterances_names, pseudo_turns_utterances_names])
    # reindex the data
    turns_utterances_names = turns_utterances_names.reset_index(drop=True)

    # Set the size of the plot
    plt.figure(figsize=(9, 5))
    # convert the round to int
    turns_utterances_names['rounds'] = turns_utterances_names['round'].astype(int)
    # Plotting lines for shared expression similarity
    sns.lineplot(data=real_speaker_turn_round, x="rounds", y="exp_post_names_lexical_similarity", 
                label='Post-names & shared constructions', color='red', marker='o', linestyle='-')  # Solid line for post
    sns.lineplot(data=real_speaker_turn_round, x="rounds", y="exp_pre_names_lexical_similarity", 
                label='Pre-names & shared constructions', color='darkblue', marker='x', linestyle='-')  # Dashed line for pre

    # Add title and axis names
    plt.ylabel('Similarity score', fontsize=15, fontweight='bold')

    # Add title and axis names
    # plt.title('Percentage of Utterances with Shared Expressions', fontsize=15)
    plt.xlabel('')
    # replace the x ticks with the round number: R1, R2, R3, R4, R5, R6
    plt.xticks([1, 2, 3, 4, 5, 6], ['R1', 'R2', 'R3', 'R4', 'R5', 'R6'], fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    # Display the plot
    # Set font weight for tick labels
    plt.yticks(fontweight='bold')

    # Set font weight for legend
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_fontweight('bold')

    # Display the plot
    plt.show()