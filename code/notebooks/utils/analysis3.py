import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_number_of_shared_constructions_and_names_similarity(pre_post_names, feature, measure1):
   # Filter the data for real pairs only
   pre_post_names = pre_post_names[pre_post_names['Pairs'] == 'Real']
   # Filter out rows with missing values in the selected feature and measures
   pre_post_names = pre_post_names.dropna(subset=[feature, measure1])
   # Create a scatter plot with regression line
   plt.figure(figsize=(8, 5))  # Increase the figure size
   # set the color palette
   palette = sns.color_palette("husl", len(pre_post_names[feature].unique()))  # Define a color palette
   joint_plot = sns.stripplot(data=pre_post_names, x=feature, y=measure1, jitter=0.21, palette=palette, size=4, label=measure1, alpha=0.8, hue=feature)  # Create a scatter plot with hue
   # sns.jointplot(data=pre_post_names, x=feature, y=measure1, kind='reg', color='blue', marker='o', scatter_kws={'s': 50}, scatter=False      
   sns.regplot(data=pre_post_names, x=feature, y=measure1, scatter=False, scatter_kws={'color': 'green', 'alpha': 0.7, 's': 20}, line_kws={'color': 'blue', 'alpha': 0.7})  # Customize the regression line
   # rename the y-axis
   plt.xlabel('Number of shared construction types', fontsize=14, fontweight='bold')
   
   # Set plot title and labels
   plt.ylabel('Post-names similarity score', fontsize=14, fontweight='bold')
   plt.xticks(fontsize=14, fontweight='bold')
   plt.yticks(fontsize=14, fontweight='bold')
   
   # Show grid lines
   plt.grid(True)
   
   # remove legend
   plt.legend([],[], frameon=False)
   
   # Show the plot
   plt.tight_layout()
   plt.show()

   # Calculate and display the correlation between the feature and measures
   from scipy.stats import spearmanr
   correlation_measure1, p_value_measure1 = spearmanr(pre_post_names[feature], pre_post_names[measure1], nan_policy='omit')
   print('Correlation between {} and {}: {:.2f} (p-value: {:.4f})'.format(feature, measure1, correlation_measure1, p_value_measure1))


def plot_two_measures_vs_feature(pre_post_names, feature, measure1, measure2):
   # Assuming speaker_turn_round is your DataFrame
   # rename measure 1 and measure 2
   pre_post_names = pre_post_names.rename(columns={measure1: 'Post Names Cosine Similarity', measure2: 'Increase in Names Similarity'})
   # Set the size of the plot
   measure1 = 'Post Names Cosine Similarity'
   measure2 = 'Increase in Names Similarity'
   plt.figure(figsize=(10, 6))
   if feature == 'Number of shared expression types':
      # consider only the ones with shared expressions
      pre_post_names = pre_post_names[pre_post_names['Number of shared expression types'] > 0]
      # rename Number of shared expression types to Number of shared construction types
      pre_post_names = pre_post_names.rename(columns={'Number of shared expression types': 'Number of shared construction types'})
      # remove the ones more than 14
      feature = 'Number of shared construction types'
   pre_post_names = pre_post_names[pre_post_names['Pairs'] == 'Real']
   # Plotting lines for shared expression similarity
   joint_plot = sns.jointplot(data=pre_post_names, x=feature, y=measure1, kind='reg', color='blue', marker='o', scatter_kws={'s': 50}, scatter=False, label=measure1)  # Regression plot for post        
   sns.regplot(data=pre_post_names, x=feature, y=measure1, ax=joint_plot.ax_joint,
            color='green', marker='o', scatter=True, scatter_kws={'s': 50, 'alpha': 0.2}, label=measure1)  # Regression plot for post


   # sns.regplot(data=pre_post_names, x=feature, y=measure2,
            # color='green', marker='x', scatter_kws={'s': 50}, label=measure2)  # Regression plot for pre

   # Enhancing plot aesthetics
   plt.xlabel(feature, fontsize=14, fontweight='bold')
   plt.ylabel('Similarity Score', fontsize=14, fontweight='bold')
   plt.xticks(fontsize=14, fontweight='bold')
   plt.yticks(fontsize=14, fontweight='bold')
   plt.grid(True)

   # Show the plot
   plt.tight_layout()
   plt.show()