import pandas as pd
import numpy as np
def read_stop_function_words():
   """Reads stop function words from file and returns them as a list."""
   most_common_words = pd.read_excel('../data/words_lists/SUBTLEX-NL.cd-3SDsmean.xlsx')  
   frequencies = most_common_words['FREQlemma'].to_numpy()
   # select the most frequent words based on threshold used in Marlou's study
   highest_frequencies = frequencies >= 85918
   stop_words = most_common_words['Word'].to_numpy()[highest_frequencies]

   function_words = [] 
   with open('../data/words_lists/function_words.txt', encoding="ISO-8859-1") as f:
      lines = f.readlines()
   for word in lines:
      word = word.split("\n")[0]
      function_words.append(word)
   return stop_words, function_words