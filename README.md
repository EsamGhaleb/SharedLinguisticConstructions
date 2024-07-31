# [Analysing Cross-Speaker Convergence in Face-to-Face Dialogue through the Lens of Automatically Detected Shared Linguistic Constructions](https://escholarship.org/uc/item/43h970fc)

This repository include code for the paper titled "[Analysing Cross-Speaker Convergence in Face-to-Face Dialogue through the Lens of Automatically Detected Shared Linguistic Constructions](https://escholarship.org/uc/item/43h970fc)" published in CogSci2024.

The repository contains the data processing methods used to operationalize linguistic alignment and the results reported in the paper. Linguistic alignment is detected based on a lemmatized speech at the level of the lemmas' form. In a dialogue, single lemmas or sequences of lemmas used at least once by **both** participants are defined as shared construction and considered instances of lexical alignment.

If you are interested in the data and the results, please refer to the last part of the this file.

# **Libraries**

You caninstall the required libaries for this repository using pip:

```
pip install -r requirements.txt
```

You will also need Torch, WhisperX, and Spacy. For these packages, please follow the following instructions:

* **GPU Support for Torch** : If you want to install PyTorch with GPU support, follow the instructions on the official[ PyTorch website](https://pytorch.org/). The command might look like this:

```
pip install torch torchvision torchaudio
```

* **WhisperX:** Please check [WhisperX repository](https://github.com/m-bain/whisperX) for the updated instructions. You can install this package as follows:

```
pip install git+https://github.com/m-bain/whisperx.git
```

* **Spacy Models** : For Spacy, you might need to download the language model separately. For example:

```
python -m spacy download nl_core_news_lg
```

# Preprocessing CABB Datasets

Before extracting the shared constructions, we pre-process the CABB dialogues using the scripts in `code/utils/data_containers_forced_alignment.py.` First, we read the Pairs data, including turns’ utterances and their corresponding information (e.g., speaker, time, co-speech gestures, etc.). Second, we pre-process speech utterances as follows:

* **Word-level time alignment:** Phoneme sequences are aligned with the corresponding transcriptions. This is done using WAV2VEC, a speech recognition model, to extract phonemes’ representations of audio raw wav frames and Connectionist Temporal Classification (CTC). For this purpose, we use the [wav2vec2-large-xlsr-53-dutch](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-dutch). Next, we utilize CTC to align the phonemes’ representations with the corresponding words, as proposed by [Graves et al., 2006](https://link.springer.com/book/10.1007/978-3-642-24797-2) and improved by [Kürzinger et al., 2020](https://arxiv.org/pdf/2007.09127.pdf). We utilized the implementation of the CTC-segmentation algorithm provided by [PyTorch tutorial on forced alignment](https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html). The resulting alignment is of high quality, which can be useful for studying co-speech gestures and other linguistic and co-speech phenomena. [N.B.: ***This information is not exploited in the analyses summarised in this report***]
* **Removal of **dysfluencies**:** We remove tokens between and with special characters, e.g., #, *, -, etc. This removes non-verbal vocal sounds, self-repairs, and filled pauses. For example:

- **We remove self repairs, e.g., “ku-” in “aan de zijkant twee ku- kubussen”**.
  - Non-verbal vocal sounds are removed, e.g., #laughs#, #sigh#, #click on the computer#, #tongue click#, etc.
  - We remove inaudible speech, e.g., “(?)” in “deze heeft een soort van (?) aan de zijkant”. However, if annotators transcribed the inaudible speech, we keep it, e.g., the utterance “aan de (zijkant),” is converted to “aan de zijkant”.
  - We remove filled pauses: e.g., "uh", "hmm", etc. For this, we use the list of interactional words accompanying the CABB dataset.
- **Part of speech (POS) tagging and lemmatisation:** We perform POS tagging per word (e.g., NOUN, VERB, etc.), and words are lemmatised into the base forms, e.g.:
  - heeft → hebben
  - Diamantjes → diamant

You can run the pre-processing and data preperation script as follows:

```
pythondata_containers_forced_alignment.py --spacy_model nl_core_news_lg --audio_path 'path_to_data/{}synced_pp{}.wav' --elan_path_small 'path_to_data/ElanFiles/New_ELAN_files/pair*.eaf' --elan_path_large 'path_to_data/ElanFiles/New_ELAN_files/*complete.eaf' --output_path 'lex_code/dialign/targets_riws_lemma{}/'
```

Note that we distinguish between two CABB versions. The **small** version refers to the data collected by [Rasenberg et al. ](https://www.tandfonline.com/doi/full/10.1080/0163853X.2021.1992235), and the **large** which refers to the [publically available CABB dataset](https://www.sciencedirect.com/science/article/pii/S1053811922008552). Both datasets have slightly different setups, such as the number of trails, the coding of self-repairs, and annotations of non-verbal sounds, which required special ways to handle these differences. Please check the code for more details.

To access the small or large CABB dataset, please contact [Marlou Rasenberg ](https://scholar.google.com/citations?user=MUCqxGAAAAAJ)and [Sara Bögels](https://www.ru.nl/en/people/bogels-s)!

# Extracting Shared Constructions

For each dialogue (i.e., each participant pair) and each given fribble, we do the following:

* We combine all sections of the dialogue (across rounds) where participants are trying to identify the given fribble.
* **The result dialogu sections per object for each pair are stored** in `code/dialign/targets_riws_lemma_lg.` Each file in this directory, contain lemmatised utterances of dialogues per target (i.e., object). The lemmetisation was applied[ through using the large model (lg) of Spacy.](https://spacy.io/models/nl#nl_core_news_lg)
* We extract all the sequences of lemmas that both dialogue participants have used across these sections of the dialogue (including sequences of length 1, i.e., single lemmas). For this step, we use the sequential pattern-matching algorithm proposed by [Duplessis et al](https://link.springer.com/article/10.1007/s10579-021-09532-w). We refer to these sequences of lemmas as **shared constructions**. We use [DialAlign](https://github.com/GuillaumeDD/dialign) package to extract these sequences
* Once you have the dialogues per object, in `code/dialign/targets_riws_lemma_lg/`, you can the following command to extract shared constructions with the format proposed used in Dialign:
* ```
  java -jar dialign.jar -i targets_riws_lemma_lg -o output_targets_riws_lemma_lg
  ```

# Processing and Linking Shared Constructions

## Clearning and Filtering Shared constructions

* We filter out shared constructions consisting exclusively of function words and other highly frequent words (e.g., `ja dat zijn'):
* Function words: To identify function words, we rely on POS tagging and exclude shared constructions that exclusively consist of them. The Universal POS tagset serves as the basis for our list of POS tags, which includes ['DET', 'PRON', 'ADP', 'CCONJ', 'SCONJ', 'AUX', 'PART', 'PUNCT', 'SYM', 'X', 'INTJ']. However, our approach can miss sequences containing only function words due to inaccurate POS tagging. To address this issue, we also refer to the function word list provided by the CABB dataset as a supplementary step in identifying and removing shared constructions with only function words.
* Highly frequent words: following [Marlou et al.](https://www.tandfonline.com/doi/full/10.1080/0163853X.2021.1992235), highly frequent words were determined using the word frequencies in the SUBTLEX-NL corpus by [Keuleers et al.](https://link.springer.com/article/10.3758/BRM.42.3.643). We used three standard deviations above the mean as a threshold for highly frequent words.
* We restrict our attention to shared constructions that are used for one single fribble within the entire dialogue, thus leaving aside shared constructions used for multiple fribbles in the current analyses (we may want to look into those in the future). Our rationale is the following: If a construction is repeated by both participants for several fribbles, it is likely to be a frequent construction (with a high probability of being reused) that our previous filters were not able to discard. This results in a set of unique shared constructions per fribble, i.e., the sequences of contentful lemmas used by both participants for each fribble exclusively.

## Some Terminology

* Initiation turn/round: The turn or round where a shared construction was first introduced by one of the participants.
* Establishing turn/round: The turn or round where the other participant, i.e. first repeated the shared construction, it becomes shared.
* Speaker role: D & M  indicate whether the director or the matcher used the shared construction.
* Initiator role: The participant who introduced the shared construction, i.e., the director or the matcher.
* Establisher role: The participant who established the shared construction, i.e., the director or the matcher.
* Turns and rounds: the turns and rounds numbers where shared constructions are used.
* Length: the number of lemmas in the shared construction.
* Target fribble: the fribble for which the shared construction was used.
* Frequency: the number of times the shared construction was used.

## Linking Shared Constructions with a common lexical core

The methods described above may result in sets of unique shared constructions per fribble that contain highly overlapping words. For example, for fribble 1, pair 10 uses the shared constructions "een spijker" and "spijker bovenop," which overlap significantly. To identify such overlaps, we use regular expressions and link the constructions based on the overlap of content lemmas. We do not link the constructions if the overlapping sequence contains only function words.

This process produces sets of **shared construction types** that reflect a common underlying conceptualization or lexical core, which we consider instances of linguistic alignment. To simplify the labeling of these types, we assign a common lexical core as the label, which is the shortest common sequence for the set.

To illustrate the linking process of shared constructions, consider the example below for Pair 10 and Fribble 1 in the following figure:

![1722418775431](image/README/1722418775431.png)

# Reproducing Paper Results

The three analyses of the paper are included in ` code/notebooks/lexical_alignment_report_final.ipynb`. This scripts make use of already extracted shared constructions, through Dialign package - per dialogue and per object. 

The notebook in ` code/notebooks/lexical_alignment_report_final.ipynb` is self continaid and include all details required to reproduce the results and process the data.

# Reference

If you make use of the code or any materials in this repository, please cite the following paper:

```
@inproceedings{ghaleb2024analysing,
  title={Analysing Cross-Speaker Convergence in Face-to-Face Dialogue through the Lens of Automatically Detected Shared Linguistic Constructions},
  author={Ghaleb, Esam and Rasenberg, Marlou and Pouw, Wim and Toni, Ivan and Holler, Judith and Ozyurek, Asli and Fernandez, Raquel},
  booktitle={Proceedings of the Annual Meeting of the Cognitive Science Society},
  volume={46},
  year={2024}
}
```
