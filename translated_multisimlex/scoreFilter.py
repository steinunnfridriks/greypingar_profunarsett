from statistics import mean, median, stdev
from collections import OrderedDict
from scipy import stats
import numpy as np
import copy
import pandas as pd
import operator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Note: The counts and ratios of adverbs, nouns, verbs and adjectives in einkunnir.txt are as follows:
# Adverbs	    123	    0.06514830508
# Nouns	        1051	0.5566737288
# Verbs	        469	    0.2484110169
# Adjectives	245	    0.1297669492

# Constants for dict_of_annotators tuple indexes.
ANN_PAIR = 0
ANN_SCORE = 1
ANN_CAT = 2
# Constants for dict_of_wordpair tuple indexes.
WORD_ANN = 0
WORD_SCORE = 1
WORD_CAT = 2
# Constant filename containing annotator scores.
ANN_FILENAME = "einkunnir.txt"
# Constant for number of participating annotators.
ANN_COUNT = 20


# Loads annotator data into an OrderedDict where key:annotator and value:[(word pair tuple), pair score, pair category]
# File format is as follows (all columns are tab-separated; note that "words" may include whitespaces):
#   Line 1 is headers, line 2 onward is data
#   Column 1: First word in word pair
#   Column 2: Second word in word pair
#   Column 3: Word pair grammatical category
#   Column 4 onward: Annotator scores for this word pair
def load_data():
    # We'll use Pandas, since it handles mixed datatypes (and certain features such as headers) better than NumPy.
    df_loader = pd.read_csv(ANN_FILENAME, sep='\t', header=0)
    return df_loader


def load_annotators_from_df(df):
    # dict_annotators has key: annotator, and value: tuple of (word pair tuple, pair score, pair category)
    list_of_ann = df.columns.values[3:]
    ann_dict = OrderedDict([(a, []) for a in list_of_ann])

    for row in df.itertuples():
        word_tuple = (row[1], row[2])
        pair_category = row[3]
        for ann_index in range(0, ANN_COUNT):
            # Note: Panda adds +1 to index, so we have to use index+4, rather than index+3 for the first pair score.
            ann_dict[list_of_ann[ann_index]].append((word_tuple, row[ann_index + 4], pair_category))
    return ann_dict


def load_wordpairs_from_df(df):
    # dict_of_wordpairs has key: word pair tuple, and value: list of tuples of (annotator, pair score, pair category)
    wp_dict = OrderedDict()
    list_of_ann = df.columns.values[3:]
    # wp_dict = OrderedDict([(a, []) for a in list_of_annotators])
    for row in df.itertuples():
        word_tuple = (row[1], row[2])
        pair_category = row[3]
        value_list = []
        # Note: Panda adds +1 to index, so we have to use index+4, rather than index+3 for the first pair score.
        for ann_index in range(4, ANN_COUNT + 4):
            value_tuple = (list_of_ann[ann_index - 4], row[ann_index], pair_category)
            value_list.append(value_tuple)
        wp_dict[word_tuple] = value_list
    return wp_dict


# Finds Pilehvar outliers based on annotator scores.
# Returns a dictionary of lists: key is annotator name, value is list of all word pair tuples
# graded by that annotator whose scores are Pilehvar outliers from the annotator's mean score.
def find_pilehvar_outliers(dict_wordpairs, ann_list):
    # dict_of_wordpairs has key: word pair tuple, and value: tuple of (annotator, pair score, pair category)
    dict_ann_pairs = OrderedDict()
    for a in ann_list:
        dict_ann_pairs[a] = []

    # For every wordpair...
    # There are some issues with iterating over a "key, value" combo when the key is a tuple,
    # so we'll go for a slightly more verbose approach
    for k_wordpair in dict_wordpairs.keys():
        v_values = dict_wordpairs[k_wordpair]
        for idx, val in enumerate(v_values):
            # Each v_values line is for a specific (word pair tuple) and is drawn from a list of
            # (annotator, score, category) tuples: [('ann1', 0), ('ann2', 3), ('ann3', 5),  ...]
            # Pick one of the list tuples. Find the mean of all the others. Compare the chosen tuple's score to
            # the calculated mean. If it's off by the Pilehvar value, append the chosen tuple (annotator name,
            # score and category - no reason to eliminate the category) to the list in our dict, under the
            # key tuple found in k_wordpair
            idx_less = [v for i, v in enumerate(v_values) if i != idx]
            val_score = val[WORD_SCORE]
            mean_score = float(mean([s[WORD_SCORE] for s in idx_less]))
            if (val_score >= (mean_score + 1.5)) or (val_score <= (mean_score - 1.5)):
                dict_ann_pairs[val[WORD_ANN]].append(k_wordpair)

    return dict_ann_pairs


# Finds annotator-specific APIAA scores and filters out as needed those with the lowest inter-annotator agreement
def filter_apiaa_annotators(dict_annotators):
    # Some documentation on the algorithms and data structures herein. This extends also to the subsequent find_apiaa()

    # remaining_annotators is a dict of those annotators - and their scores - still under consideration. It original
    # form is a (deep) copy of dict_annotators. Each time we eliminate an annotator entry due to APIAA scores, its
    # corresponding key entry is removed from this dict, and subsequent APIAA matrices are built from what remains.
    #
    # score_matrix is a list of all word pair scores. Each line is all scores from one annotator, with the
    # count order of the lines corresponding to the pair order in the original list of 1,888 pairs.
    #
    # 0 [ 1, 4, 6, 3, 2, 4, 5, ..... 1 ] (Annotator 0, all 1,888 scores)
    # 1 [ 2, 5, 3, 5, 2, 6, 1, ..... 3 ] (Annotator 1, all 1,888 scores)
    # ... (lines for all other annotators)
    #
    # spear_matrix is a 2d list of all Spearman's correlations between annotators.
    # Each line of Spearman's correlation numbers is calculated based on two lines from score_matrix::
    #      annotators 0 and 1, 0 and 2, 0 and ... n
    #                 1 and 2, 1 and 3, 1 and ... n
    #                 2 and 3, 2 and 4, 2 and ... n
    #                 ....
    #                 n-1 and n
    #
    # The results fill in the UPPER triangle of spear_matrix:
    # (Note: We start counting from 0, and use n for convenience, so the actual number of annotators is n+1)
    # (annotator) 0   [ ----, s0:1, s0:2, s0:3, s0:4, ..., s0:n ]
    # (annotator) 1   [ ----, ----, s1:2, s1:3, s1:4, ..., s1:n ]
    # (annotator) 2   [ ----, ----, ----, s2:3, s2:4, ..., s2:n ]
    #    ...               ...
    # (annotator) n-1 [ ----, ----, ----, ...., ----, ----, s(n-1):n ]
    # (annotator) n   [ ----, ----, ----, ...., ----, ----, --- ]
    # Note that the matrix's last line (representing annotator n) is empty, since there are no scores left to compute.
    # (Note also that non-used cells will contain zero, since random placeholder values would break our calculations.)
    #
    # As per the article, before we start calculating the overall APIAA, we first need to eliminate those annotators
    # who exhibit the least average agreement with other annotators. If we examine spear_matrix, we see that for
    # annotator i, the total sum of Spearman's correlations with other annotators is given by two sums: all values in
    # column i (COLi), plus all values in row i (ROWi). In terms of spear_matrix, that would be these numbers:
    #
    #               COLi
    # [ ...., ...., s0-2, ...., ...., ..., .... ]
    # [ ...., ...., s1-2, ...., ...., ..., .... ]
    # [ ----, ----, ----, s2-3, s2-4, ..., s2-n ] ROWi
    #
    # What we now want to find is which annotator has the lowest average agreement with all the other annotators.
    # For annotator i, and a total number of N annotators, this average agreement is equivalent to:
    # (COLi + ROWi) / (N-1)
    #
    # We add these n averages to a new list, ann_spear_avg. Recall that a higher number in slot i means that the
    # scores from annotator i are in strong agreement with those of other annotators. The list then looks like this:
    #
    # [ 2.4, 5.6, 3.3, 1.3, 4.3, ..., n ]
    #
    # If there are more than 10 annotators in the list, we now iterate over it and look for the lowest number.
    # If this number is higher than the one we found in the *last* iteration, we drop it from the list, and remove
    # the corresponding annotator from our list of candidates. (Note that by default, the very first iteration will
    # always drop the lowest-scoring annotator, so long as more than 10 remain. This is part of the methodology
    # in the original article, so we'll follow suit here.) We then recreate the two matrices and recalculate.
    #
    # Eventually we have a final list of the annotators whose values we want to include. That list is our return value.

    score_matrix = []
    remaining_annotators = copy.deepcopy(dict_annotators)
    ann_spear_avg = []
    # While new_round is True, we need to recalculate spear_matrix and check whether we should drop another annotator.
    if len(remaining_annotators) > 10:
        new_round = True
    else:
        new_round = False
    lowest_avg_score = -1.0

    while new_round:
        remaining_annotators_reference = list(remaining_annotators.keys())
        remaining_annotator_count = len(remaining_annotators)

        # Create score_matrix from remaining_annotators
        score_matrix.clear()
        for r in remaining_annotators_reference:
            score_matrix.append(list(tup[ANN_SCORE] for tup in remaining_annotators[r]))

        # Create spear_matrix from score_matrix (i.e. calculate and save all Spearman's correlation values)
        spear_matrix = np.zeros(shape=(remaining_annotator_count, remaining_annotator_count), dtype=float)
        for i in range(0, remaining_annotator_count):
            for j in range(i + 1, remaining_annotator_count):
                # Calculate Spearman's correlation between the two.
                # Note that spearmnr() returns a tuple. We want the first value, hence the [0]
                spear_matrix[i][j] = stats.spearmanr(score_matrix[i], score_matrix[j])[0]
        column_sums = spear_matrix.sum(axis=0)
        row_sums = spear_matrix.sum(axis=1)

        ann_spear_avg.clear()
        # Populate ann_spear_avg from spear_matrix
        for i in range(0, remaining_annotator_count):
            avg_i = (column_sums[i] + row_sums[i]) / (remaining_annotator_count - 1)
            ann_spear_avg.append(avg_i)

        # Find the lowest value (and its position) in ann_spear_avg:
        curr_low_idx, curr_low_val = min(enumerate(ann_spear_avg), key=operator.itemgetter(1))

        if (curr_low_val > lowest_avg_score) and (remaining_annotator_count > 10):
            annotator_key = remaining_annotators_reference[curr_low_idx]
            remaining_annotators.pop(annotator_key)
            new_round = True
        else:
            new_round = False

    return remaining_annotators


# Find overall APIAA, ordered by part-of-speech, using a pre-APIAA-filtered list of annotators and annotations.
def find_apiaa(filtered_ann_dict):
    # The calculations of score_matrix and spear_matrix are the same as in filter_apiaa_annotators().
    # Since we need the list of remaining_annotators for other functions, we split the actual final APIAA calculations
    # off from the filter_apiaa_annotators() function and implement it here instead.
    # We then recreate score_matrix and spear_matrix as before, using only the annotators left after
    # filter_apiaa_annotators(). This time around, we also create variants of these matrices based on scores for
    # specific grammatical categories.
    # For all the variants, we calculate the overall APIAA. If SUMALL is the sum of every number in spear_matrix,
    # and N is the number of annotators represented by that matrix, then APIAA = (2 * SUMALL) / [N * (N - 1) ]
    pos_categories = ['no', 'lo', 'so', 'ao', 'allt']
    annotator_count = len(filtered_ann_dict)

    # apiaa_pos_tuples is our return value: A list of tuples, where each tuple contains a part-of-speech category
    # designation (from pos_categories) and the corresponding APIAA from filtered_ann_dict.
    apiaa_pos_tuples = []

    for pos_cat in pos_categories:
        cat_score_matrix = []
        for k_r, v_r in filtered_ann_dict.items():
            list_cat_scores = []
            for v_tup in v_r:
                if (pos_cat is 'allt') or ((v_tup[ANN_CAT] == pos_cat) and (pos_cat is not 'allt')):
                    list_cat_scores.append(v_tup[ANN_SCORE])
            cat_score_matrix.append(list_cat_scores)

        cat_spear_matrix = np.zeros(shape=(annotator_count, annotator_count), dtype=float)
        for i in range(0, annotator_count):
            for j in range(i + 1, annotator_count):
                cat_spear_matrix[i][j] = stats.spearmanr(cat_score_matrix[i], cat_score_matrix[j])[0]
        cat_spear_sum = cat_spear_matrix.sum()
        apiaa_val = (2 * cat_spear_sum) / (annotator_count * (annotator_count - 1))
        apiaa_tuple = (pos_cat, apiaa_val)
        apiaa_pos_tuples.append(apiaa_tuple)

    return apiaa_pos_tuples


def find_amiaa(filtered_apiaa_dict):
    # This time around, we have two variants of the score_matrix from find_apiaa:
    #
    # score_matrix_original is identical to the original score_matrix. Line i contains all pair scores from annotator i:
    # 0 [ 1, 4, 6, 3, 2, 4, 5, ..... 1 ] (Annotator 0, all scores for that particular word category (nouns, verbs, etc.)
    # 1 [ 2, 5, 3, 5, 2, 6, 1, ..... 3 ] (Annotator 0, all scores for that particular word category (nouns, verbs, etc.)
    # ... (lines for all other annotators)
    #
    # score_matrix_others has the same structure as score_matrix_original, but now line i contains the average of all
    # annotator scores for that particular pair except for the score from annotator i:
    # 0 [ 3.1, 5.5, 2.6, 5.3, 2.6, 4.3, 5.5, ..... 1.2 ] (Average category scores of everyone except annotator 0)
    # 1 [ 2.5, 5.4, 6.0, 5.7, 2.2, 4.3, 1.2, ..... 3.8 ] (Average category scores of everyone except annotator 0)
    # ... (lines for all other annotators)
    #
    # spear_matrix is highly simplified this time around. We'll call the Spearman's correlation p(Si,Ui) where Si
    # is annotator i's scores (that is, row i in score_matrix_original) while Ui is the average scores of everyone
    # except annotator i (row i in score_matrix_others).
    #
    # Our initial matrix of all correlations then becomes:
    # [ s0u0, s1u1, s2u2, ..., sNuN ]
    #
    # To find the overall AMIAA score, we simply add up all the entries in that list and divide the result by the
    # total number of annotators, N. We do this for each part-of-speech category, and also for the entire collection
    # of scores.
    pos_categories = ['no', 'lo', 'so', 'ao', 'allt']
    annotator_count = len(filtered_apiaa_dict)

    # amiaa_pos_tuples is our return value: A list of tuples, where each tuple contains a part-of-speech category
    # designation (from pos_categories) and the corresponding AMIAA from filtered_apiaa_dict.
    amiaa_pos_tuples = []

    for pos_cat in pos_categories:
        score_matrix_original_array = []
        for k_r, v_r in filtered_apiaa_dict.items():
            list_cat_scores = []
            for v_tup in v_r:
                if (pos_cat is 'allt') or ((v_tup[ANN_CAT] == pos_cat) and (pos_cat is not 'allt')):
                    list_cat_scores.append(v_tup[ANN_SCORE])
            score_matrix_original_array.append(list_cat_scores)

        score_matrix_original = np.asarray(score_matrix_original_array)
        score_matrix_original_sums = score_matrix_original.sum(axis=0)
        score_matrix_others_array = []

        for ann_i in range(0, annotator_count):
            row_i = []
            category_size = len(score_matrix_original_sums)
            for cell in range(0, category_size):
                average_other_annotators = (score_matrix_original_sums[cell] - score_matrix_original[ann_i][cell]) / (
                        annotator_count - 1)
                row_i.append(average_other_annotators)
            score_matrix_others_array.append(row_i)

        score_matrix_others = np.asarray(score_matrix_others_array)
        sum_spearman = 0
        for i in range(0, annotator_count):
            sum_spearman += stats.spearmanr(score_matrix_original[i], score_matrix_others[i])[0]
        amiaa_val = sum_spearman / annotator_count
        amiaa_tuple = (pos_cat, amiaa_val)
        amiaa_pos_tuples.append(amiaa_tuple)

    return amiaa_pos_tuples


# Returns the percentage distribution of concept pairs over one-step (1) intervals
def find_interval_distribution(filtered_apiaa_dict):
    return_intervals = []
    interval_sums = [0] * 6

    for val_list in filtered_apiaa_dict.values():
        for val_tup in val_list:
            if val_tup[ANN_SCORE] == 6:
                interval_sums[5] += 1
            else:
                interval_sums[val_tup[ANN_SCORE]] += 1
        total_count = sum(interval_sums)

    for lower_bound in range(0, 6):
        if lower_bound == 5:
            interval_text = "[5, 6]"
        else:
            interval_text = "[" + str(lower_bound) + ", " + str(lower_bound + 1) + ")"
        bound_distribution = interval_sums[lower_bound] / total_count
        return_intervals.append((interval_text, bound_distribution))

    return return_intervals


def create_score_list(filtered_dict):
    all_scores = []
    for ann_data in filtered_dict.values():
        for ann_tup in ann_data:
            all_scores.append(ann_tup[ANN_SCORE])
    return all_scores


def find_averages(filtered_dict):
    # Gera dict, og nota talningu á dict_annotators til að fá meðaltalið?
    dict_pairs = OrderedDict()
    ann_count = len(filtered_dict)
    for tuple_index, tuple_list in enumerate(filtered_dict.values()):
        if tuple_index == 0:
            for word_tuple in tuple_list:
                dict_pairs[word_tuple[ANN_PAIR]] = 0
        else:
            for word_tuple in tuple_list:
                dict_pairs[word_tuple[ANN_PAIR]] += word_tuple[ANN_SCORE]
    for score_key, score_value in dict_pairs.items():
        dict_pairs[score_key] = score_value / ann_count
    return dict_pairs


def create_msl_dataframe(filtered_dict):
    #Recreate filtered_dict as a 2-dimensional array (list of lists)
    #Then create that as a dataframe
    # dict_annotators has key: annotator, and value: tuple of (word pair tuple, pair score, pair category)
    score_array =[]
    for ann_data in filtered_dict.values():
        ann_list = []
        for ann_tup in ann_data:
            ann_list.append(ann_tup[ANN_SCORE])
        score_array.append(ann_list)
    #Score array has word pairs for x-axis and annotators for the y-axis, so we need to transpose it
    #in order to run the subsequent PCA.
    np_ndmsl = np.transpose(score_array)
    df = pd.DataFrame(np_ndmsl)
    return df


### START OF PROGRAM

# Load data from file into dict
df_ann = load_data()

# Convert data from DataFrame to OrderedDict
# dict_annotators has key: annotator, and value: tuple of (word pair tuple, pair score, pair category)
dict_of_annotators = load_annotators_from_df(df_ann)

list_of_annotators = list(dict_of_annotators.keys())
# dict_of_wordpairs has key: word pair tuple, and value: tuple of (annotator, pair score, pair category)
dict_of_wordpairs = load_wordpairs_from_df(df_ann)

dict_pilehvar = find_pilehvar_outliers(dict_of_wordpairs, list_of_annotators)

#print("Pilehvar")
#print(dict_pilehvar)

dict_remaining_ann = filter_apiaa_annotators(dict_of_annotators)

#print("Remaining annotators")
#print(dict_remaining_ann.keys())

apiaa_by_pos = find_apiaa(dict_remaining_ann)

#print("APIAA")
#print(apiaa_by_pos)

amiaa_by_pos = find_amiaa(dict_remaining_ann)

#print("AMIAA")
#print(amiaa_by_pos)

dist_by_interval = find_interval_distribution(dict_remaining_ann)

list_of_all_scores = create_score_list(dict_remaining_ann)

mean_score = mean(list_of_all_scores)
median_score = median(list_of_all_scores)
stdev_score = stdev(list_of_all_scores)

pair_averages_dict = find_averages(dict_remaining_ann)
#print(pair_averages_dict)

#print(mean_score)
#print(median_score)
#print(stdev_score)

#NOTE: The following code is only for data presentation and for embedding pre-processing.
# It is not part of the MultiSimLex calcuations, and does not have to be run in order
# to complete or verify them.
with open('var_scores.txt', 'w', encoding='utf-8') as f_varscores:
    f_varscores.write("POS-sorted agreement with independent translator:\n")
    f_varscores.write("Adjectives: 50.0%\n")
    f_varscores.write("Adverbs: 50.0%\n")
    f_varscores.write("Nouns: 83.9% (full raw: 83.92857143%)\n")
    f_varscores.write("Verbs: 70.8% (full raw: 70.833333333%)\n")
    f_varscores.write("Overall: 74.0%\n")

    f_varscores.write("\nSum totals of Pilehvar word pairs per annotator, BEFORE APIAA filtering:\n")
    list_pilehvar_ann = list(dict_pilehvar.keys())
    for a in list_pilehvar_ann:
        pil_per_ann_str = str(a) + ": " + str(len(dict_pilehvar[a])) + "\n"
        f_varscores.write(pil_per_ann_str)

    f_varscores.write("\nFinal filtered list of annotators:\n")
    str_filtered_annotators = ", ".join(list(dict_remaining_ann.keys()))
    f_varscores.write(str_filtered_annotators)

    f_varscores.write("\n\nAPIAA scores, sorted by POS:\n")
    for apiaa_tup in apiaa_by_pos:
        apiaa_str = str(apiaa_tup[0]) + ": " + str(apiaa_tup[1]) + "\n"
        f_varscores.write(apiaa_str)

    f_varscores.write("\nAMIAA scores, sorted by POS:\n")
    for amiaa_tup in amiaa_by_pos:
        amiaa_str = str(amiaa_tup[0]) + ": " + str(amiaa_tup[1]) + "\n"
        f_varscores.write(amiaa_str)

    f_varscores.write("\nPercentage distribution of concept pairs over one-step (1) intervals:\n")
    for d_tup in dist_by_interval:
        dist_str = str(d_tup[0]) + ": " + str(d_tup[1]*100) + "\n"
        f_varscores.write(dist_str)

    f_varscores.write("\nOverall values:\n")
    mean_str = "Mean: " + str(mean_score) + "\n"
    f_varscores.write(mean_str)
    median_str = "Median: " + str(median_score) + "\n"
    f_varscores.write(median_str)
    stdev_str = "Standard deviation (spread): " + str(stdev_score) + "\n"
    f_varscores.write(stdev_str)

with open('training_data_org.txt', 'w', encoding='utf-8') as f_train, \
        open('training_data_scaled.txt', 'w', encoding='utf-8') as f_scaled:
    for pair_avg_tup in pair_averages_dict.items():
        wordpair_str = str(pair_avg_tup[0][0]) + "\t" + str(pair_avg_tup[0][1]) + "\t" + str(pair_avg_tup[1]) + "\n"
        f_train.write(wordpair_str)
        # We're scaling a range of [0,6] to [0.0,1.0], so we simply divide by six.
        wordpair_scl_str = str(pair_avg_tup[0][0]) + "\t" + str(pair_avg_tup[0][1]) + "\t" + str(pair_avg_tup[1]/6) + "\n"
        f_scaled.write(wordpair_scl_str)

