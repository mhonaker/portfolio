"""
Sequence alignment project to demonstrate dynamic programming.
Most function names and return structures were required by the
machine grader, but all other work is my own unless otherwise
noted. Written for python 2.7.
"""

import urllib2
import random
import string
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

def read_scoring_matrix(filename):
    """
    Reads in a scoring matrix from a file located in cloud storage
    or locally. Outputs a scoring matrix as a dictionary of dictionaries.
    """

    scoring_dict = {}
    # uncomment line below if gathering resources from web
    #scoring_file = urllib2.urlopen(filename) 
    scoring_file = open(filename)
    ykeys = scoring_file.readline()
    ykeychars = ykeys.split()
    
    for line in scoring_file.readlines():
        vals = line.split()
        xkey = vals.pop(0)
        scoring_dict[xkey] = {}
        for ykey, val in zip(ykeychars, vals):
            scoring_dict[xkey][ykey] = int(val)
    scoring_file.close()
    return scoring_dict

def read_protein(filename):
    """
    Takes a file and reads in a protein sequence as
    single letter AA codes Returns a string of the sequence.
    """

   # if reading from the web, uncomment the line below   
   #protein_file = urllib2.urlopen(filename)
    
    protein_file = open(filename)
    protein_seq = protein_file.read()
    protein_seq = protein_seq.rstrip()
    protein_file.close()
    return protein_seq

def read_words(filename):
    """
    Takes a file containing a word list 
    and returns a list of strings.
    """

    word_file = urllib2.urlopen(filename)
    words = word_file.read()
    word_list = words.split('\n')
    print "loaded words with", len(word_list), "words"
    return word_list

def build_scoring_matrix(alphabet, match_score, unmatch_score, gap_penalty):
    """
    Build a 'scoring matrix' which is modeled as a 
    dictionary of dictionaries.
    Inputs: an alphabet, scores that indicate a match, 
    an unmatched pair, and a gap penalty.
    """

    score_mat = {}
    for letter in alphabet:
        score_mat[letter] = {}
    for xval in score_mat:
        for yval in alphabet:
            if xval == yval and xval != '-': 
                score_mat[xval][yval] = match_score
            elif xval == '-' or yval == '-':
                score_mat[xval][yval] = gap_penalty
            else:
                score_mat[xval][yval] = unmatch_score
    return score_mat

def compute_alignment_matrix(seq_x, seq_y, scoring_matrix, global_flag):
    """

    Computes the scoring matrix for an alignment.
    Takes two sequences, a scoring matrix, and a global
    aligment flag. Computes the global alignment if True,
    and local if False. When computing the local alignment,
    all negative scores are replaced with 0
    """

    align_mat = [[0 for idx in range(len(seq_y)+1)]
                 for idx in range(len(seq_x)+1)]
    
    for idx in range(1, len(seq_x) + 1):
        align_mat[idx][0] = align_mat[idx - 1][0] +
        scoring_matrix[seq_x[idx - 1]]['-']

        if not global_flag and align_mat[idx][0] < 0:
            align_mat[idx][0] = 0
    
    for idx in range(1, len(seq_y) + 1):
        align_mat[0][idx] = align_mat[0][idx - 1] +
        scoring_matrix['-'][seq_y[idx - 1]]

        if not global_flag and align_mat[0][idx] < 0:
            align_mat[0][idx] = 0

    for idx_i in range(1, len(seq_x) + 1):
        for idx_j in range(1, len(seq_y) + 1):
            align_mat[idx_i][idx_j] = max(align_mat[idx_i-1][idx_j-1] +
                                          scoring_matrix[seq_x[idx_i-1]][seq_y[idx_j-1]],
                                          align_mat[idx_i-1][idx_j] +
                                          scoring_matrix[seq_x[idx_i-1]]['-'],
                                          align_mat[idx_i][idx_j-1] +
                                          scoring_matrix['-'][seq_y[idx_j-1]])
            if not global_flag and align_mat[idx_i][idx_j] < 0:
                align_mat[idx_i][idx_j] = 0
    return align_mat

def compute_global_alignment(seq_x, seq_y, scoring_matrix, alignment_matrix):
    """
    Takes two sequences, a scoring matrix (dict of dicts),
    an alignment matrix (list of lists). 
    Output is a tuple with the score and alignment.
    """

    idx_i, idx_j = len(seq_x), len(seq_y)
    score = alignment_matrix[idx_i][idx_j]
    align_x, align_y = '', ''

    while idx_i > 0 or idx_j > 0:
        if alignment_matrix[idx_i][idx_j] == alignment_matrix[idx_i-1][idx_j-1] + \
                                 scoring_matrix[seq_x[idx_i-1]][seq_y[idx_j-1]]:
            align_x = seq_x[idx_i-1] + align_x
            align_y = seq_y[idx_j-1] + align_y
            idx_i -= 1
            idx_j -= 1
        else:
            if alignment_matrix[idx_i][idx_j] == alignment_matrix[idx_i-1][idx_j] + \
                                              scoring_matrix[seq_x[idx_i-1]]['-']:
                align_x = seq_x[idx_i-1] + align_x
                align_y = '-' + align_y
                idx_i -= 1
            else:
                align_x = '-' + align_x
                align_y = seq_y[idx_j-1] + align_y
                idx_j -= 1

    return (score, align_x, align_y)


def compute_local_alignment(seq_x, seq_y, scoring_matrix, alignment_matrix):
    """
    Same as compute global alignmenet, but for a local alignment.
    """

    max_score = -1000
    
    for idx_i in range(len(seq_x)+1):
        for idx_j in range(len(seq_y)+1):
            if alignment_matrix[idx_i][idx_j] >= max_score:
                max_score_position = [idx_i, idx_j]
                max_score = alignment_matrix[idx_i][idx_j]
                
        
    idx_i, idx_j = max_score_position[0], max_score_position[1]
    score = alignment_matrix[idx_i][idx_j]
    align_x, align_y = '', ''

    while alignment_matrix[idx_i][idx_j] != 0: 
        if alignment_matrix[idx_i][idx_j] == alignment_matrix[idx_i-1][idx_j-1] + \
                                 scoring_matrix[seq_x[idx_i-1]][seq_y[idx_j-1]]:
            align_x = seq_x[idx_i-1] + align_x
            align_y = seq_y[idx_j-1] + align_y
            idx_i -= 1
            idx_j -= 1
        else:
            if alignment_matrix[idx_i][idx_j] == alignment_matrix[idx_i-1][idx_j] + \
                                              scoring_matrix[seq_x[idx_i-1]]['-']:
                align_x = seq_x[idx_i-1] + align_x
                align_y = '-' + align_y
                idx_i -= 1
            else:
                align_x = '-' + align_x
                align_y = seq_y[idx_j-1] + align_y
                idx_j -= 1

    return (score, align_x, align_y)

def generate_null_distribution(seq_x, seq_y, scoring_matrix, num_trials):
    """
    Takes in two sequences and a scoring matrix, and run num_trials.
    Returns a dictionary the represents an un-normalized distribution from
    1.generating a random permutation of seq_y 
    2.computing local alignment score of seq_x and random_seq_y 
    3. adding this to that score in the dictionary each time it is the same
    """

    score_dict = {}
    yvals = list(seq_y)
    for trial in range(num_trials):
        random.shuffle(yvals)
        rand_y = ''.join(yvals)
        align_mat = compute_alignment_matrix(
            seq_x, rand_y, scoring_matrix, False)
        score = compute_local_alignment(
            seq_x, rand_y, scoring_matrix, align_mat)[0]
        if score in score_dict:
            score_dict[score] += 1
        else:
            score_dict[score] = 1
    return score_dict

def create_normalized_data(dist, num_trials):
    """
    Takes a dictionary of scores and number of occurences of those scores
    and outputs a list of tuples where the first list has the scores
    and the second, the normalized fraction of that score, in the same order.
    """

    score_list = []
    fraction_list = []
    for key in dist:
        score_list.append(key)
        fraction_list.append(dist[key] / float(num_trials))
    plot_list = []
    plot_list.append(score_list)
    plot_list.append(fraction_list)

    return plot_list

def bar_plot(plot_data, mu, sd):
    """
    Create a bar plot (histogram).
    """

    scores = plot_data[0]
    frac = plot_data[1]
    plt.bar(scores, frac, align = 'center', color = 'teal', alpha = 0.5)
    plt.xlabel('Local Alignment Score')
    plt.ylabel('Number of Observations (normalized to fraction of total)')
    plt.title('Normalized Distibution of Random Local Alignment Scores')
    plt.savefig('align_dist.png')
    plt.show()

def check_spelling(word, distance, word_list, scoring_matrix):
    """
    Takes a word and returns a list of all words from a given list
    which are within the given edit distance 
    """

    result = []
    for item in word_list:
        align_mat = compute_alignment_matrix(word, item, scoring_matrix, True)
        score = len(word) + len(item) - compute_global_alignment(word, item, scoring_matrix, align_mat)[0]
        if score <= distance:
            result.append(item)
    return result       

#--------------------------------------------------------------------
#
# Now do some actual alignments for the project.
#
#--------------------------------------------------------------------

pam_score_mat = read_scoring_matrix('PAM50.txt')
human_eyeless = read_protein('HumanEyelessProtein.txt')
fly_eyeless = read_protein('FruitflyEyelessProtein.txt')

hf_loc_align_mat = compute_alignment_matrix(human_eyeless,
                                            fly_eyeless,
                                            pam_score_mat, False)
hf_local = compute_local_alignment(human_eyeless,
                                   fly_eyeless,
                                   pam_score_mat, hf_local_align_mat)  


print "human - fly local alignment", hf_local
print "length human", len(h_f_local[1])
print "length fly", len(h_f_local[2])


con_pax_dom = read_protein('ConsensusPAXDomain.txt')
human_pax_global_align_mat = compute_alignment_matrix(human_local,
                                                      con_pax_dom,
                                                      pam_score_mat, True)
fly_pax_global_align_mat = compute_alignment_matrix(fly_local,
                                                    con_pax_dom,
                                                    pam_score_mat, True)

print 'human-pax global alignment: ', compute_global_alignment(human_local, 
                                                               con_pax_dom, 
                                                               pam_score_mat, 
                                                               human_pax_global_align_mat)
print 'fly-pax global alignment: ', compute_global_alignment(fly_local,
                                                             con_pax_dom,
                                                             pam_score_mat,
                                                             fly_pax_global_align_mat)

null_dist = generate_null_distribution(human_eyeless, fly_eyeless, pam_score_mat, 1000)
normal_data = create_normalized_data(null_dist, 1000)

bar_plot(normal_data, mean, std)

spell_score_mat = build_scoring_matrix(string.lowercase + '-', 2, 1, 0)
word_list = read_words(WORD_LIST_URL)[:-1]

print 'words within 1 edit distance from humble: ', \
        check_spelling('humble', 1, word_list , spell_score_mat)
print 'words within 2 edit distance from firefly: ', \
        check_spelling('firefly', 2, word_list, spell_score_mat)
