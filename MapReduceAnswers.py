"""
The functions here are part of course projects. The return
structures of the functions are as required by the machine grader.
In this case, they also depend on havinf MapReduce.py in the same
directory, and variable mr as defined here. Written for python 2.7.
For the orginal assignment, each pair of MapReduce function were in
a separate file, called at the command line by the grader. I have
placed them all here in one file, but DO NOT expect them to work
as-is.
"""

import MapReduce
import sys

mr = MapReduce.MapReduce()


# Problem 1
#--------------------------------------------------------------------
#
# Create an inverted index from a set of documents.
# Mapper input is a list [doc_id, text]
# Reducer output is tuple (word, doc_id)
# data for testing is books.json (in) and inverted_index.json (out)
#
#--------------------------------------------------------------------

def mapper(record):
    # key: document identifier
    # value: document contents
    key = record[0]
    value = record[1]
    words = value.split()
    for w in words:
      mr.emit_intermediate(w,key)

def reducer(key, list_of_values):
    # key: word
    # value: list of occurrence counts
    result = []
    for v in list_of_values:
        if v not in result:
            result.append(v)
    mr.emit((key, result))


# Problem 2
#--------------------------------------------------------------------
#
# Implement a SQL-like join as a MapReduce query
# Input is a list of strings representing a tuple in a database, 
# where each list element cooresponds to a table attribute. The
# first item in each record is one of 'line_item' or 'order', the
# second element is the order_id. line_items have 17 attributes, and
# orders have 10 attributes.
# The output should be a joined record consisting of a single list
# containing attributes of the order, then the line_item.
# data for testing is records.json (in) and join.json (out)
#
#--------------------------------------------------------------------

def mapper(record):
    # key: order id
    # value: the rest of the attributes
    key = record[1]
    value = record
    mr.emit_intermediate(key, value)

def reducer(key, list_of_values):
    # key: order id
    # value: rest of the attributes
    value = []
    #for j in key:
    for i in range(1,len(list_of_values)):
        value = list_of_values[0] + list_of_values[i]
        mr.emit(value)


# Problem 3
#--------------------------------------------------------------------
#
# Counting friends in a social network
# Inputs are 2 element lists [personA, personB] where person B is a
# friend of person A. The reverse may or may not be true.
# Outputs should be pairs [person, friend_count]
# data for testing is friends.json (in) and friends_count.json (out)
# 
#--------------------------------------------------------------------

def mapper(record):
    # key: person A name
    # value: friend's name
    key = record[0]
    value = record[1]
    mr.emit_intermediate(key, 1)

def reducer(key, list_of_values):
    # key: word
    # value: list of occurrence counts
    total = 0
    for v in list_of_values:
      total += v
    mr.emit((key, total))


# Problem 4
#--------------------------------------------------------------------
#
# Counting asymmetric friends relations in a social network
# Inputs are 2 element lists [personA, personB] where person B is a
# friend of person A. The reverse may or may not be true, as above.
# Outputs should be pairs of (person, friend) and (friend, person),
# without duplicates!
# data for testing is friends.json (in) and asymmetric.json (out)
# 
#--------------------------------------------------------------------

def mapper(record):
    # key: person
    # value: friend
    key = record[0]
    value = record[1]
    mr.emit_intermediate(key, value)
    key2 = record[1]
    value2 = record[0]
    mr.emit_intermediate(key2, value2)

def reducer(key, list_of_values):
    # key: person 
    # value: friends
    result = [(key, friend) for friend in list_of_values]
    for j in result:
        if result.count(j) < 2:
            mr.emit(j)



# Problem 5
#--------------------------------------------------------------------
#
# Inputs records are 2 element lists [seq_id, nucleotide]
# where seq_id is a unique id, and nucleotide is a string
# of 1 letter base codes.
# Outputs are the seq_id and nucleotide sequences trimmed by 10
# bases, and duplicats removed.
# data for testing is dna.json (in) and unique_trims.json (out)
# 
#--------------------------------------------------------------------

def mapper(record):
    # key: sequence id
    # value: sequence
    key = record[0]
    value = record[1]
    #swiching the order of the key and value
    #this automatically makes the sequences unique
    #also trimming in this step
    mr.emit_intermediate(value[:-10],key)

def reducer(key, list_of_values):
    # key: sequence (unique)
    # value: sequence id
    mr.emit(key) 


# Problem 6
#--------------------------------------------------------------------
#
# Design a MapReduce algorithm to compute the matrix multiplication
# of A X B
# Inputs are rows of each matrix as lists [matrix, i,j]
# where matrix is A or B, i and j are integers.
# Output should also be matrix rows, in tuples, (i,j, value)
# data for testing is matrix.json (in) and multiply.json (out)
# 
#--------------------------------------------------------------------

def mapper(record):
    # key: document identifier
    # value: document contents
    if record[0] == "a":
        i = record[1]
        value = record[3]
        for k in range(5):
            mr.emit_intermediate((i, k), (record[2],value))
    else:
        m = record[2]
        value = record[3]
        for j in range(5):
            mr.emit_intermediate((j,m), (record[1], value))

def reducer(key, list_of_values):
    # key: word
    # value: list of occurrence counts
    result = []
    a = sorted(list_of_values)
    for i in range(len(a)-1):
        if a[i][0] == a[i+1][0]:
            result.append(a[i][1] * a[i+1][1])
    #answer = key +(sum(result))
    mr.emit((key[0], key[1], sum(result)))

#====================================================================
#
# After copying each function the header material (imports and mr)
# and the footer just below each MapReduce job can be run at the
# command line with the appropriate arguements.

if __name__ == '__main__':
    inputdata = open(sys.argv[1])
    mr.execute(inputdata, mapper, reducer)


