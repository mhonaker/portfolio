"""
A very simplistic and brief set of analyses of tweets for a class.
The tweets come from a couple of minutes worth of tweets
scraped from the 1% live feed using the Twitter API.
For the machine grader, everything was orginally set up 
to print line by line to stdout, but has since been
refactored slightly to return the values as lists or dicts.
"""

import json
import re

def freq(tweet_file):
    """
    Returns the normalized frequency of terms in an opened
    or streaming twitter file.
    """
    terms = []
    term_dict = {}    
    for line in tweet_file: 
        tweet = json.loads(line)
        if "text" in tweet:
            tweet_text = tweet["text"].lower()    
            words = re.split(r'[_\W]', tweet_text)
            terms += [word for word in words if word != ""]
    total_terms = float(len(terms))
    for term in terms:
        if term in term_dict:
            term_dict[term] = term_dict[term] + 1
        else:
            term_dict[term] = 1
    return {key: value / total_terms for key, value in term_dict.items()}

def sentiment(sent_file):
    """
    Extracts the phrases from a term sentiment document 
    for one dict and the words for dict then returns them.
    """

    sentiment_phrases = {}
    sentiment_words = {}
    
    with open(sent_file) as f:
        for line in f:
            term, score = line.split('\t')
            if len(term.split()) > 1 or "-" in term:
                sentiment_phrases[term] = int(score)
            else:
                sentiment_words[term] = int(score)
    return [sentiment_phrases, sentiment_words]

def sent_scores(phrase_dict, word_dict, twt_file):
    """
    Computes a sentiment score of each tweet based on the 
    sentiment file and returns the total score of each tweet.
    Really only works with english currently.
    """
    scores = []
    for line in twt_file:
        score = 0
        tweet = json.loads(line)
        if "text" in tweet:
            tweet_text = tweet["text"].lower()
            for key in phrase_dict:
                replace = re.subn(key, "", tweet_text)
                tweet_text = replace[0]
                score += phrase_dict[key] * replace[1]
            for word in re.split(r'[_\W]', tweet_text):
                key = word
                if key in word_dict:
                    score += word_dict[key]
        scores.append(score)
    return scores

def tag_count(tweet_file):
    """
    Returns the 10 most common hashtags from a file
    of tweets, followed by count.
    """
    tags = []
    tag_dict = {}
    for line in tweet_file: 
        tweet = json.loads(line)
        if "text" in tweet:
            for item in tweet["entities"]["hashtags"]:
                if item["text"]:
                    tags.append(item["text"])
    for tag in tags:
        if tag in tag_dict:
            tag_dict[tag] += 1
        else:
            tag_dict[tag] = 1
    return [(tag, tag_dict[tag]) for tag in sorted(
        tag_dict, key=tag_dict.get, reverse=True)][:11]

states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

reverse_states = {y:x for x,y in states.items()}
    
def happiest(phrase_dict, word_dict, t_file):
    """
    Computes the average sentiment score of each tweet
    with a location in the places field of the tweet
    Returns the state abbreviation with the highest
    average sentiment score.
    """
    
    avg_scores = []
    states = []
    for line in t_file:
        score = 0
        count = 0
        tweet = json.loads(line)
        if "text" in tweet and tweet["place"]:
            tweet_text = tweet["text"].lower()
            for key in phrase_dict:
                replace = re.subn(key, "", tweet_text)
                tweet_text = replace[0]
                score += phrase_dict[key] * replace[1]
                count += 1
            words = re.split(r'[_\W]', tweet_text)
            for word in words:
                key = word
                if key in word_dict:
                    score += word_dict[key]
                    count += 1
            if score == 0:
                avg_score = 0.0
            else:
                avg_score = score / float(count)
            for key in reverse_states:
                if key in tweet["place"]["full_name"]:
                    state = reverse_states[key]
                else:
                    if reverse_states[key] in tweet["place"]["full_name"]:
                            state = reverse_states[key]
            if state not in states:
                states.append(state)
                avg_scores.append(avg_score)
            else:
                avg_scores[states.index(state)] = (
                    avg_scores[states.index(state)] + avg_score) / 2.0
    return states[avg_scores.index(max(avg_scores))]



# make the sentiment dictionaries
sentiments = sentiment('AFINN-111.txt')
phrases = sentiments[0]
words = sentiments[1]


