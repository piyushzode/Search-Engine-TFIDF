################################
####   Author: Piyush Zode  ####
####   ID: XXXXXXXXXX       ####
################################
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
from math import log10,sqrt
import operator # For sorting

# Get the Files from the presidential_debates Folder
corpusroot = './presidential_debates'
new_tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
my_stopwords = sorted(stopwords.words('english'))   # Get the English StopWords
stemmer = PorterStemmer()

# Clean the files(Stop word Removal, Stemming)
def clean_files(filename):
    words_in_doc = []
    tmp_doc_list = {}
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')  # Open the file in readmode
    doc = file.read()
    file.close() 
    doc = doc.lower()   # Convert to lower case to make the compare the words in future
    
    # Tokenize the document
    tokens = new_tokenizer.tokenize(doc)
    count = 0   #Counting the total number of words in the doc
     
    for word in tokens:
        if word not in my_stopwords:
            after_stemming = stemmer.stem(word)     # Stemming to get the root
            words_in_doc.append(after_stemming)
            count += 1

    tmp_doc_list['filename'] = filename
    tmp_doc_list['words'] = words_in_doc
    tmp_doc_list['count'] = count
    return tmp_doc_list


# This function finds the cosine similarity for the files where all the words match i.e for the documents which contain all the query tokens
def cosine_similarity_all_words_match(query_dict,filename,list_actual_sim,temp_list):
    final_similarity = 0
    for every_word in temp_list:
        for i,j in temp_list[every_word]:
            if(filename == str(i)):
                wt_term_doc = float(j)
                wt_term_query = query_dict[every_word]
                final_similarity += (wt_term_doc * wt_term_query)
                break

    list_actual_sim[filename] = final_similarity
    return(list_actual_sim)


# Calculating the Upper Bound for all the words
def calculate_upper_bound(upperbound_dict,temp_list):    
    for every_word in temp_list:
        (i,j) = temp_list[every_word][-1]
        upperbound_dict[every_word] = float(j)
    return(upperbound_dict)


# This function finds the cosine similarity for the files where some the words match i.e for the documents which contain only few of the query tokens
def cosine_similarity_some_words_match(query_dict,filename,list_upperbound_actual_sim,temp_list):
    upperbound_dict = {}
    upperbound_dict = calculate_upper_bound(upperbound_dict,temp_list)
    similarity_query_doc = 0
    
    for every_word in query_dict:
        flag = 0
        for doc,tfidf_wt in temp_list[every_word]:
            if (doc == filename):   # Actual value found for the word
                similarity_query_doc += (tfidf_wt * query_dict[every_word])
                flag = 1

        if(flag == 0):  # No value found for the word, so fetch it from the upperbound_dict
            similarity_query_doc += (upperbound_dict[every_word] * query_dict[every_word])
        
    list_upperbound_actual_sim[filename] = similarity_query_doc
    return(list_upperbound_actual_sim)


# Compares the actual scores vs the actual score and outer bound score
def check_scores(list_actual_sim,list_upperbound_actual_sim):
    if not (list_actual_sim and list_upperbound_actual_sim):
        return "None",0

    if not list_actual_sim:
        return "Fetch More",0
    
    if not list_upperbound_actual_sim:
        for actual_doc,tfidf in list_actual_sim:
            return (actual_doc,tfidf)

    for actual_doc,tfidf in list_actual_sim:
        for up_doc,wt in list_upperbound_actual_sim:
            if(tfidf >= wt):
                return (actual_doc,tfidf)
            else:   # Just break as the list is sorted in Desc order and we need to compare the 1st value only
                return "Fetch More",0


# Main Function which processes all the documents
tfidf_list = {}
doc_frequency = {}

def main_function():
    doc_list=[]

    # Read all the input files from the directory
    for filename in os.listdir(corpusroot):
        tmp_doc_list = clean_files(filename)
        doc_list.append(tmp_doc_list)

    # Calculating the term Frequency for every word in every document
    list_with_term_freq = []

    for one_tuple in doc_list:
        tuple_dict = {}
        tmp_word_and_count = []
        tuple_dict['filename'] = one_tuple['filename']
        tuple_dict['count'] = one_tuple['count'] 

        word_counts = Counter(one_tuple['words'])    # Counter to count the occurrence of every word
        
            
        for i,j in word_counts.most_common():
            tmp_list = {}
            tmp_list['word'] = str(i)
            tmp_list['wordcount'] = str(j)
            tmp_word_and_count.append(tmp_list)
            
        tuple_dict['words'] = tmp_word_and_count
        list_with_term_freq.append(tuple_dict)


    # Calculating Document Frequency
    list_of_words = []
    for rec in list_with_term_freq:
        for word in rec['words']:
            list_of_words.append(word['word'])

    global doc_frequency
    doc_frequency = Counter(list_of_words)

    # Calculating TFIDF
    for rec in list_with_term_freq:
        dict_words={}

        filename = rec['filename']
        count = rec['count']
        
        for word in rec['words']:
            wt = (1+log10(float(word['wordcount']))) * (log10(30/float(doc_frequency[word['word']])))
            dict_words[word['word']] = wt

        tfidf_list[filename] = dict_words


    # For normalizing the vectors (L2 normalization)
    for every_file in tfidf_list:
        tmp_list = []
        sum_of_squares = 0
        sqrt_of_sum_of_squares = 0

        for each_word in tfidf_list[every_file]:
            tmp_list.append(tfidf_list[every_file][each_word])      # Getting the tfidf values of all the words in the list
        
        squared_list = [i ** 2 for i in tmp_list]
        sum_of_squares = sum(squared_list)
        sqrt_of_sum_of_squares = sqrt(sum_of_squares)

        for each_word in tfidf_list[every_file]:
            tfidf_list[every_file][each_word] = (tfidf_list[every_file][each_word]) / sqrt_of_sum_of_squares

    return


# Clean the query vector and return the document with the wt
def query(query_words):
    query_dict = {}
    list_upperbound_actual_sim = {}
    list_actual_sim={}
    query_words = query_words.lower()

    # Tokenize the Query Vector
    tokens = new_tokenizer.tokenize(query_words)    

    words_in_doc = []
    for word in tokens:
        if word not in my_stopwords:
            after_stemming = stemmer.stem(word)     # Stemming to get the root
            words_in_doc.append(after_stemming)

    word_counts = Counter(words_in_doc)    # Counter to count the occurrence of every word
            
    for i,j in word_counts.most_common():
        wt = (1+log10(float(j)))
        query_dict[i] = wt


    temporary_list = []
    ## Normalizing the query Vector
    for word in query_dict:
        temporary_list.append(query_dict[word])
        
    query_squared_list = [i ** 2 for i in temporary_list]
    query_sum_of_squares = sum(query_squared_list)
    query_sqrt_of_sum_of_squares = sqrt(query_sum_of_squares)

    for word in query_dict:
        query_dict[word] = query_dict[word] / query_sqrt_of_sum_of_squares


    ##### 7.1 Creating a Posting List#####
    list_of_words = list(set(words_in_doc))

    main_dict = {}
    for each_word in list(set(list_of_words)):
        temp_dict = {}
        
        for every_record in tfidf_list:
            if each_word in tfidf_list[every_record]:
                temp_dict[every_record] = tfidf_list[every_record][each_word]

        main_dict[each_word] = temp_dict

    # Sort the list in Desc Order
    for every_word in main_dict:
        main_dict[every_word] = sorted(main_dict[every_word].items(), key=operator.itemgetter(1), reverse=True)     # Sorting every element in the dictionary based on the tfidf

    temp_list = {}
    ## 7.2 Take top 10 records only##
    for every_word in main_dict:
        temp_list[every_word] = main_dict[every_word][0:10]


    ### 7.3,7.4 If document d appears in the top-10 elements of every query token or some query tokens
    list_of_docs = []
    for everyword in temp_list:
        for i,j in temp_list[everyword]:
            list_of_docs.append(i)

    for i,j in Counter(list_of_docs).most_common():
        if(j == len(main_dict)):
            cosine_similarity_all_words_match(query_dict,str(i),list_actual_sim,temp_list)
        else:
            list_upperbound_actual_sim = cosine_similarity_some_words_match(query_dict,str(i),list_upperbound_actual_sim,temp_list)

    list_upperbound_actual_sim = sorted(list_upperbound_actual_sim.items(), key=operator.itemgetter(1), reverse=True)   # actual+ upperbound score
    list_actual_sim = sorted(list_actual_sim.items(), key=operator.itemgetter(1), reverse=True)     # actual score

    result_query,tfidf = check_scores(list_actual_sim,list_upperbound_actual_sim)
    return(result_query,tfidf)


# Get the tfidf weight of a token in a document
def getweight(filename,token):
    if(token not in tfidf_list[filename]):
        return 0
    else:
        return (tfidf_list[filename][token])


# Get the inverse document frequency of a token
def getidf(token):
    global doc_frequency
    if(token not in doc_frequency):
        return -1
    else:
        return (log10(30/doc_frequency[token]))    


# Start of the program
main_function()

print("(%s, %.12f)" % query("health insurance wall street"))
print("(%s, %.12f)" % query("particular constitutional amendment"))
print("(%s, %.12f)" % query("terror attack"))
print("(%s, %.12f)" % query("vector entropy"))

print("%.12f" % getweight("2012-10-03.txt","health"))
print("%.12f" % getweight("1960-10-21.txt","reason"))
print("%.12f" % getweight("1976-10-22.txt","agenda"))
print("%.12f" % getweight("2012-10-16.txt","hispan"))
print("%.12f" % getweight("2012-10-16.txt","hispanic"))

print("%.12f" % getidf("health"))
print("%.12f" % getidf("agenda"))
print("%.12f" % getidf("vector"))
print("%.12f" % getidf("reason"))
print("%.12f" % getidf("hispan"))
print("%.12f" % getidf("hispanic"))