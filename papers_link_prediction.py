import pandas as pd
import networkx as nx
import numpy as np
import random

from tqdm import tqdm_notebook

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def classify_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
                     
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    precision_train = precision_score(y_train, y_pred_train)
    precision_test = precision_score(y_test, y_pred_test)
    recall_train = recall_score(y_train, y_pred_train)
    recall_test = recall_score(y_test, y_pred_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_macro_train = f1_score(y_train, y_pred_train, average='macro')
    f1_macro_test = f1_score(y_test, y_pred_test, average='macro')
    f1_micro_train = f1_score(y_train, y_pred_train, average='micro')
    f1_micro_test = f1_score(y_test, y_pred_test, average='micro')
    logloss_train = log_loss(y_train, y_pred_train)
    logloss_test = log_loss(y_test, y_pred_test)
    roc_auc_train = roc_auc_score(y_train, y_pred_train)
    roc_auc_test = roc_auc_score(y_test, y_pred_test)

    return model, y_pred_train, y_pred_test, \
        precision_train, precision_test, recall_train, recall_test, accuracy_train, accuracy_test, \
        f1_macro_train, f1_macro_test, f1_micro_train, f1_micro_test, \
        roc_auc_train, roc_auc_test, logloss_train, logloss_test 

def generate_negative_edges(graph, count_gen_edges, part_neg_directed):
    negative_edges = set()
    nodes = list(graph.nodes())
    edges = list(graph.edges())

    count_neg_directed = int(part_neg_directed*count_gen_edges)
    for a, b in edges:
        if len(negative_edges) >= count_neg_directed:
            break
        if not graph.has_edge(b, a):
            negative_edges.add((b, a))       
    
    while len(negative_edges) < count_gen_edges:
        i = random.randint(0, len(nodes) - 1)
        j = random.randint(0, len(nodes) - 1)
        if (i != j) and not graph.has_edge(nodes[i], nodes[j]):
            negative_edges.add((nodes[i], nodes[j]))
    return list(negative_edges)


def generate_negative_edges_test(graph, test_nodes, count_gen_edges):
    negative_edges = set()
    nodes = list(graph.nodes())
    while len(negative_edges) < count_gen_edges:
        i = random.randint(0, len(test_nodes) - 1)
        j = random.randint(0, len(nodes) - 1)
        if i == j:
            continue
        if graph.has_edge(test_nodes[i], nodes[j]):
            continue
        negative_edges.add((test_nodes[i], nodes[j]))
    return list(negative_edges)


def link_to_time(links_df, meta_df):
    years = meta_df[['id','year']]
    years.columns = ['citing', 'year_citing']
    links_df = links_df.merge(years, how = 'left', on = 'citing')
    years.columns = ['cited', 'year_cited']
    links_df = links_df.merge(years, how='left', on = 'cited')
    links_df['out_cites_count'] = links_df.groupby('citing')['cited'].transform(lambda x: x.count())
    links_df['in_cites_count'] = links_df.groupby('cited')['citing'].transform(lambda x: x.count())
    links_df['cite_rank'] = links_df.groupby('citing')['cited'].transform(lambda x: x.rank())
    return links_df


def train_test_split_by_year(links_years_df, year, part=None):
    """
    links_years_df - cite edges dataframe with year of citing paper
    year - first year of test period
    part - part of test period edges to include into train"""
    if part:
        train = links_years_df[(links_years_df.year_citing < year)|\
                               ((links_years_df.year_citing >= year)&\
                                ((links_years_df.cite_rank < part*links_years_df.out_cites_count + 1)|\
                                (links_years_df.in_cites_count == 1)))]
        test = links_years_df[((links_years_df.year_citing >= year)&\
                                (links_years_df.cite_rank >= part*links_years_df.out_cites_count + 1)&\
                              (links_years_df.in_cites_count != 1))]
    else:
        train = links_years_df[(links_years_df.year_citing < year)]
        test = links_years_df[(links_years_df.year_citing >= year)]
    return train.reset_index(drop=True), test.reset_index(drop=True)
    
def train_test_preprocess(positive_df, neg_list):
    pairs = list(zip(list(positive_df['citing']), list(positive_df['cited']), [1]*len(positive_df)))
    neg_pairs = list(zip(list(zip(*neg_list))[0],list(zip(*neg_list))[1], [0]*len(neg_list)))
    pairs += neg_pairs
    random.shuffle(pairs)
    return pairs
    
    
def product(u,v):
    return u*v
def mean(u,v):
    return (u+v)/2
def l1(u,v):
    return np.abs(u-v)
def l2(u,v):
    return (u-v)**2

def concat(u,v):
    return np.concatenate([u, v])