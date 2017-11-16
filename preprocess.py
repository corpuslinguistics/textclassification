# -*- coding: utf8 -*-

import ConfigParser
import fasttext
import imblearn.ensemble
import imblearn.over_sampling
import imblearn.under_sampling
from itertools import cycle
import logging
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy import interp
from sklearn.metrics import auc, average_precision_score, classification_report, log_loss, precision_recall_curve, roc_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import binarize, label_binarize
import sklearn.model_selection
import sys
from Cython.Compiler.Naming import label_prefix

reload(sys)
sys.setdefaultencoding('utf-8')

config_parser = ConfigParser.ConfigParser()
config_parser.read('input/config.txt')
n_classes = int(config_parser.get('input', 'n_classes'))
proba_threshold = float(config_parser.get('input', 'proba_threshold'))
n_subsets = int(config_parser.get('input', 'n_subsets'))
epoch = int(config_parser.get('input', 'epoch'))
    
def load_sample(file_name):
    x_list = list()
    y_list = list()
    proba_list = list()
    f = file(file_name)
    for line in f:                
        line = line.decode('utf8').strip()
        item_list = line.split('\t')
        x_list.append('\t'.join(item_list[3:]))
        y_list.append(item_list[0].replace('__label__', ''))
        if len(item_list[1]) > 0 and len(item_list[2]) > 0:
            proba_list.append([float(item_list[1]), float(item_list[2])])
    f.close()
    proba_list = np.array(proba_list)
#     for x in x_list:
#         print x
#     print y_list
#     print proba_list    

    return x_list, y_list, proba_list


def serialize(file_name, x_list, y_list, proba_list=None, prefix=''):
    line_list = list()
    for i in range(len(x_list)):
        line = prefix + y_list[i] + '\t'
        if proba_list is None:
            line += '\t\t'
        else:
            line += str(proba_list[i][0]) + '\t' + str(proba_list[i][1]) + '\t' 
        line += x_list[i].encode('utf8')
        
        line_list.append(line)
#             line_list.append(str(int(y_list[i])).encode('utf8') + '\t\t\t' + str(x_list[i][0]).encode('utf8'))
#             line_list.append(str(int(y_list[i])).encode('utf8') + '\t' + str(proba_list[i][0]) + '\t' + str(proba_list[i][1]) + '\t' + str(x_list[i][0]).encode('utf8'))
        
    f = file(file_name, 'w')
    f.write('\n'.join(line_list))
    f.close()

    return


def classify(train_file, x_test_list):
    # Set parameters
#     epoch = 100
    word_ngrams = 3
    bucket=2000000
    silent = 1
    
    # Train the classifier
    classifier = fasttext.supervised(train_file, '/tmp/classifier', epoch=epoch, word_ngrams=word_ngrams, bucket=bucket, silent=silent)
                    
#     # Predict
#     f = file(test_file)
#     line_list = list()
#     for line in f:
#         line = line.decode('utf8').strip()
#         line_list.append(line)
#     f.close()
#
#             y_hat_list = classifier.predict(line_list)
#             for y_hat in y_hat_list:
#                 print y_hat[0]
    
    # Or predict with the probability
    # k-best labels, sorted
    label_list = classifier.predict_proba(x_test_list, k=n_classes)
#     print label_list
    # sum(proba) < 1
    # [[(u'0', 0.863281), (u'1', 0.134766)], [(u'1', 0.849609), (u'0', 0.148438)], ... 

    y_hat_list = list()
    proba_list = list()
    for label in label_list:
#         print label
        y_hat_list.append(label[0][0])
#         proba_list.append([label[0][1], label[1][1]])
        # swap if necessary
        proba = [0] * 2
        proba[int(label[0][0])] = label[0][1]
        proba[int(label[1][0])] = label[1][1]
#         proba = dict()
#         proba[label[0][0]] = label[0][1]
#         proba[label[1][0]] = label[1][1]
        proba_list.append(proba)

    proba_list = np.array(proba_list)
#     print y_hat_list
#     print proba_list
#     print proba_list[:, 0]

    return y_hat_list, proba_list


def evaluate(y_test_list, y_hat_list, proba_list, file_name):
    print y_test_list
    print y_hat_list
    
    # macro-average: 对类求平均
    print(classification_report(y_test_list, y_hat_list))
    
#     print(log_loss(y_test_list, y_hat_list))
    
    # Shape will be [n_samples, 1] for binary problems.
    # y_test_list = label_binarize(y_test_list, classes=[0, 1])
    y_test_list = label_binarize(y_test_list, classes=[0, 1, 2])
#     print y_test_list
#     print y_test_list[:, 0]

    precision = dict()
    recall = dict()
    threshold = dict()
    average_precision = dict()
    for i in range(n_classes):
        # 寒
        # precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test_list[:, 1], proba_list[:, 1])
        # 热
        # precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test_list[:, 0], proba_list[:, 0])
#         print y_test_list[:, i]
#         print proba_list
        precision[i], recall[i], threshold[i] = precision_recall_curve(y_test_list[:, i], proba_list[:, i])
        average_precision[i] = average_precision_score(y_test_list[:, i], proba_list[:, i])
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_list[:, 0:2].ravel(), proba_list.ravel())
    average_precision["micro"] = average_precision_score(y_test_list[:, 0:2], proba_list, average="micro")
#     print('Average precision score, micro-averaged over all classes: {0:0.2f}'
#           .format(average_precision["micro"]))    
    
    # A "macro-average"
    # 仿照roc
    average_precision["macro"] = average_precision_score(y_test_list[:, 0:2], proba_list, average="macro")
#     print('Average precision score, macro-averaged over all classes: {0:0.2f}'
#           .format(average_precision["macro"]))    
    
    # First aggregate all false positive rates
    all_precision = np.unique(np.concatenate([precision[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_recall = np.zeros_like(all_precision)
    for i in range(n_classes):
        mean_recall += interp(all_precision, precision[i], recall[i])
    
    # Finally average it and compute AUC
    mean_recall /= n_classes
    
    precision["macro"] = all_precision
    recall["macro"] = mean_recall
    average_precision["macro"] = auc(precision["macro"], recall["macro"])

    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
    lines.append(l)
    labels.append('iso-f1 curves')

    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    
    l, = plt.plot(recall["macro"], precision["macro"], color='pink', lw=2)
    lines.append(l)
    labels.append('macro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["macro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))
    
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    
#     plt.show()
    plt.savefig(file_name + '_prc.png', format='png', dpi=300)
    plt.close()

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_list[:, i], proba_list[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_list[:, 0:2].ravel(), proba_list.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    
#     plt.show()
    plt.savefig(file_name + '_roc.png', format='png', dpi=300)
    plt.close()
    
    return


def label_cold_heat():
    # 删除与辩证无关术语，例如“阴阳辩证”中的“三阴交”，“腰阳关”
    stop_category_set = set()
    f = open('input/stop_category.txt')
    for line in f:
        line = line.decode('utf8').strip()
        stop_category_set.add(line)
#         print line
    f.close()

    stop_word_set = set()
    f = file('input/entity_type.txt')
    for line in f:
        try:
            line = line.decode('utf8').strip()
            item_list = line.split()
            if item_list[1] not in stop_category_set:
                continue
            if item_list[0].find(u'寒') > 0 or item_list[0].find(u'热') > 0:
                stop_word_set.add(item_list[0])
#                 print line
        except:
            logging.error(line)
    f.close()
    print 'stop words: ' + str(len(stop_word_set))
    
    x_list = list()
    y_list = list()
    f = file('input/ancient_modern_case_utf8.txt')
    for line in f:
        line = line.decode('utf8').strip()
        line = line.replace('\t', ' ')
#         print line

        # 找到所有出现停用词的位置
        match_list = list()
        for stop_word in stop_word_set:
            idx = line.find(stop_word)
            while idx >= 0:    
                match_list.append([idx, len(stop_word)]) 
                idx = line.find(stop_word, idx + 1)
#                 print line
#                 print stop_word

        # 将所有出现停用词的位置置为空
        line_null = line
        for match in match_list:
            line_null = line_null[:match[0]] + ' '.join(['' for i in range(match[1])]) + line_null[match[0] + match[1]:]
#         if line != line_null:
#             print line
#             print line_null

        # multi-label
        if line_null.find(u'寒') >= 0 and line_null.find(u'热') >= 0:
            continue
        
        if line_null.find(u'寒') >= 0:
            idx = 0
            line_split = list()
            item_list = line_null.split(u'寒')
            for item in item_list:
                line_split.append(line[idx:idx + len(item)])
                idx += len(item) + 1
            line = ''.join(line_split)
#             print line
            line = re.sub(r'(.)', r'\g<1> ', line)
#             print line
            x_list.append([line])
#             y_list.append(u'__label__寒')
            y_list.append('__label__1')
   
        if line_null.find(u'热') >= 0:
            idx = 0
            line_split = list()
            item_list = line_null.split(u'热')
            for item in item_list:
                line_split.append(line[idx:idx + len(item)])
                idx += len(item) + 1
            line = ''.join(line_split)
#             print line
            line = re.sub(r'(.)', r'\g<1> ', line)
#             print line
            x_list.append([line])
#             y_list.append(u'__label__热')
            y_list.append('__label__0')
    f.close()
        
    return x_list, y_list
    
    
def classify_cold_heat():
    x_list, y_list = label_cold_heat()
    print 'total: ' + str(len(x_list))
    serialize(x_list, y_list, None, 'output/ancient_modern_case_coldheat.txt')
                       
    x_train_list, x_test_list, y_train_list, y_test_list = sklearn.model_selection.train_test_split(np.array(x_list).reshape(-1, 1), y_list, test_size=0.1, shuffle=True)
    print 'train: ' + str(len(x_train_list))
    serialize(x_train_list, y_train_list, None, 'output/ancient_modern_case_coldheat_train.txt')
    print 'test: ' + str(len(x_test_list))
    serialize(x_test_list, y_test_list, None, 'output/ancient_modern_case_coldheat_test.txt')
          
#     # 写入临时文件
#     train_size_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000]
                 
    # 对训练集进行欠采样
    rus = imblearn.under_sampling.RandomUnderSampler()
    x_train_undersample_list, y_train_undersample_list = rus.fit_sample(np.array(x_train_list).reshape(-1, 1), y_train_list)
    serialize(x_train_undersample_list, y_train_undersample_list, None, 'output/ancient_modern_case_coldheat_train_undersample.txt')
    print 'undersample: ' + str(len(x_train_undersample_list))
             
    # 对训练集进行过采样
    ros = imblearn.over_sampling.RandomOverSampler()
    x_train_oversample_list, y_train_oversample_list = ros.fit_sample(np.array(x_train_list).reshape(-1, 1), y_train_list)
    serialize(x_train_oversample_list, y_train_oversample_list, None, 'output/ancient_modern_case_coldheat_train_oversample.txt')
    print 'oversample: ' + str(len(x_train_oversample_list))
               
    # bagging
    ee = imblearn.ensemble.EasyEnsemble(n_subsets=n_subsets)
    x_train_bagging_list, y_train_bagging_list = ee.fit_sample(np.array(x_train_list).reshape(-1, 1), y_train_list)
    for i in xrange(len(x_train_bagging_list)):
        serialize(x_train_bagging_list[i], y_train_bagging_list[i], None, 'output/ancient_modern_case_coldheat_train_bagging_' + str(i) + '.txt')
        print len(x_train_bagging_list[i])
    print 'easy ensemble: ' + str(len(x_train_bagging_list))
     
    x_test_list, y_test_list, proba_list = load_sample('output/ancient_modern_case_coldheat_test.txt')
     
    print 'normal'
    y_hat_list, proba_list = classify('output/ancient_modern_case_coldheat_train.txt', 'output/ancient_modern_case_coldheat_test.txt')
    y_hat_list = binarize(proba_list[:, 1].reshape(-1, 1), proba_threshold)
    serialize(x_test_list, y_hat_list, proba_list, 'output/ancient_modern_case_coldheat_test_hat.txt')
    x_hat_list, y_hat_list, proba_list = load_sample('output/ancient_modern_case_coldheat_test_hat.txt')
    evaluate(y_test_list, y_hat_list, proba_list, 'output/figure/normal')
           
    print 'undersample'
    y_hat_list, proba_list = classify('output/ancient_modern_case_coldheat_train_undersample.txt', 'output/ancient_modern_case_coldheat_test.txt')
    y_hat_list = binarize(proba_list[:, 1].reshape(-1, 1), proba_threshold)
    serialize(x_test_list, y_hat_list, proba_list, 'output/ancient_modern_case_coldheat_test_hat_undersample.txt')
    x_hat_list, y_hat_list, proba_list = load_sample('output/ancient_modern_case_coldheat_test_hat_undersample.txt')
    evaluate(y_test_list, y_hat_list, proba_list, 'output/figure/undersample')
               
    print 'oversample'
    y_hat_list, proba_list = classify('output/ancient_modern_case_coldheat_train_oversample.txt', 'output/ancient_modern_case_coldheat_test.txt')
    y_hat_list = binarize(proba_list[:, 1].reshape(-1, 1), proba_threshold)
    serialize(x_test_list, y_hat_list, proba_list, 'output/ancient_modern_case_coldheat_test_hat_oversample.txt')
    x_hat_list, y_hat_list, proba_list = load_sample('output/ancient_modern_case_coldheat_test_hat_oversample.txt')
    evaluate(y_test_list, y_hat_list, proba_list, 'output/figure/oversample')
    
    print 'bagging'
    y_bagging_list = list()
    proba_bagging_list = list()
    for i in xrange(n_subsets):
        print i
        y_hat_list, proba_list = classify('output/ancient_modern_case_coldheat_train_bagging_' + str(i) + '.txt', 'output/ancient_modern_case_coldheat_test.txt')
        y_hat_list = binarize(proba_list[:, 1].reshape(-1, 1), proba_threshold)
        serialize(x_test_list, y_hat_list, proba_list, 'output/ancient_modern_case_coldheat_test_hat_bagging_' + str(i) + '.txt')
        x_hat_list, y_hat_list, proba_list = load_sample('output/ancient_modern_case_coldheat_test_hat_bagging_' + str(i) + '.txt')
        evaluate(y_test_list, y_hat_list, proba_list, 'output/figure/bagging_' + str(i))
        y_bagging_list.append(y_hat_list)
        proba_bagging_list.append(proba_list)
#     print y_bagging_list
#     print proba_bagging_list
    y_bagging_list = np.array(y_bagging_list)
    proba_bagging_list = np.array(proba_bagging_list)
#     print y_bagging_list
#     print proba_bagging_list
  
    label_set = {0, 1}
    y_hat_list = list()
    proba_list = list()
    for i in xrange(len(x_test_list)):
        y_hat_list.append(max(label_set, key=list(y_bagging_list[:, i]).count))
        proba_list.append([np.average(proba_bagging_list[:, i, 0]), np.average(proba_bagging_list[:, i, 1])])
#     print proba_bagging_list[:, 1, 0]
#     print proba_bagging_list[:, 1, 1]
#     print np.average(proba_bagging_list[:, 1, 0])
#     print np.average(proba_bagging_list[:, 1, 1])
     
#     print y_hat_list
#     print proba_list
    y_hat_list = np.array(y_hat_list)
    proba_list = np.array(proba_list)
    y_hat_list = binarize(proba_list[:, 1].reshape(-1, 1), proba_threshold)
    serialize(x_test_list, y_hat_list, proba_list, 'output/ancient_modern_case_coldheat_test_hat_bagging.txt')
    x_hat_list, y_hat_list, proba_list = load_sample('output/ancient_modern_case_coldheat_test_hat_bagging.txt')
    evaluate(y_test_list, y_hat_list, proba_list, 'output/figure/bagging')
        
    
def label_yin_yang():
    x_list = list()
    y_list = list()
    f = file('input/modern_case_utf8.txt')
    for line in f:
        try:
            line = line.decode('utf8').strip()
                            
            cnt = 0
            if line.find(u'阴虚') >= 0:
#                 label = u'阴虚'
                label = '0'
                cnt += 1
            if line.find(u'阳虚') >= 0:
#                 label = u'阳虚'
                label = '1'
                cnt += 1
            if cnt == 1:
                # 仅出现阴虚，或仅出现阳虚
#                 print line
                
                item_list = line.split('\t')
#                 print item_list[11]
                match = re.search(u'[阴阳]虚', item_list[11])
                if match: 
                    # 中医证候
                    del item_list[11]
#                     print '\t'.join(item_list)
                    x_list.append('\t'.join(item_list))
#                     y_list.append('__label__' + label)
                    y_list.append(label)
#                 else:
#                     print line
#             else:
#                 print line
        except:
            logging.error(line)
    f.close()    
    
    return x_list, y_list
        
        
def classify_yin_yang():
    x_list, y_list = label_yin_yang()
    print 'total: ' + str(len(x_list))
    serialize('output/modern_case_yin_yang.txt', x_list, y_list, proba_list=None, prefix='')

    # cross validation
    x_list, y_list, proba_list = load_sample('output/modern_case_yin_yang.txt')
    kf = KFold(n_splits=10, shuffle=False)
    for train_idx, test_idx in kf.split(x_list, y_list):
#         print train_idx
#         print test_idx
         
        x_test_list = list()
        for i in test_idx:
            x_test_list.append(x_list[i])             
        y_test_list = list()
        for i in test_idx:
            y_test_list.append(y_list[i])             
        serialize('output/modern_case_yin_yang_test.txt', x_test_list, y_test_list, proba_list=None, prefix='')
        
        x_train_list = list()
        for i in train_idx:
            x_train_list.append(x_list[i])
        y_train_list = list()
        for i in train_idx:
            y_train_list.append(y_list[i])             
        serialize('output/modern_case_yin_yang_train.txt', x_train_list, y_train_list, proba_list=None, prefix='__label__')
         
        x_test_list, y_test_list, proba_list = load_sample('output/modern_case_yin_yang_test.txt')
      
        y_hat_list, proba_list = classify('output/modern_case_yin_yang_train.txt', x_test_list)
        y_hat_bin = binarize(proba_list[:, 1].reshape(-1, 1), proba_threshold).tolist()
        y_hat_list = list()
        for i in xrange(len(y_hat_bin)):
            y_hat_list.append(str(int(y_hat_bin[i][0])))
        serialize('output/modern_case_yin_yang_test_hat.txt', x_test_list, y_hat_list, proba_list)
        x_hat_list, y_hat_list, proba_list = load_sample('output/modern_case_yin_yang_test_hat.txt')
        evaluate(y_test_list, y_hat_list, proba_list, 'output/figure/normal')


if __name__ == '__main__':
    """ main """
    # Log
    logging.basicConfig(filename=config_parser.get('output', 'log'), filemode='w', level=logging.DEBUG, 
    format='[%(levelname)s\t%(asctime)s\t%(funcName)s\t%(lineno)d]\t%(message)s')
    
    classify_yin_yang()
    
#     classify_cold_heat()
    