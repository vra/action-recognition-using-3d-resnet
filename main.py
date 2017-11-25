from __future__ import print_function
import json
import time
import argparse

import numpy as np
from numpy.random import shuffle as shuffle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

def get_params():
	parser = argparse.ArgumentParser('Pytorch Tools for Extract 3D Resnet \
		Features and Classify Videos')
	parser.add_argument('-dataset', required=True, choices=['ucf101', 'hmdb51'])
#	parser.add_argument('-pooling', require=True, choices=['mean', 'max'])
	return parser
	
	
def svm_classifier(x_train, y_train, x_test, y_test):
	clf = OneVsRestClassifier(LinearSVC())
	clf.fit(x_train, y_train)
	accu = clf.score(x_test, y_test)
	print ('accu: ', accu)


def parse_json(json_file, tr_itme_file, te_item_file):
	tr_items = [l.strip().split(' ')[0] for l in 
				open(tr_item_file, 'r').readlines()]
	te_items = [l.strip().split(' ')[0] for l in
				open(te_item_file, 'r').readlines()]
	tr_cls = [int(l.strip().split(' ')[1]) for l in
				open(tr_item_file, 'r').readlines()]
	te_cls = [int(l.strip().split(' ')[1]) for l in
				open(te_item_file, 'r').readlines()]

	item_cls_dict = {}
	for i in range(len(tr_items)):
		item_cls_dict[tr_items[i]] = tr_cls[i]
	for i in range(len(te_items)):
		item_cls_dict[te_items[i]] = te_cls[i]

	with open(json_file) as f:
		jdata = json.load(f)

	nsample = len(jdata)
	assert nsample > 0
	ndim = len(jdata[0]['clips'][0]['features'])

	mean_data = np.zeros((nsample, ndim))
	max_data = np.zeros((nsample, ndim))
	item_data = {}
	for idx, data in enumerate(jdata):
		item = data['video'].split('.')[0]
		curr_data = []
		for clip in data['clips']:
			clip_feature = np.array(clip['features'])
			curr_data.append(clip_feature)
		curr_data = np.array(curr_data)
		curr_mean_data = np.mean(curr_data, axis=0)
		curr_max_data = np.max(curr_data, axis=0)
		
		mean_data[idx,:] = curr_mean_data
		max_data[idx, :] = curr_max_data


	tr_mean_data = []
	te_mean_data = []
	tr_max_data = []
	te_max_data = []
	tr_cls = []
	te_cls = []
	for idx, data in enumerate(jdata):
		curr_item = data['video'].split('.')[0]
		if curr_item in tr_items:
			tr_cls.append(item_cls_dict[curr_item])
			tr_mean_data.append(mean_data[idx,:])
			tr_max_data.append(max_data[idx,:])
		elif curr_item in te_items:
			te_cls.append(item_cls_dict[curr_item])
			te_mean_data.append(mean_data[idx,:])
			te_max_data.append(max_data[idx,:])

	tr_mean_data = np.array(tr_mean_data)
	tr_max_data = np.array(tr_max_data)
	te_mean_data = np.array(te_mean_data)
	te_max_data = np.array(te_max_data)
	tr_cls = np.array(tr_cls)
	te_cls = np.array(te_cls)
	
	return tr_mean_data, tr_max_data, te_mean_data, te_max_data, tr_cls, te_cls



if __name__ == '__main__':
	parser = get_params()
	args = parser.parse_args()

	tr_item_file =  'data//lists/{}/tr_items_labels.txt'.format(args.dataset)
	te_item_file =  'data/lists/{}/te_items_labels.txt'.format(args.dataset)
	json_file = 'data/jsons/{}/resnet_3d_34.json'.format(args.dataset)
	
	t1 = time.time()
	print('begin loading data...')
	tr_me, tr_ma, te_me, te_ma, tr_lbl, te_lbl = parse_json(json_file,
		tr_item_file, te_item_file)
	t2 = time.time()
	print('finish loading data, time cost {0:.2f}s'.format(t2-t1))
	t3 = time.time()
	print('running svm on mean data...')
	svm_classifier(tr_me, tr_lbl, te_me, te_lbl)
	t4 = time.time()
	print('finish svm on mean data, time cost {0:.2f}s'.format(t4-t3))
	t5 = time.time()
	print('running svm on max data...')
	svm_classifier(tr_ma, tr_lbl, te_ma, te_lbl)
	t6 = time.time()
	print('finish svm on max data, time cost {0:.2f}s'.format(t6-t5))
