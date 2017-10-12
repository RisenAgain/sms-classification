from nltk.parse.stanford import StanfordDependencyParser
import pandas as pd
import argparse
import pickle
import pdb

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Prepare dependency tree data')
	parser.add_argument('--data', type=str,help="Data file prefix")
	parser.parse_args()
	args = parser.parse_args()

	trainF = args.data + "train.tsv"
	testF = args.data + "test.tsv"
	tuneF = args.data + "tune.tsv"

	dep_parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

	X = pd.read_csv(testF,sep="\t")
	test_deformed_count = 0
	count = 0
	test_list = []

	for x in X['Message']:
		count += 1
		print("testing instance = " + str(count))
		try:
			parsed = dep_parser.raw_parse(x)
			tree = [parse.tree() for parse in parsed]
			test_list.append(tree)
		except Exception as e:
			test_deformed_count += 1
			empty_list = []
			test_list.append(empty_list)

	with open(args.data+"test_data_tree.p","wb") as f:
		pickle.dump(test_list,f)

	print("test_deformed_count = " + str(test_deformed_count))
	print("test file completed.")

	X = pd.read_csv(trainF,sep="\t")
	train_deformed_count = 0
	count = 0
	train_list = []

	for x in X['Message']:
		count += 1
		print("testing instance = " + str(count))
		try:
			parsed = dep_parser.raw_parse(x)
			tree = [parse.tree() for parse in parsed]
			train_list.append(tree)
		except Exception as e:
			train_deformed_count += 1
			empty_list = []
			train_list.append(empty_list)

	with open(args.data+"train_data_tree.p","wb") as f:
		pickle.dump(train_list,f)

	print("train_deformed_count = " + str(train_deformed_count))
	print("train file completed.")