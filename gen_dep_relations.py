from nltk.parse.stanford import StanfordDependencyParser
import pandas as pd
import argparse
import pdb

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Prepare dependency parser data')
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
	with open(args.data+"test_dependency.txt","w") as f:
		with open(args.data+"test_deformed_sentences.txt","w") as d:
			for x in X['Message']:
				count += 1
				print("testing instance = " + str(count))
				try:
					parsed = dep_parser.raw_parse(x)
					dependencies = [list(parse.triples()) for parse in parsed]
					f.write(str(dependencies))
				except Exception as e:
					d.write(x)
					d.write("\n")
					test_deformed_count += 1
					empty_list = []
					f.write(str(empty_list))
				f.write("\n")
	print("test_deformed_count = " + str(test_deformed_count))
	print("test file completed.")

	X = pd.read_csv(trainF,sep="\t")
	train_deformed_count = 0
	count = 0
	with open(args.data+"train_dependency.txt","w") as f:
		with open(args.data+"train_deformed_sentences.txt","w") as d:
			for x in X['Message']:
				count += 1
				print("training instance = " + str(count))
				try:
					parsed = dep_parser.raw_parse(x)
					dependencies = [list(parse.triples()) for parse in parsed]
					f.write(str(dependencies))
				except Exception as e:
					d.write(x)
					d.write("\n")
					train_deformed_count += 1
					empty_list = []
					f.write(str(empty_list))
				f.write("\n")
	print("train_deformed_count = " + str(train_deformed_count))
	print("train file completed.")