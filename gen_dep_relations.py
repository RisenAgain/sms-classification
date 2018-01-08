from nltk.parse.stanford import StanfordDependencyParser
import pandas as pd
import argparse
import pickle
import copy

def gen_dep_files(read_file,op,dep_parser):
	X = pd.read_csv(read_file,sep="\t")
	deformed_count = 0
	count = 0
	rel_list = []
	tree_list = []
		
	with open(args.data+op+"_deformed_sentences.txt","w") as d:
		for x in X['Message']:
			count += 1
			print(op+" instance = " + str(count))
			try:
				parsed = dep_parser.raw_parse(x)
				parsed_copy = copy.deepcopy(parsed)
				relations = [list(parse.triples()) for parse in parsed]
				tree = [parse.tree() for parse in parsed_copy]
				rel_list.append(relations)
				tree_list.append(tree)
			except Exception as e:
				d.write(x)
				d.write("\n")
				deformed_count += 1
				empty_list = []
				rel_list.append(empty_list)
				tree_list.append(empty_list)
				
	print(op+"_deformed_count = " + str(deformed_count))

	with open(args.data+op+"_dependency_rel.p","wb") as rel:
		pickle.dump(rel_list,rel)

	with open(args.data+op+"_dependency_tree.p","wb") as tr:
		pickle.dump(tree_list,tr)

	print(op+" file completed.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Prepare dependency parser data')
	parser.add_argument('--data', type=str,help="Data file prefix")
	parser.parse_args()
	args = parser.parse_args()

	trainF = args.data + "train.tsv"
	testF = args.data + "test.tsv"
	# tuneF = args.data + "tune.tsv"    

	dep_parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

	gen_dep_files(trainF,"train",dep_parser)
	gen_dep_files(testF,"test",dep_parser)