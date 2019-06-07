import time
from utility_functions import get_docs


def load_corpus(path):
	corpus = {}
	for i, doc in enumerate(get_docs(path)):
		corpus[i] = doc
	return corpus


def load_tf(path):
	tf = {}
	with open(path, 'r', encoding='utf8') as f:
		for i, line in enumerate(f):
			tf[i] = json.load(line.strip('\n'))
	return tf


def load_df(path):
	with open(path, 'r', encoding='utf8') as f:
		return json.load(df)


def main():

	cpath = '/mnt/storage/harlie/users/jgoldz/output/dblp/processed_corpus/pp_token_corpus.txt'
	tf_path = '/mnt/storage/harlie/users/jgoldz/output/dblp/frequencies/tf_tokens.json'
	tf_path = '/mnt/storage/harlie/users/jgoldz/output/dblp/frequencies/df_tokens.json'
	print('Starting...')
	time.sleep(5)
	print('load_corpus...')
	corpus = load_corpus(cpath)
	tf = load_tf(tf_path)
	load_df(df_path)
	print(x)
	print('loading_done...')
	time.sleep(5)
	print('Ending...')


if __name__ == '__main__':
	main()