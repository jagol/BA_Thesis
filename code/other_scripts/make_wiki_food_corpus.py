import sys
from gensim.corpora import WikiCorpus

# """
# Creates a corpus from Wikipedia dump file.
# Inspired by
# https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html
# which is inspired by:
# https://github.com/panyang/Wikipedia_Word2vec/blob/master/v1/process_wiki.py
# """
#
#
def make_corpus(fn_in, fn_out):
    """Convert Wikipedia xml dump file to text corpus"""
    with open(fn_out, 'w', encoding='utf8') as f_out:
        wiki = WikiCorpus(fn_in)

    i = 0
    for text in wiki.get_texts():
        # output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        print(text)
        i += 1
        if i > 2:
            break
        # if (i % 10000 == 0):
        #     print('Processed ' + str(i) + ' articles')
    # print('Processing complete!')


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print(
            'Usage: python make_wiki_corpus.py <wikipedia_dump_file> <processed_text_file>')
        sys.exit(1)
    fn_in = sys.argv[1]
    fn_out = sys.argv[2]
    make_corpus(fn_in, fn_out)
