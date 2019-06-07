import json
import os
import numpy as np


def count_words():
    print('Counting word occurences per file...')
    path = '/mnt/storage/harlie/users/jgoldz/preprocessed_corpora/sp'
    fpaths = [os.path.join(path, fn) for fn in os.listdir(path)
              if os.path.isfile(os.path.join(path, fn))]
    # word_counts = {}  # {lemma: [count_doc1, count_doc2, ...]}
    id_num = 0
    lemma_to_id = {}
    id_to_lemma = {}
    word_counts = []
    for fpath in fpaths:
        print(fpath)
        with open(fpath, 'r', encoding='utf8') as f:
            summaries = json.load(f)
            for i in summaries:
                i_int = int(i)
                doc = summaries[i]
                for key in doc:
                    sent = doc[key]
                    for word in sent:
                        if word[1].startswith('N'):
                            lemma = word[2]
                            if lemma not in lemma_to_id:
                                lemma_to_id[lemma] = id_num
                                id_to_lemma[id_num] = lemma
                                word_counts.append(np.zeros(94475, dtype=int))
                                word_counts[id_num][i_int] += 1
                                id_num += 1
                                print(id_num)
                            else:
                                id_num = lemma_to_id[lemma]
                                word_counts[id_num][i_int] += 1



    print('Writing lemma_to_id to file...')
    with open(path+'lemma_to_id.json', 'w', encoding='utf8') as f:
        json.dump(lemma_to_id, f, ensure_ascii=False)

    print('Writing id_to_lemma to file...')
    with open(path+'id_to_lemma.json', 'w', encoding='utf8') as f:
        json.dump(id_to_lemma, f, ensure_ascii=False)

    print('Writing word_counts to file...')
    with open(path+'word_counts.tsv', 'w', encoding='utf8') as f:
        for id_num in range(len(word_counts)):
            lemma_freqs = word_counts[id_num]
            lemma = id_to_lemma[id_num]
            line = '{}\t{}\t'.format(lemma, id_num)
            for lf in lemma_freqs:
                line += str(lf)+'\t'
            f.write(line)


if __name__ == '__main__':
    count_words()