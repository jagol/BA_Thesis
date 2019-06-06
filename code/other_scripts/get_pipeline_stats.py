import json
import argparse


def get_stats(stats_dict):
    for s in stats_dict:
        fpath = stats_dict[s][0]
        num_objects = get_num_json_objects(fpath)
        stats_dict[s][1] = num_objects
    return stats_dict


def get_num_json_objects(fpath):
    """Count and return the number of json objects."""
    with open(fpath, 'r', encoding='utf8') as f:
        d = json.load(f)
        if isinstance(d, dict):
            return len(d.keys())
        elif isinstance(d, list):
            return len(d)
        else:
            raise Exception('{}: format not known!'.format(fpath))


def add_threshholds(args, stats_dict):
    threshholds = {}
    if args.cval_direct_threshhold:
        threshholds['cval_d'] = args.cval_direct_threshhold
    if args.tfidf_direct_threshhold:
        threshholds['tfidf_d'] = args.tfidf_direct_threshhold
    if args.cval_comb_threshhold:
        threshholds['cval_i'] = args.cval_comb_threshhold
    if args.tfidf_comb_threshhold:
        threshholds['tfidf_i'] = args.tfidf_comb_threshhold
    if args.number_of_files:
        num_files = args.number_of_files
        stats_dict['num_files'] = num_files

    stats_dict['threshholds'] = threshholds
    return stats_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cval_d", "--cval_direct_threshhold")
    parser.add_argument('-tfidf_d', '--tfidf_direct_threshhold')
    parser.add_argument("-cval_i", "--cval_comb_threshhold")
    parser.add_argument('-tfidf_i', '--tfidf_comb_threshhold')
    parser.add_argument('-nf', '--number_of_files')
    args = parser.parse_args()
    stats_dict = {
        'cval': ['temp/cval.json', 0],
        'onto_terms': ['temp/onto_terms.json', 0],
        'term_info': ['temp/term_info.json', 0],
        'tfidf': ['temp/tfidf.json', 0],
        'triples': ['temp/triples.json', 0],
        'word_counts': ['temp/word_counts.json', 0]
    }
    stats_dict = get_stats(stats_dict)
    stats_dict = add_threshholds(args, stats_dict)
    with open('pipeline_stats.json', 'w', encoding='utf8') as f:
        json.dump(stats_dict, f)


if __name__ == '__main__':
    main()
