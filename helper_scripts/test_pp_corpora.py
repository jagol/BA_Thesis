from collections import defaultdict
import json

print('Analyze corpora.')

path_pp_token_corpus = 'output/dblp/processed_corpus/pp_token_corpus.txt'
path_pp_lemma_corpus = 'output/dblp/processed_corpus/pp_lemma_corpus.txt'
path_token_idx_corpus = 'output/dblp/processed_corpus/token_idx_corpus.txt'
path_lemma_idx_corpus = 'output/dblp/processed_corpus/lemma_idx_corpus.txt'

paths = [path_pp_token_corpus, path_pp_lemma_corpus, path_token_idx_corpus,
         path_lemma_idx_corpus]

for path in paths:
    wdict = defaultdict(int)
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            if line != '\n':
                line = line.strip('\n')
                words = line.split(' ')
                for w in words:
                    wdict[w]+=1
    print(30*'-')
    print(path)
    print(len(wdict))
    print(sorted(list(wdict.keys()))[:10])

# ------------------------------------------------

print(50*'*')
print(50*'*')
print('Analyze extracted terms.')

path_token_terms = 'output/dblp/processed_corpus/token_terms.txt'
path_lemma_terms = 'output/dblp/processed_corpus/lemma_terms.txt'
path_token_term_idxs = 'output/dblp/processed_corpus/token_terms_idxs.txt'
path_lemma_term_idxs = 'output/dblp/processed_corpus/lemma_terms_idxs.txt'

paths = [path_token_terms, path_lemma_terms, path_token_term_idxs,
         path_lemma_term_idxs]

for path in paths:
    words = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip('\n')
            words.append(line)
    print(30 * '-')
    print(path)
    print(len(words))
    print(sorted(words)[-10:])

# ------------------------------------------------

print(50*'*')
print(50*'*')
print('Analyze embedding vocab.')

vocab = []
with open('cmd_test.vec', 'r') as f:
    for line in f:
        line = line.strip('\n')
        vals = line.split(' ')
        vocab.append(vals[0])

# print(vocab)

# Check which lemma-terms are not in embedding vocab.
with open(path_pp_lemma_corpus, 'r') as f:
    wdict = defaultdict(int)
    for line in f:
        if line != '\n':
            line = line.strip('\n')
            words = line.split(' ')
            for w in words:
                wdict[w]+=1
print(len(wdict))
tg = [w for w in vocab if w not in wdict]
print(len(vocab))
print(len(tg))
# print(tg)

# ----------------------------------------
# print(50*'~')
# print('Analyze which ids produce KeyError in pipeline but are not ')
#
# no_vecs = ['19423', '19424', '19425', '19426', '19427', '19429', '19430', '19431', '19432', '19433', '19434', '19435', '19436', '19437', '19438', '19439', '19440', '19444', '19445', '19446', '19447', '19451', '19452', '19453', '19454', '19455', '19456', '19457', '19458', '19459', '19460', '19461', '19462', '19463', '19464', '19465', '19466', '19467', '19468', '19469', '19470', '19471', '19472', '19473', '19474', '19475', '19476', '19477', '19478', '19479', '19480', '19481', '19482', '19483', '19484', '19485', '19486', '19487', '19488', '19489', '19490', '19492', '19493', '19494', '19495', '19496', '19497', '19498', '19499', '19500', '19501', '19502', '19503', '19504', '19505', '19506', '19507', '19508', '19509', '19510', '19511', '19512', '19514', '19515', '19517', '19518', '19519', '19520', '19521', '19522', '19523', '19524', '19525', '19526', '19527', '19528', '19529', '19530', '19531', '19532', '19533', '19534', '19535', '19536', '19537', '19538', '19539', '19540', '19542', '19543', '19544', '19546', '19547', '19548', '19549', '19550', '19551', '19552', '19553', '19554', '19555', '19556', '19557', '19558', '19559', '19560', '19561', '19562', '19563', '19564', '19565', '19566', '19567', '19568', '19569', '19570', '19571', '19572', '19573', '19574', '19575', '19576', '19577', '19578', '19579', '19580', '19581', '19582', '19583', '19584', '19585', '19586', '19587', '19588', '19589', '19590', '19591', '19592', '19593', '19594', '19595', '19597', '19598', '19599', '19602', '19603', '19604', '19605', '19606', '19607', '19608', '19609', '19610', '19611', '19612', '19613', '19614', '19615', '19616', '19617', '19618', '19619', '19620', '19621', '19622', '19623', '19624', '19625', '19626', '19627', '19628', '19629', '19630', '19631', '19632', '19633', '19634', '19636', '19637', '19638', '19639', '19640', '19641', '19642', '19643', '19645', '19646', '19647', '19648', '19649', '19650', '19651', '19652', '19653', '19654', '19655', '19656', '19657', '19658', '19659', '19660', '19661', '19662', '19663', '19664', '19665', '19666', '19667', '19668', '19669', '19670', '19671', '19672', '19673', '19674', '19675', '19676', '19677', '19678', '19679', '19680', '19682', '19683', '19684', '19685', '19686', '19687', '19688', '19689', '19690', '19691', '19692', '19693', '19694', '19695', '19696', '19697', '19698', '19700', '19701', '19702', '19703', '19704', '19705', '19706', '19708', '19709', '19710', '19711', '19712', '19713', '19714', '19715', '19716', '19717', '19718', '19719', '19720', '19721', '19722', '19723', '19724', '19725', '19726', '19727', '19728', '19729', '19730', '19731', '19732', '19733', '19734', '19735', '19736', '19737', '19738', '19739', '19740', '19741', '19742', '19745', '19746', '19747', '19748', '19749', '19750', '19751', '19752', '19753', '19754', '19755', '19758', '19761', '19762', '19763', '19764', '19765', '19766', '19767', '19768', '19769', '19770', '19771', '19772', '19773', '19774', '19775', '19776', '19777', '19778', '19779', '19780', '19781', '19782', '19783', '19784', '19785', '19786', '19787', '19788', '19789', '19790', '19791', '19792', '19793', '19794', '19795', '19796', '19797', '19798', '19799', '19800', '19801', '19802', '19803', '19804', '19805', '19806', '19807', '19808', '19809', '19810', '19811', '19813', '19814', '19815', '19816', '19817', '19820', '19821', '19822', '19823', '19824', '19825', '19826', '19827', '19828', '19829', '19830', '19831', '19832', '19833', '19834', '19835', '19836', '19837', '19838', '19839', '19840', '19841', '19842', '19843', '19844', '19845', '19846', '19847', '19848', '19849', '19850', '19851', '19852', '19853', '19854', '19855', '19856', '19857', '19859', '19860', '19861', '19862', '19863', '19864', '19865', '19866', '19867', '19869', '19870', '19871', '19872', '19873', '19874', '19875', '19876', '19877', '19878', '19879', '19880', '19881', '19882', '19883', '19884', '19885', '19886', '19887', '19888', '19889', '19890', '19891', '19892', '19893', '19894', '19895', '19896', '19897', '19898', '19899', '19900', '19901', '19902', '19903', '19904', '19906', '19907', '19908', '19909', '19910', '19911', '19912', '19913', '19914', '19915', '19916', '19917', '19918', '19919', '19920', '19921', '19922', '19923', '19924', '19925', '19926', '19927', '19928', '19929', '19930', '19931', '19932', '19933', '19934', '19935', '19936', '19937', '19938', '19939', '19940', '19941', '19942', '19943', '19944', '19945', '19946', '19947', '19948', '19949', '19950', '19951', '19952', '19953', '19954', '19955', '19956', '19957', '19958', '19959', '19960', '19961', '19962', '19963', '19964', '19965', '19966', '19967', '19968', '19969', '19970', '19971', '19972', '19973', '19974', '19975', '19976', '19977', '19978', '19979', '19980', '19981', '19983', '19984', '19985', '19986', '19987', '19988', '19989', '19991', '19992', '19993', '19994', '19995', '19996', '19997', '19998', '19999', '20000', '20001', '20002', '20003', '20004', '20005', '20006', '20007', '20008', '20009', '20010', '20011', '20012', '20013', '20014', '20015', '20016', '20017', '20018', '20019', '20020', '20022', '20023', '20024', '20025', '20026', '20027', '20028', '20029', '20030', '20031', '20032', '20033', '20034', '20035', '20036', '20037']
#
# wdict = defaultdict(int)
# with open(path_lemma_idx_corpus, 'r', encoding='utf8') as f:
#     for line in f:
#         if line != '\n':
#             line = line.strip('\n')
#             words = line.split(' ')
#             for w in words:
#                 wdict[w]+=1
#
# l = []
# for v in no_vecs:
#     if v not in wdict:
#         l.append(v)
# print(len(l), len(no_vecs))

# Check if all ids in emb vocab are in term_idxs and other way around.
# Check if all ids in emb vocab are in idx-to-term-embs an other way
# around.


def load_emb_vocab(path):
    vocab = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            vals = line.split(' ')
            vocab.append(vals[0])
    return vocab


def load_terms(path):
    terms = []
    with open(path, 'r') as f:
        for line in f:
            term = line.strip('\n')
            terms.append(term)
    return terms


print('Analyze emb vocab over lemma-idxs.')
path_lemma_idx_vec = 'lemma_idx.vec'
path_idx_to_lemma = 'output/dblp/indexing/idx_to_lemma.json'
path_lemma_to_idx = 'output/dblp/indexing/lemma_to_idx.json'

lemma_term_idxs = load_terms(path_lemma_term_idxs)
emb_vocab = load_emb_vocab(path_lemma_idx_vec)
with open(path_idx_to_lemma, 'r') as f:
    idx_to_lemma = json.load(f)
with open(path_lemma_to_idx, 'r') as f:
    lemma_to_idx = json.load(f)

print('length lemma_term_idxs: {}'.format(len(lemma_term_idxs)))
print('length emb vocab: {}'.format(len(emb_vocab)))
print('length idx_to_lemma: {}'.format(len(idx_to_lemma)))
print('length lemma_to_idx: {}'.format(len(lemma_to_idx)))
diff_emb_vocab_idx_to_lemma = [x for x in emb_vocab if x not in idx_to_lemma]
print('In emb vocab but not in idx_to_lemma: {}'.format(diff_emb_vocab_idx_to_lemma))
idxs = [str(i) for i in range(19424)]
print('print if idx not in emb_vocab:')
for idx in idxs:
    if idx not in emb_vocab:
        print(idx)
print('print if idx not in idx_to_lemma:')
for idx in idxs:
    if idx not in idx_to_lemma:
        print(idx)