### Installation

##### Download the source code:

`$ mkdir building_on_taxogen`

`$ cd building_on_taxogen`

`$ git clone https://github.com/jagol/BA_Thesis`

##### You can download the preprocessed dblp corpus from the 
TaxoGen repository: https://github.com/franticnerd/taxogen
Or you can download the original corpus at 
https://dblp.uni-trier.de/faq/How+can+I+download+the+whole+dblp+dataset
and use this repository's preprocessing.

##### The pipeline was tested on python 3.6.5. Other version might also work.

To install the required packages use:

`$ pip3 install -r requirements.txt`

##### Download an English language model from: 
https://spacy.io/usage/models
 and point to it in `configs.json`

### Preprocessing

Configure an output path in `configs.json`. The pipeline will place 
all temporary and output files in this directory (or subdirectories).

##### If downloaded from TaxoGen repository:

For the thesis I used the preprocessed corpus from the TaxoGen 
repository. If you have downloaded this corpus, adjust the paths in 
`configs.json`.

You can choose between `server` paths and `local` paths. If you filled in the 
server paths then use: 

`$ python3 preprocessing_main.py -c dblp -l server`

Else use: 

`$ python3 preprocessing_main_tg.py -c dblp -l local`

##### If downloaded from dblp repository:

To preprocess the corpus use:

`$ python3 preprocessing_main.py -c dblp -l local` 

Note: This step will take one to two days. Better do it on a server.

### Generating the taxonomy.

Use `configs.json` to configure which embeddings are used, 
which clustering algorithm is used and
if the label score is used.

To generate the taxonomy if you configured the server paths use:

`$ python3 generate_taxonomy.py -c dblp -l server`

If you configured the local paths, use:

`$ python3 generate_taxonomy.py -c dblp -l local`

### Evaluation

To generate the evaluation files use:

`$ python3 generate_eval_files.py -i <path_to_output_dir>/concept_terms/tax_labels_sim.csv -o <output directory>`

This script produces three files in the output directory:
`hypernym_eval.csv`, `issubtopic_eval.csv`, `topic_level.csv`

The relations in these files can be labeled with 0 (False) or 1 (True) 
in the first free column.

Then calculate the relation accuracy with:

`$ python3 process_eval.py -t <path_to_output_dir>/concept_terms/tax_labels_sim.csv -e <output_directory_of_precious_script>`
 