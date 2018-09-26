This is a pipeline to extract an ontology from raw text.

### How To

Until now the pipeline requires the additional python3 packages:
- spacy (including a language model for english which has to be downloaded seperately)


1. For the europarl corpus: Use the bash command `csplit --prefix=europarl_ europarl-v7.de-en.en /^$/ {*}` to split the europarl corpus into documents and move the documents into a separate folder. 
For other corpora: Split the corpus into documents (if it not already is). Move those documents into a separate folder. The input files should contain one sentence per line.

2. Then adjust the paths in `configs_template.json` to your setup and rename the file to `configs.json`. 

3. Call `preprocessor.py` if you want to use the local paths or `preprocessor.py -s` if you want to use the server paths.
4. Call `term_extractor.py` if you want to use the local paths or `term_extractor.py -s` if you want to use the server paths.
