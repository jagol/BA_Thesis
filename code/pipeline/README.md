Here, the Ontology-Learning pipeline will reside.

Before calling the preprocessor, use the bash command

`csplit --prefix=europarl_ europarl-v7.de-en.en /^$/ {*}`

to split the europarl corpus into documents and move the documents into one folder.