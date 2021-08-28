# Developer Guides

Step 1: Download wikipedia dump 

```
export TS=20210720
mkdir -p ~/.underthesea/data/viwiki-$TS
cd ~/.underthesea/data/viwiki-TS
wget https://dumps.wikimedia.org/viwiki/20210820/viwiki-20210820-pages-articles.xml.bz2
wget https://raw.githubusercontent.com/NTT123/viwik18/master/WikiExtractor.py
bzip2 -d viwiki-20210820-pages-articles.xml.bz2
python WikiExtractor.py -s --lists viwiki-20210820-pages-articles.xml -q -o - | perl -CSAD -Mutf8 cleaner.pl > viwik18.txt
```

Step 2: Run 

```
python utils/col_wiki_clean.py
python utils/col_wiki_ud.py
```

Step 3: Run 

```
python utils/col_dictionary.py
python utils/col_dictionary_import.py 
```