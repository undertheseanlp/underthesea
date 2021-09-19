# Developer Guides

Step 1: Download Wikipedia dump

```
export TS=20210720
mkdir -p ~/.underthesea/data/viwiki-$TS
cd ~/.underthesea/data/viwiki-$TS
wget https://dumps.wikimedia.org/viwiki/$TS/viwiki-$TS-pages-articles.xml.bz2
wget https://raw.githubusercontent.com/NTT123/viwik18/master/WikiExtractor.py
bzip2 -d viwiki-$TS-pages-articles.xml.bz2
python WikiExtractor.py --no-templates -b 10M -s --lists viwiki-$TS-pages-articles.xml
```

Step 2: Clean data

```
python underthesea/utils/col_wiki_clean.py
python underthesea/utils/col_wiki_ud.py
```

Step 3: Run

```
python underthesea/utils/col_dictionary.py
python underthesea/utils/col_dictionary_import.py
```
For Mac OS >= Mojave, alternatively run
```
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python underthesea/utils/col_dictionary.py
python underthesea/utils/col_dictionary_import.py
```
