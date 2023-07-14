================================================================================
History
================================================================================

6.4.0 (2023-07-14)
--------------------------------------------------------------------------------

* GH-686: Fix fixed_words regex

6.3.0 (2023-06-28)
--------------------------------------------------------------------------------

* GH-685: Support MacOS ARM

6.2.0 (2023-03-04)
--------------------------------------------------------------------------------

* GH-173: Add Text to Speech API by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/668
* GH-502: Provide training script for word segmentation and pos tagging and named entity recognition by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/666
* GH-622: Create UTS_Dictionary v1.0 datasets by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/663

6.1.4 (2023-02-26)
--------------------------------------------------------------------------------

* GH-588: Support underthesea_core with python 3.11 by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/659
* GH-588: update underthesea_core version by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/660

6.1.3 (2023-02-25)
--------------------------------------------------------------------------------

* Bump django from 4.1.6 to 4.1.7 in /apps/languages/backend by @dependabot in https://github.com/undertheseanlp/underthesea/pull/652
* Bump django from 3.2.17 to 3.2.18 in /apps/service by @dependabot in https://github.com/undertheseanlp/underthesea/pull/651
* GH-502: Training example for word segmentation by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/654
* Add two new datasets UTS_Text and UTS_WTK

6.1.2 (2023-02-15)
--------------------------------------------------------------------------------

* GH-648: Add option fixed_words to tokenize and word_tokenize api by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/649

6.1.1 (2023-02-10)
--------------------------------------------------------------------------------

* GH-641: Correct the error with the filename of the dataset in Windows by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/644
* Bump django from 3.2.16 to 3.2.17 in /apps/service by @dependabot in https://github.com/undertheseanlp/underthesea/pull/640
* Bump django from 4.1.4 to 4.1.6 in /apps/languages/backend by @dependabot in https://github.com/undertheseanlp/underthesea/pull/639
* Bump ua-parser-js from 0.7.28 to 0.7.33 in /apps/directory/components/json_viewer/component/frontend by @dependabot in https://github.com/undertheseanlp/underthesea/pull/636
* Bump future from 0.16.0 to 0.18.3 in /apps/service by @dependabot in https://github.com/undertheseanlp/underthesea/pull/645

6.1.0 (2023-02-08)
--------------------------------------------------------------------------------

* GH-641: fix issue filename of dataset is not correct by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/642

6.0.3 (2023-01-25)
--------------------------------------------------------------------------------

* GH-622: Initialize Dictionary page feature by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/633
* GH-622: Add dictionary page by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/634

6.0.2 (2023-01-17)
--------------------------------------------------------------------------------

* GH-628: Create unittest for django API by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/629
* GH-607: add test frontend with jest by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/630

**Full Changelog**: https://github.com/undertheseanlp/underthesea/compare/v6.0.1...v6.0.2

6.0.1 (2023-01-08)
--------------------------------------------------------------------------------

* GH-607: add Articles UI by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/620
* GH-621: Corpus CP_Vietnamese_VLC_v2_2022 by @rain1024 in https://github.com/undertheseanlp/underthesea/pull/624

6.0.0 (2023-01-01)
--------------------------------------------------------------------------------

* Happy New Year 2023! Let's bump up the version! (GH-616)

1.4.1 (2022-12-17)
--------------------------------------------------------------------------------

* Create underthesea app (GH-607)
* Add viet2ipa module (GH-437)
* Training NER model with VLSP2016 dataset using BERT (GH-437)
* Remove unidecode as a dependency (GH-569)

1.3.5 (2022-10-31)
--------------------------------------------------------------------------------

* Add Text Normalization module (GH-534)
* Release underthesea_core version 0.0.5a2 (GH-550)
* Support GLIBC_2.17 (GH-530)
* Update resources path (GH-540)
* Fix function word_tokenize (GH-528)

1.3.4 (2022-01-08)
--------------------------------------------------------------------------------

* Demo chatbot with  rasa (GH-513)
* Lite version of underthesea (GH-505)
* Increase word_tokenize speed 1.5 times (GH-185)
* Add build for windows (GH-185)

1.3.3 (2021-09-02)
--------------------------------------------------------------------------------

* Update torch and transformer dependency (GH-403)

1.3.2 (2021-08-04)
--------------------------------------------------------------------------------

* Publish two ABSA open datasets (GH-417)
* Migrate from travis-ci to github actions (GH-410)
* Update ParserTrainer (GH-392)
* Add pipeline folder (GH-351)

1.3.1 (2021-01-11)
--------------------------------------------------------------------------------

* Compatible with newer version of scikit-learn (GH-313)
* Retrain classification and sentiment models with latest version of scikit-learn (GH-381)
* Add ClassifierTrainer (from languageflow) (GH-381)
* Add 3 new datasets (GH-351)
* [Funny Update] Change underthesea's avatar (GH-371)
* [CI] Add Stale App: Automatically close stale Issues and Pull Requests that tend to accumulate during a project (GH-351)

1.3.0 (2020-12-11)
--------------------------------------------------------------------------------

* Remove languageflow dependency (GH-364)
* Remove tabulate dependency (GH-364)
* Remove scores in text classification and sentiment section (GH-351)
* Add information of dependency_parse module in info function (GH-351)
* Try to use Github Actions (GH-353)
* Dependency Parsing (GH-157)

1.2.3 (2020-11-28)
--------------------------------------------------------------------------------

* Refactor config for resources (GH-300)
* Thêm API xử lý dữ liệu (GH-299)

1.2.2 (2020-11-04)
--------------------------------------------------------------------------------

* Remove nltk strict version (GH-308)
* Add word_hyphen rule (GH-290)
* Sanity check python version (GH-320)
* Handle exception case in sentiment module (GH-321)
* Cập nhật quản lý resources từ languageflow (GH-295)
* Loại bỏ languageflow trong quá trình cài đặt (GH-295)
* Cập nhật phiên bản fasttext (GH-304)

1.1.16 (2019-06-15)
--------------------------------------------------------------------------------

* Bumping up version of the languageflow dependency (GH-231)
* Update phiên bản scikit-learn 0.20.2 (GH-229)
* Cập nhật lại các dependencies (GH-241)
* Cập nhật mô hình trên bộ dữ liệu VNTC (GH-246)
* Cập nhật mô hình trên bộ dữ liệu UTS2017_BANK_TC (GH-243)
* Cập nhật mô hình trên bộ dữ liệu UTS2017_BANK_SA (GH-244)
* Lỗi với các câu sentiment ở demo (GH-236)
* Thống nhất cách đặt tên và quản lý model (GH-225)

1.1.12 (2019-03-13)
--------------------------------------------------------------------------------

* Add sentence segmentation feature

1.1.9 (2019-01-01)
--------------------------------------------------------------------------------

* Improve speed of word_tokenize function
* Only support python 3.6+
* Use flake8 for style guide enforcement

1.1.8 (2018-06-20)
--------------------------------------------------------------------------------

* Fix word_tokenize error when text contains tab (\t) character
* Fix regex_tokenize with url

1.1.7 (2018-04-12)
--------------------------------------------------------------------------------

* Rename word_sent function to word_tokenize
* Refactor version control in setup.py file and __init__.py file
* Update documentation badge url

1.1.6 (2017-12-26)
--------------------------------------------------------------------------------

* New feature: aspect sentiment analysis
* Integrate with languageflow 1.1.6
* Fix bug tokenize string with '=' (#159)

1.1.5 (2017-10-12)
--------------------------------------------------------------------------------

* New feature: named entity recognition
* Refactor and update model for word_sent, pos_tag, chunking


1.1.4 (2017-09-12)
--------------------------------------------------------------------------------

* New feature: text classification
* [bug] Fix Text error
* [doc] Add facebook link

1.1.3 (2017-08-30)
--------------------------------------------------------------------------------

* Add live demo: https://underthesea.herokuapp.com/

1.1.2 (2017-08-22)
--------------------------------------------------------------------------------

* Add dictionary

1.1.1 (2017-07-05)
--------------------------------------------------------------------------------

* Support Python 3
* Refactor feature_engineering code

1.1.0 (2017-05-30)
--------------------------------------------------------------------------------

* Add chunking feature
* Add pos_tag feature
* Add word_sent feature, fix performance
* Add Corpus class
* Add Transformer classes
* Integrated with dictionary of Ho Ngoc Duc
* Add travis-CI, auto build with PyPI

1.0.0 (2017-03-01)
--------------------------------------------------------------------------------

* First release on PyPI.
* First release on Readthedocs
