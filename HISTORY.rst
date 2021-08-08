================================================================================
History
================================================================================

Current
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
