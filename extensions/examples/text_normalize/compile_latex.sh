echo 'Compile latex'

rm -rf outputs/books/*
python render_book.py
xelatex -output-directory outputs/books/ outputs/books/book.tex
