from sklearn import feature_extraction
import numpy as np

from languageflow.transformer.count import CountVectorizer


class TfidfVectorizer(feature_extraction.text.TfidfVectorizer):
    """
    Convert a collection of raw documents to a matrix of TF-IDF features.

    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.
        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.
        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.
    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.
    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.
    strip_accents : {'ascii', 'unicode', None}
        Remove accents during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.
    analyzer : string, {'word', 'char'} or callable
        Whether the feature should be made of word or character n-grams.
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.
    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.
    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.
    stop_words : string {'english'}, list, or None (default)
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. 'english' is currently the only supported string
        value.
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.
        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.
    lowercase : boolean, default True
        Convert all characters to lowercase before tokenizing.
    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).
    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    max_features : int or None, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
        This parameter is ignored if vocabulary is not None.
    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.
    binary : boolean, default=False
        If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only that the tf term in tf-idf
        is binary. (Set idf and normalization to False to get 0/1 outputs.)
    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().
    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    use_idf : boolean, default=True
        Enable inverse-document-frequency reweighting.
    smooth_idf : boolean, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.
    idf_ : array, shape = [n_features], or None
        The learned idf vector (global term weights)
        when ``use_idf`` is set to True, None otherwise.
    stop_words_ : set
        Terms that were ignored because they either:
          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).
        This is only available if no vocabulary was given.
    """

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        super(TfidfVectorizer, self).__init__(input, encoding, decode_error,
                                              strip_accents, lowercase,
                                              preprocessor, tokenizer, analyzer,
                                              stop_words, token_pattern,
                                              ngram_range, max_df, min_df,
                                              max_features, vocabulary, binary,
                                              dtype, norm, use_idf, smooth_idf,
                                              sublinear_tf)

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return term-document matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects
        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        documents = super(TfidfVectorizer, self).fit_transform(
            raw_documents=raw_documents, y=y)
        count = CountVectorizer(encoding=self.encoding,
                                decode_error=self.decode_error,
                                strip_accents=self.strip_accents,
                                lowercase=self.lowercase,
                                preprocessor=self.preprocessor,
                                tokenizer=self.tokenizer,
                                stop_words=self.stop_words,
                                token_pattern=self.token_pattern,
                                ngram_range=self.ngram_range,
                                analyzer=self.analyzer,
                                max_df=self.max_df,
                                min_df=self.min_df,
                                max_features=self.max_features,
                                vocabulary=self.vocabulary_,
                                binary=self.binary,
                                dtype=self.dtype)
        count.fit_transform(raw_documents=raw_documents, y=y)
        self.period_ = count.period_
        self.df_ = count.df_
        self.n = count.n
        return documents
