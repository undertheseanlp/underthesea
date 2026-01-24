# -*- coding: utf-8 -*-
"""Tests for underthesea.transforms.conll module.

GitHub issue #714: KeyError: '__getitems__' when calling dependency_parse()
"""
from unittest import TestCase
from underthesea.transforms.conll import Sentence, CoNLL


class TestSentenceGetattr(TestCase):
    """Test case for GitHub issue #714: KeyError: '__getitems__'

    The Sentence.__getattr__ method was raising KeyError instead of
    AttributeError when accessing undefined attributes. This caused
    issues with PyTorch/pandas batching operations that look up
    __getitems__.
    """

    def test_undefined_attribute_raises_attribute_error(self):
        """Accessing undefined attribute should raise AttributeError, not KeyError."""
        transform = CoNLL()
        sentence = Sentence(transform)
        sentence.values = [['test']]

        with self.assertRaises(AttributeError):
            _ = sentence.undefined_attribute

    def test_dunder_getitems_raises_attribute_error(self):
        """Accessing __getitems__ should raise AttributeError, not KeyError.

        This is the specific error from issue #714 where PyTorch/pandas
        looks up __getitems__ for batching operations.
        """
        transform = CoNLL()
        sentence = Sentence(transform)
        sentence.values = [['test']]

        with self.assertRaises(AttributeError):
            _ = sentence.__getitems__

    def test_valid_attribute_access_via_maps(self):
        """Valid attributes in maps should still be accessible."""
        transform = CoNLL()
        sentence = Sentence(transform)
        # Manually set up maps and values to test the __getattr__ logic
        sentence.maps['test_field'] = 0
        sentence.keys.add('test_field')
        sentence.values = [['test_value']]

        # Should not raise any exception
        result = sentence.test_field
        self.assertEqual(['test_value'], result)

    def test_attribute_error_message_contains_class_name(self):
        """AttributeError message should contain the class name."""
        transform = CoNLL()
        sentence = Sentence(transform)
        sentence.values = [['test']]

        with self.assertRaises(AttributeError) as context:
            _ = sentence.nonexistent

        self.assertIn('Sentence', str(context.exception))
        self.assertIn('nonexistent', str(context.exception))
