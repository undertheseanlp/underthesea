import os
import tempfile
from unittest import TestCase

from underthesea.pipeline.dependency_parse.visualize import (
    DEFAULT_OPTIONS,
    render,
    save,
)


class TestVisualize(TestCase):
    def setUp(self):
        self.sample_parse = [
            ('Tôi', 2, 'nsubj'),
            ('yêu', 0, 'root'),
            ('Việt Nam', 2, 'obj'),
        ]

    def test_render_returns_svg(self):
        svg = render(self.sample_parse)
        self.assertIn('<svg', svg)
        self.assertIn('</svg>', svg)

    def test_render_contains_words(self):
        svg = render(self.sample_parse)
        self.assertIn('Tôi', svg)
        self.assertIn('yêu', svg)
        self.assertIn('Việt Nam', svg)

    def test_render_contains_relations(self):
        svg = render(self.sample_parse)
        self.assertIn('nsubj', svg)
        self.assertIn('root', svg)
        self.assertIn('obj', svg)

    def test_render_with_custom_options(self):
        options = {'font_size': 20, 'arc_color': '#ff0000'}
        svg = render(self.sample_parse, options)
        self.assertIn('font-size: 20px', svg)
        self.assertIn('#ff0000', svg)

    def test_save_creates_file(self):
        svg = render(self.sample_parse)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
            filepath = f.name

        try:
            save(svg, filepath)
            self.assertTrue(os.path.exists(filepath))
            with open(filepath, encoding='utf-8') as f:
                content = f.read()
            self.assertIn('<svg', content)
        finally:
            os.unlink(filepath)

    def test_render_root_arc(self):
        svg = render(self.sample_parse)
        # Root arc should have a vertical line from top
        self.assertIn('root', svg)

    def test_render_empty_parse(self):
        svg = render([])
        self.assertIn('<svg', svg)
        self.assertIn('</svg>', svg)

    def test_default_options_exist(self):
        self.assertIn('word_spacing', DEFAULT_OPTIONS)
        self.assertIn('font_size', DEFAULT_OPTIONS)
        self.assertIn('arc_color', DEFAULT_OPTIONS)
