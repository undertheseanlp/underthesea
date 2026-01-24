"""
Tests for JAX API compatibility in the tts module.

These tests verify that the JAX tree.map API is used correctly after
the deprecation of jax.tree_map in JAX v0.6.0.

See: https://github.com/undertheseanlp/underthesea/issues/762
"""
import unittest
from pathlib import Path


class TestJaxTreeMapAPI(unittest.TestCase):
    """Test that jax.tree.map works correctly (replaced deprecated jax.tree_map)."""

    def test_jax_tree_map_exists(self):
        """Test that jax.tree.map API exists and works."""
        import jax
        import jax.numpy as jnp

        # Test basic tree.map functionality
        tree = {"a": jnp.array([1, 2, 3]), "b": jnp.array([4, 5, 6])}
        result = jax.tree.map(lambda x: x * 2, tree)

        self.assertEqual(result["a"].tolist(), [2, 4, 6])
        self.assertEqual(result["b"].tolist(), [8, 10, 12])

    def test_jax_tree_map_with_tuple(self):
        """Test jax.tree.map with tuple (as used in model.py line 41)."""
        import jax
        import jax.numpy as jnp

        x = jnp.array([[1, 2, 3], [4, 5, 6]])
        mask = jnp.array([[True, False, True], [False, True, False]])

        # This mimics the usage in TokenEncoder.__call__
        x_flipped, mask_flipped = jax.tree.map(lambda arr: jnp.flip(arr, axis=1), (x, mask))

        self.assertEqual(x_flipped.tolist(), [[3, 2, 1], [6, 5, 4]])
        self.assertEqual(mask_flipped.tolist(), [[True, False, True], [False, True, False]])

    def test_jax_tree_map_multiple_trees(self):
        """Test jax.tree.map with multiple trees (replaced jax.tree_multimap)."""
        import jax
        import jax.numpy as jnp

        # This mimics the zoneout_decoder usage in AcousticModel.__call__
        mask = {"h": jnp.array([0.1, 0.2]), "c": jnp.array([0.3, 0.4])}
        prev_state = {"h": jnp.array([1.0, 2.0]), "c": jnp.array([3.0, 4.0])}
        state = {"h": jnp.array([5.0, 6.0]), "c": jnp.array([7.0, 8.0])}

        # jax.tree.map now supports multiple trees (like the old tree_multimap)
        result = jax.tree.map(
            lambda m, s1, s2: s1 * m + s2 * (1 - m),
            mask, prev_state, state
        )

        # Verify the zoneout calculation
        expected_h = prev_state["h"] * mask["h"] + state["h"] * (1 - mask["h"])
        expected_c = prev_state["c"] * mask["c"] + state["c"] * (1 - mask["c"])

        self.assertTrue(jnp.allclose(result["h"], expected_h))
        self.assertTrue(jnp.allclose(result["c"], expected_c))


class TestTTSModuleImport(unittest.TestCase):
    """Test that the tts module's model code uses correct JAX APIs."""

    def test_no_deprecated_jax_tree_map_in_source(self):
        """Test that source code doesn't use deprecated jax.tree_map."""
        from pathlib import Path

        model_path = Path(__file__).parent.parent.parent.parent / \
            "underthesea/pipeline/tts/viettts_/nat/model.py"
        trainer_path = Path(__file__).parent.parent.parent.parent / \
            "underthesea/pipeline/tts/viettts_/nat/acoustic_tpu_trainer.py"

        for filepath in [model_path, trainer_path]:
            content = filepath.read_text()
            self.assertNotIn("jax.tree_map", content,
                f"Found deprecated jax.tree_map in {filepath.name}")
            self.assertNotIn("jax.tree_multimap", content,
                f"Found deprecated jax.tree_multimap in {filepath.name}")
            # Verify correct API is used
            self.assertIn("jax.tree.map", content,
                f"Expected jax.tree.map in {filepath.name}")


class TestLexiconEncoding(unittest.TestCase):
    """Test that lexicon file is read with UTF-8 encoding.

    See: https://github.com/undertheseanlp/underthesea/issues/727
    """

    def setUp(self):
        self.lexicon_path = Path(__file__).parent.parent.parent.parent / \
            "underthesea/pipeline/tts/viettts_/assets/infore/lexicon.txt"

    def test_lexicon_file_exists(self):
        """Test that the lexicon file exists."""
        self.assertTrue(self.lexicon_path.exists(),
            f"Lexicon file not found at {self.lexicon_path}")

    def test_lexicon_cannot_be_read_with_cp1252(self):
        """Test that lexicon file contains chars that fail with cp1252 (Windows default)."""
        with open(self.lexicon_path, "rb") as f:
            content = f.read()

        with self.assertRaises(UnicodeDecodeError):
            content.decode("cp1252")

    def test_lexicon_can_be_read_with_utf8(self):
        """Test that lexicon file can be read with UTF-8 encoding."""
        with open(self.lexicon_path, encoding="utf-8") as f:
            content = f.read()

        self.assertGreater(len(content), 0)

    def test_load_lexicon_uses_utf8(self):
        """Test that load_lexicon function uses UTF-8 encoding."""
        try:
            from underthesea.pipeline.tts.viettts_.nat.text2mel import load_lexicon
        except ImportError:
            self.skipTest("TTS dependencies not installed (requires underthesea[voice])")

        # This should not raise UnicodeDecodeError on any platform
        lexicon = load_lexicon(self.lexicon_path)

        self.assertIsInstance(lexicon, dict)
        self.assertGreater(len(lexicon), 0)

    def test_source_code_specifies_utf8_encoding(self):
        """Test that text2mel.py specifies UTF-8 encoding when reading files."""
        text2mel_path = Path(__file__).parent.parent.parent.parent / \
            "underthesea/pipeline/tts/viettts_/nat/text2mel.py"

        content = text2mel_path.read_text()

        # Check that open() calls for text files specify encoding
        self.assertIn('encoding="utf-8"', content,
            "text2mel.py should specify UTF-8 encoding when reading text files")


if __name__ == "__main__":
    unittest.main()
