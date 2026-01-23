# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class EnviT5Translator:
    """Vietnamese-English translator using VietAI/envit5-translation model."""

    def __init__(self):
        model_name = "VietAI/envit5-translation"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(self, text, source_lang='vi', target_lang='en'):
        """
        Translate text between Vietnamese and English.

        Parameters
        ----------
        text : str
            Text to translate
        source_lang : str
            Source language code ('vi' or 'en')
        target_lang : str
            Target language code ('en' or 'vi')

        Returns
        -------
        str
            Translated text
        """
        if not text or not text.strip():
            return ""

        # EnviT5 uses language prefix format
        input_text = f"{source_lang}: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip target language prefix from output (e.g., "en: Hello" -> "Hello")
        prefix = f"{target_lang}: "
        if result.startswith(prefix):
            result = result[len(prefix):]

        return result
