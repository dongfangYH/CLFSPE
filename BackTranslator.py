from transformers import MarianMTModel, MarianTokenizer
import torch
"""
example usage:

translator = BackTranslator()

# Example text to translate
texts = [
    '[Audit] Password expiration change is not included in the Audit',
    '[BE] User administration - ensure that user does not exist in Auth0 prior to creating it',
    '[FE][Page Config] Creating a single-state-manager page or copying a PWB page breaks the Select Station widget'
]

# Perform back translation
back_translated_texts = translator.back_translate(texts, language_src="en", language_dst="es")

print("Original texts:", texts)
print("Back translated texts:", back_translated_texts)

available lang:
fr,fr_BE,fr_CA,fr_FR,wa,frp,oc,ca,rm,lld,fur,lij,lmo,
es,es_AR,es_CL,es_CO,es_CR,es_DO,es_EC,es_ES,es_GT,
es_HN,es_MX,es_NI,es_PA,es_PE,es_PR,es_SV,es_UY,es_VE,
pt,pt_br,pt_BR,pt_PT,gl,lad,an,mwl,it,it_IT,co,nap,scn,vec,sc,ro,la

"""
class BackTranslator:
    def __init__(self, src_model_name='Helsinki-NLP/opus-mt-ROMANCE-en', tgt_model_name='Helsinki-NLP/opus-mt-en-ROMANCE'):
        # Load models and tokenizers
        self.src_tokenizer, self.src_model = self._download(src_model_name)
        self.tgt_tokenizer, self.tgt_model = self._download(tgt_model_name)

    def _download(self, model_name):
        """Helper function to download data for a language"""
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model

    def translate(self, texts, model, tokenizer, language):
        """Translate texts into a target language"""
        # Format the text as expected by the model
        formatter_fn = lambda txt: f"{txt}" if language == "en" else f">>{language}<< {txt}"
        original_texts = [formatter_fn(txt) for txt in texts]
        tokens = tokenizer(original_texts, return_tensors="pt", padding=True, truncation=True)

        # Translate
        with torch.no_grad(): 
            translated = model.generate(**tokens)

        # Decode (tokens to text)
        translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

        return translated_texts

    def back_translate(self, texts, language_src, language_dst):
        """Implements back translation"""
        # Translate from source to target language
        translated = self.translate(texts, self.tgt_model, self.tgt_tokenizer, language_dst)
        
        # Translate from target language back to source language
        back_translated = self.translate(translated, self.src_model, self.src_tokenizer, language_src)

        return back_translated