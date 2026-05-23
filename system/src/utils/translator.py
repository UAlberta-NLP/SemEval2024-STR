def translate_texts(text: str, src_lang, tgt_lang) -> dict:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    from google.cloud import translate_v2 as translate
    
    if len(text) == 0:
        return []
    
    translate_client = translate.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")
    
    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=tgt_lang, source_language=src_lang)

    result = [r["translatedText"].replace("&quot;", "\"").replace("&#39;", "'") for r in result]

    return result

