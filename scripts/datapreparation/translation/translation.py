import time

from googletrans import Translator
from langdetect import detect
from mongoConnection import getCollection, insertCollection


class Translation:
    """Translation class is used for translating text in different languages from a dataframe to english
    """

    def __init__(self):
        """Initializes object of Translator function from googletrans
        """

        self.translator = Translator()

    def detect_translate(self, df, collection):
        """detect_translate function detects the language of each row from onlyText column of a dataframe,
        if the language is not english then translates that text into english and finally insert the dataframe
        into the database

        :param1 df: dataframe on which detect_translate is to be applied
        :param2 collection: collection in which dataframe with translated onlyText column is to be inserted

        :type1 df: Pandas Dataframe
        :type2 collection: String

        """
        for index, row in df['onlyText'].iteritems():
            try:
                lang = detect(row)
                if lang != 'en':
                    translated = self.translator.translate(row)
                    df.loc[index, 'onlyText'] = (translated.text)
                    time.sleep(0.2)
                df.loc[index, 'detectLang'] = lang
            except:
                pass
        insertCollection('06_NationalIdentity_Translated', collection, df)


if __name__ == '__main__':

    translate = Translation()

    df_post = getCollection('05_NationalIdentity', 'ni_post')
    translate.detect_language(df_post, 'ni_post_translated')

    df_comment = getCollection('05_NationalIdentity', 'ni_comment')
    translate.detect_language(df_comment, 'ni_comment_translated')

    df_subcomment = getCollection('05_NationalIdentity', 'ni_subcomment')
    translate.detect_language(df_subcomment, 'ni_subcomment_translated')



