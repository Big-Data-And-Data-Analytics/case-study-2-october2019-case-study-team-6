import time

from googletrans import Translator
from langdetect import detect
from scripts.mongoConnection import getCollection, insertCollection


class Translation:
    """Translation class is used for translating text in different languages from a dataframe to english
    """

    def __init__(self):
        """Initializes object of Translator function from googletrans
        """

        self.translator = Translator()

    def detect_translate(self, df):
        """detect_translate function detects the language of each row from onlyText column of a dataframe,
        if the language is not english then translates that text into english and finally insert the dataframe
        into the database

        :param1 df: dataframe on which detect_translate is to be applied
        :type1 df: Pandas Dataframe

        :return df: Translated dataframe
        :rtype df: Pandas Dataframe
        
        """
        ##TODO check :param1, return too
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
        return df

if __name__ == '__main__':

    translate = Translation()

    df = getCollection('05_NationalIdentity', 'ni_post')
    df = translate.detect_translate(df)
    insertCollection('06_NationalIdentity_Translated', 'ni_post_translated', df)

    df = getCollection('05_NationalIdentity', 'ni_comment')
    df = translate.detect_translate(df)
    insertCollection('06_NationalIdentity_Translated', 'ni_comment_translated', df)

    df = getCollection('05_NationalIdentity', 'ni_subcomment')
    df = translate.detect_translate(df)
    insertCollection('06_NationalIdentity_Translated', 'ni_subcomment_translated', df)



