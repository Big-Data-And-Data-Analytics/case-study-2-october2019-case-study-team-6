import numpy as np
from scripts.mongoConnection import getCollection, insertCollection

class Cleaning:
    """Cleaning class provides functions for cleaning the data.

        :func:
            - **For the best results, apply the functions as follows:**
                1 :func:`removeWhiteSpaces()`
                2 :func:`changeEmptyToNA()`
                3 :func:`removeDuplicates()`
            
                If the order changes, then duplicates might still be in the data

        :Example:

            from pymongo import MongoClient
            import pandas as pd
            import numpy as np

            client = MongoClient('localhost', 27017)
            db = client['01_NationalIdentity_Crawled']
            data = db.youTube_Video_Comments_Raw
            data = data.find({})
            data = list(data)
            data = pd.DataFrame(data)

            cleaner = Cleaning()

            data = cleaner.removeDuplicates(data)
            data = cleaner.removeWhiteSpaces(data, "textOriginal")
            data = cleaner.changeEmptyToNA(data, "textOriginal")
        
    """

    version = "0.1"

    def __init__(self, new_data=None):
            self.data = new_data

        # db = client['01_NationalIdentity_Crawled']
        # data = db.youTube_Video_Comments_Raw

    def remove_white_spaces(self, column): # Remove whitepsaces in the given data-column
        """Removes all whitespaces before and after the text and multi whitespaces inside the text

        :param data: which dataframe should be used
        :type data: pandas dataframe
        :param column: the name of the column
        :type column: string
        :return: dataframe with removed whitespaces in the specified column
        :rtype: pandas dataframe
        """        
        
        self.data[column] = self.data[column].str.strip()
        self.data[column] = self.data[column].str.replace("  +", " ")
        print("Whitespaces removed!")
        return self.data

    def change_empty_tona(self, column): # Change empty strings to NA in the given data-column
        """Changes all empty strings ("") to NA

        :param data: which dataframe should be used
        :type data: pandas dataframe
        :param column: the name of the column
        :type column: string
        :return: dataframe with NA instead of empty strings in the specified column
        :rtype: pandas dataframe
        """

        self.data[column] = self.data[column].replace(r'^\s*$', np.nan, regex=True)
        print("Empty strings changed!")
        return self.data

    def remove_duplicates(self): # Remove duplicates
        """Removes duplicated rows

        :param data: which dataframe should be used
        :type data: pandas dataframe
        :return: dataframe without duplicates
        :rtype: pandas dataframe
        """

        self.data = self.data.drop(columns="_id")
        self.data = self.data.drop_duplicates()
        print("Duplicates removed!")
        return self.data


if __name__ == "__main__":

    data = getCollection(db="01_NationalIdentity_Crawled", col="youTube_Video_Comments_Raw")

    cleaner = Cleaning(data)

    data = cleaner.remove_duplicates()
    data = cleaner.remove_white_spaces("textOriginal")
    data = cleaner.change_empty_tona("textOriginal")

    insertCollection('01_NationalIdentity_Crawled', 'cleaned_data', data)
