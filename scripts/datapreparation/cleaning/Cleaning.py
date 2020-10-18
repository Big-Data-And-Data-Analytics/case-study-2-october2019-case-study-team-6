class Cleaning:
    """This class provides functions for cleaning the data.

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

        .. note::
            - **For the best results, apply the functions as follows:**
                1 :func:`removeWhiteSpaces()`
                2 :func:`changeEmptyToNA()`
                3 :func:`removeDuplicates()`
            
                If the order changes, then duplicates might still be in the data
        
    """

    version = "0.1"

    def __init__(self):
        pass

    def removeWhiteSpaces(self, data, column): # Remove whitepsaces in the given data-column
        """
        Removes all whitespaces before and after the text and multi whitespaces inside the text

        :type data: pandas dataframe
        :param data: which dataframe should be used
        
        :type column: string
        :param column: the name of the column
        """

        data[column] = data[column].str.strip()
        data[column] = data[column].str.replace("  +", " ")
        print("Whitespaces removed!")
        return data

    def changeEmptyToNA(self, data, column): # Change empty strings to NA in the given data-column
        """
        Changes all empty strings ("") to NA

        :type data: pandas dataframe
        :param data: which dataframe should be used
        
        :type column: string
        :param column: the name of the column
        """

        data[column] = data[column].replace(r'^\s*$', np.nan, regex=True)
        print("Empty strings chaned!")
        return data

    def removeDuplicates(self, data): # Remove duplicates
        """
        Removes duplicated rows

        :type data: pandas dataframe
        :param data: which dataframe should be used
        """

        data = data.drop(columns="_id")
        data = data.drop_duplicates()
        print("Duplicates removed!")
        return data


if __name__ == "__main__":
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
