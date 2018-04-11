try:
    from urllib.request import urlretrieve
except:
    from urllib import urlretrieve
from os.path import dirname, join, isfile

components = [
    {
        "name": "classification.fasttext.model",
        "url": "https://github.com/magizbox/underthesea.models/raw/master/classification/fasttext.model",
        "destination": ["classification", "fasttext.model"]
    }
]


def download_component(component_name):
    try:
        component = [component for component in components if
                     component["name"] == component_name][0]
        try:
            folder = dirname(dirname(__file__))
            file_name = join(folder, join(*component["destination"]))
            if isfile(file_name):
                print(
                "Component '{}' is already existed.".format(component["name"]))
            else:
                print("Start download component '{}'".format(component["name"]))
                print(file_name)
                download_file(component["url"], file_name)
                print("Finish download component '{}'".format(component["name"]))
        except Exception as e:
            print(e)
            print("Cannot download component '{}'".format(component["name"]))
    except:
        message = "Error: Component with name '{}' does not exist.".format(
            component_name)
        print(message)


def download_file(url, file_name):
    """ Cross platform file download helper function
    :param url: url of file
    :param file_name: destination file
    """
    urlretrieve(url, file_name)


if __name__ == '__main__':
    pass
