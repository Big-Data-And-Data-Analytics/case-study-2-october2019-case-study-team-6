import subprocess
import sys

# Intsall package function
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Install packages
install_package("numpy")
install_package("sklearn")
install_package("imblearn")
install_package("pandas")
install_package("scipy")
install_package("pymongo")
install_package("nltk")
install_package("matplotlib")
install_package("langid")
install_package("emojis")
install_package("langdetect")
install_package("googletrans")
