import getpass
import os

def get_password_edited(check=False):
    """
    Check if password is provided as ``PANSAT_PASSWORD`` environment variable.
    If this is not the case query user to input password.

    Params:
        check(``bool``): Whether user should insert the password twice
            to avoid spelling errors.

   
    """
    try:
        password = os.environ["PANSAT_PASSWORD"]
    except KeyError:
        print("Please enter your pansat user password:")
        password = getpass.getpass()

        f = open("word.txt", "w")
        f.write(password)
        f.close()

get_password_edited()



	
	
