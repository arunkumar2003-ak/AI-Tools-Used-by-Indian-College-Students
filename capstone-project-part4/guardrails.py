import re


def has_pii(text):

    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'

    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'

    return bool(
        re.search(email_pattern, text)
        or re.search(phone_pattern, text)
    )


if __name__ == "__main__":

    test1 = "my email is arun@gmail.com"

    test2 = "Age is 45 and income is 60000"

    print("Test 1:", has_pii(test1))

    print("Test 2:", has_pii(test2))