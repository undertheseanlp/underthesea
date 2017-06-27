# coding: utf-8

def get_utf8_number(character):
    """ get utf8 number of character

    :type character: unicode 
    """
    return character.encode("utf-8").encode("hex")


def get_unicode_number(character):
    """ get unicode number of character

    :type character: unicode 
    """
    return hex(ord(character))[2:].zfill(4).upper()


def analyze_characters(s):
    """ core function: analyze characters
    print utf8 number and unicode number of each characters in text

    :param unicode s: input string
    :type s: unicode 
    """
    print u"      unicode      utf8"
    for i in s:
        utf8_number = get_utf8_number(i)
        unicode_number = get_unicode_number(i)
        if utf8_number in ["cc80", "cc83", "cca3"]:
            format_string = u"{:3s} -> {:>7s} -> {:>6s}"
        else:
            format_string = u"{:2s} -> {:>7s} -> {:>6s}"
        print format_string.format(i, unicode_number, utf8_number)


if __name__ == '__main__':
    # unicode dựng sẵn
    s = u"cộng hòa xã hội"
    analyze_characters(s)

    print

    # unicode tổ hợp
    s = u"cộng hòa xã hội"
    analyze_characters(s)
