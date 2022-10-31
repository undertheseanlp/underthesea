# flake8: noqa
from __future__ import absolute_import
# Source code from https://github.com/kirbyj/vPhon
###########################################################################
#       vPhon.py
#       Copyright 2008-2020 James Kirby <j.kirby@ed.ac.uk>
#
#
#       vPhon is free software: you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation, either version 3 of the License, or
#       (at your option) any later version.
#
#       vPhon is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with vPhon.  If not, see <http://www.gnu.org/licenses/>.
#
###########################################################################

import sys, re, io, string, argparse
from os.path import join, dirname
from types import SimpleNamespace

from underthesea.utils.vietnamese_ipa import vietnamese_sort_key
from vphone_rules import *


def trans(word, dialect, chao, eight, nosuper, glottal, phonemic):
    ##
    # Setup
    ##

    eight_tones = eight_lower if nosuper else eight_super
    if dialect == 's':
        chao_tones = chao_s_lower if nosuper else chao_s_super
    elif dialect == 'c':
        chao_tones = chao_c_lower if nosuper else chao_c_super
    else:
        chao_tones = chao_n_lower if nosuper else chao_n_super

    # Set variable for surface phonetic representation of labiodorsal finals
    ld_nas = 'ŋ͡m'  # if nosuper else 'ŋᵐ'
    ld_plo = 'k͡p'  # if nosuper else 'kᵖ'

    # Set case for palatalized dorsal finals
    pal_nas = 'ɲ' if nosuper else 'ʲŋ'
    pal_plo = 'c' if nosuper else 'ʲk'

    # Set case for labiovelar glide
    lv_gli = 'w' if nosuper else 'ʷ'

    # Set case for aspirated coronal onset
    if nosuper: onsets['th'] = 'th'

    # Set case for most <qu> words
    if nosuper: onsets['qu'] = 'kw'

    ons = ''
    gli = ''
    nuc = ''
    cod = ''
    ton = ''
    oOffset = 0
    cOffset = 0
    length = len(word)

    if length > 0:
        if word[0:3] in onsets:  # if onset is 'ngh'
            ons = onsets[word[0:3]]
            oOffset = 3
        elif word[0:2] in onsets:  # if onset is 'nh', 'gh', etc
            ons = onsets[word[0:2]]
            oOffset = 2
        elif word[0] in onsets:  # if single onset
            ons = onsets[word[0]]
            oOffset = 1

        if word[length - 2:length] in codas:  # if two-character coda
            cod = codas[word[length - 2:length]]
            cOffset = 2
        elif word[length - 1] in codas:  # if one-character coda
            cod = codas[word[length - 1]]
            cOffset = 1

        nucl = word[oOffset:length - cOffset]

        ##
        # Onsets
        ##

        # Edge cases
        # Prepend glottal stop to onsetless syllables for now
        if oOffset == 0: ons = 'ʔ'

        # Deal with 'quy', 'quý'... (fails if tone misplaced)
        if word in qu:
            ons = qu[word][0]
            nuc = qu[word][-1]
            if len(qu[word]) > 2: gli = lv_gli

        # Deal with gi, giền and giêng
        if word[0:2] in gi:
            if word == 'giền':
                nucl = 'â'  # See Emeneau 1951: 30
            elif length == 2 or (length == 3 and word[2] in ['n', 'm']):
                nucl = 'i'
            elif nucl in nuclei and word[2] in ['ê', 'ế', 'ề', 'ể', 'ễ', 'ệ']:
                nucl = 'iê'
            ons = onsets['gi']

        ##
        # Vowels
        ##

        if nucl in nuclei:
            nuc = nuclei[nucl]

        elif nucl in onglides:  # if there is an onglide...
            nuc = onglides[nucl]  # modify the nuc accordingly
            if ons != 'ʔ':  # if there is a (non-glottal) onset...
                ons = ons + lv_gli  # labialize it, but...
            else:  # if there is no onset...
                ons = 'w'  # add a labiovelar onset

        elif nucl in onoffglides:
            if ons != 'ʔ':  # if there is a (non-glottal) onset...
                ons = ons + lv_gli  # labialize it, but...
            else:  # if there is no onset...
                ons = 'w'  # add a labiovelar onset
            nuc = onoffglides[nucl][0:-1]
            cod = onoffglides[nucl][-1]

        elif nucl in offglides:
            cod = offglides[nucl][-1]
            nuc = offglides[nucl][:-1]

        else:
            # Can already tell that something is non-Viet: bail out
            return (None, None, None, None, None)

        # Deal with other labialized onsets
        # if ons == 'kʷ': ons = 'k'; gli = lv_gli
        if len(ons) == 2 and ons[1] == lv_gli: ons = ons[0]; gli = lv_gli
        # if ons == 'tʰʷ': ons = 'tʰ'; gli = lv_gli
        if len(ons) == 3 and ons[2] == lv_gli: ons = ons[0:2]; gli = lv_gli

        ##
        # Tones
        ##

        # A1 needs to be added here or else it's impossible to
        # logically determine which character to care about
        tonelist = [tones[word[i]] for i in range(0, length) if word[i] in tones]
        if tonelist:
            ton = str(tonelist[len(tonelist) - 1])
        else:
            ton = 'A1'
        if (ton == 'B1' and cod in ['p', 't', 'c', 'k']): ton = 'D1'
        if (ton == 'B2' and cod in ['p', 't', 'c', 'k']): ton = 'D2'

        if eight:
            ton = eight_tones[ton]
        elif chao:
            ton = chao_tones[ton]
        else:
            if not nosuper: ton = gedney_super[ton]

        ##
        # Generate internal G2P representation
        ##

        # If flagged, delete predictable glottal onsets
        if glottal and ons == 'ʔ': ons = ''

        # Velar fronting
        if nuc == 'aː':
            if cod == 'c': nuc = 'ɛ'
            if cod == 'ɲ': nuc = 'ɛ'

        # Ignore some transforms if producing spelling pronunciation output

        if not dialect == 'o':

            # Capture vowel/coda interactions of ɛ/ɛː and e/eː
            if cod in ['ŋ', 'k']:
                if nuc == 'ɛ': nuc = 'ɛː'
                if nuc == 'e': nuc = 'eː'

        else:

            if word[0:2] in 'gi': ons = 'ʑ'
            if ons in ['j']: ons = 'z'

        ##
        # Northern
        ##

        # Transform internal G2P to UR
        if dialect in ['n', 'o']:

            # No surface palatal codas
            if cod in ['c', 'ɲ']:
                if cod == 'c': cod = 'k'
                if cod == 'ɲ': cod = 'ŋ'

        if dialect == 'n':

            # Onset mergers
            if ons in ['j', 'r']:
                ons = 'z'
            elif ons in ['c', 'ʈ']:
                ons = 'tɕ'
            elif ons == 'ʂ':
                ons = 's'

            # No surface palatal codas
            # Moved above
            # if cod in ['c', 'ɲ']:
            #    if cod == 'c': cod = 'k'
            #    if cod == 'ɲ': cod = 'ŋ'

            # Now, if generating SRs (default), apply rules
            if not phonemic:

                # Palatalized and labiodorsal codas
                if cod in ['k', 'ŋ']:
                    if nuc in ['e', 'ɛ', 'i']:
                        if cod == 'k': cod = pal_plo
                        if cod == 'ŋ': cod = pal_nas
                    elif nuc in ['u', 'ɔ', 'o'] and word != 'quốc':
                        if cod == 'k': cod = 'k͡p'
                        if cod == 'ŋ': cod = 'ŋ͡m'

                # Surface pre-palatal vowel centralization
                if cod in [pal_nas, pal_plo]:
                    if nuc == 'ɛ': nuc = 'a'

                # Lengthen surface monophthongs where there is no length contrast
                if len(nuc) == 1 and nuc not in ['a', 'ə']:
                    if len(cod) == 1 and nuc != 'ɨ':
                        nuc += 'ː'
                    elif len(cod) == 0:
                        nuc += 'ː'

            else:
                # Shorten long monophthongs in open syllables for UR
                if not cod and nuc in ['aː', 'əː']:
                    if nuc == 'aː': nuc = 'a'
                    if nuc == 'əː': nuc = 'ə'

        ##
        # Central/Southern
        ##

        elif dialect in ['s', 'c']:
            if ons == 'z': ons = 'j'
            if ons == 'k' and gli == lv_gli: ons = 'w'; gli = ''
            if ons == 'ɣ': ons = 'ɡ'

            # Hanoi diphthongs are long monophthongs
            if cod and nuc in ['iə', 'uə', 'ɨə']:
                if nuc == 'iə': nuc = 'iː'
                if nuc == 'ɨə': nuc = 'ɨː'
                if nuc == 'uə': nuc = 'uː'

            # Partial ɔ/o merger
            if nuc == 'ɔ' and cod in ['n', 't']: nuc = 'ɔː'
            if nuc == 'o' and cod in ['ŋ', 'k']: nuc = 'ɔ'

            if nuc == 'ɛ' and cod in ['n', 't']:
                if cod == 'n': cod = 'ŋ'
                if cod == 't': cod = 't'
                nuc = 'ɛː'

            # No coronals after long vowels
            if cod and len(nuc) == 2:
                if cod == 'n': cod = 'ŋ'
                if cod == 't': cod = 'k'

            # No coronals after central vowels
            if cod and nuc in ['ɨ', 'ə', 'a', 'u', 'o']:
                if cod == 'n': cod = 'ŋ'
                if cod == 't': cod = 'k'

            # No dorsals after short front vowels
            if cod and nuc in ['i', 'e', 'ɛ']:
                if cod == 'ŋ': cod = 'n'
                if cod == 'k': cod = 't'

            # All non-labial codas are dorsal or coronal
            if cod in ['ɲ', 'c']:
                if cod == 'ɲ': cod = 'n'
                if cod == 'c': cod = 't'

            # Now, if generating SRs (default), apply rules
            if not phonemic:

                # Surface <x> <s> merger
                if ons == 'ʂ': ons = 's'

                # Pre-coronal centralization
                if cod in ['n', 't']:
                    if nuc in ['i', 'e', 'ɛ']:
                        if nuc == 'i': nuc = 'ɨ'
                        if nuc == 'ɛ': nuc = 'a'
                        if nuc == 'e': nuc = 'əː'

                # Centralization of /u/ before labials
                if nuc == 'u' and cod in ['m', 'p']: nuc = 'ɨ'

                # No short surface /e ɛ o ɔ/ (except before labiodorsals)
                if nuc in ['e', 'ɛ', 'o', 'ɔ']:
                    if nuc == 'e': nuc = 'eː'
                    if nuc == 'ɛ': nuc = 'ɛː'
                    if not cod in ['ŋ', 'k']:
                        if nuc == 'o': nuc = 'oː'
                        if nuc == 'ɔ': nuc = 'ɔː'

                # Labiodorsals after [u ɔ oː]
                if nuc in ['u', 'ɔ', 'oː'] and cod in ['ŋ', 'k']:
                    if cod == 'ŋ': cod = 'ŋ͡m'
                    if cod == 'k': cod = 'k͡p'

        ##
        # Universal UR modifications
        ##

        # Shorten long monophthongs in open syllables for UR unless spelling pronunciation
        if not dialect == 'o':
            if phonemic and not cod and nuc in ['aː', 'əː']:
                if nuc == 'aː': nuc = 'a'
                if nuc == 'əː': nuc = 'ə'

        ##
        # All done
        ##

        return (ons, gli, nuc, cod, ton)


def convert(word, dialect, chao, eight, nosuper, glottal, phonemic, delimit):
    """Convert a single orthographic string to IPA."""

    ons = ''
    gli = ''
    nuc = ''
    cod = ''
    ton = ''
    seq = ''

    try:
        (ons, gli, nuc, cod, ton) = trans(word, dialect, chao, eight, nosuper, glottal, phonemic)
        if None in (ons, gli, nuc, cod, ton):
            seq = '[' + word + ']'
        else:
            seq = delimit + delimit.join(filter(None, (ons, gli, nuc, cod, ton))) + delimit
    except TypeError:
        pass

    return seq


def to_ipa(word):
    delimit = ""
    tokenize = False
    chao = True
    glottal = True
    nosuper = False
    phonemic = False
    eight = False
    output_ortho = False

    compound = ''
    ortho = ''
    dialect = "n"
    ortho += word
    word = word.strip(string.punctuation).lower()
    # if tokenize==true:
    # call this routine for each substring and re-concatenate
    if (tokenize and '-' in word) or (tokenize and '_' in word):
        substrings = re.split(r'(_|-)', word)
        values = substrings[::2]
        delimiters = substrings[1::2] + ['']
        ipa = [convert(x, dialect, chao, eight, nosuper, glottal, phonemic, delimit).strip() for x in
               values]
        seq = ''.join(v + d for v, d in zip(ipa, delimiters))
    else:
        seq = convert(word, dialect, chao, eight, nosuper, glottal, phonemic, delimit).strip()
    # concatenate

    compound = compound + seq

    # entire line has been parsed
    if ortho == '':
        pass
    else:
        ortho = ortho.strip()
        # print orthography?
        if output_ortho: print(ortho, output_ortho, sep='', end='')
        print(compound)
    return compound


Args = SimpleNamespace()


def main():
    chao = False
    delimit = ''
    dialect = 'n'
    nosuper = False
    glottal = False
    phonemic = False
    output_ortho = ''
    eight = False
    tokenize = False

    # Command line options
    parser = argparse.ArgumentParser(description="python vPhon.py")
    parser.add_argument("-d", "--dialect", choices=["n", "c", "s", "o"],
                        help="Specify dialect region (Northern, Central, Southern) or spelling pronunciation",
                        type=str.lower)
    parser.add_argument("-c", "--chao", action="store_true", help="Phonetize tones as Chao values")
    parser.add_argument("-g", "--glottal", action="store_true", help="No glottal stops in underlying forms")
    parser.add_argument("-8", "--eight", action="store_true", help="Encode tones as 1-8")
    parser.add_argument("-n", "--nosuper", action="store_true", help="No superscripts anywhere")
    parser.add_argument("-p", "--phonemic", action="store_true", help="Underlying transcriptions after Pham (2006)")
    parser.add_argument("-m", "--delimit", action="store", type=str,
                        help="produce delimited output (bi ca = .b.i.33. .k.a.33.)")
    parser.add_argument("-o", "--ortho", action="store", type=str, dest="output_ortho",
                        help="output orthography as well as IPA (quốc: wok⁴⁵)")
    parser.add_argument("-t", "--tokenize", action="store_true",
                        help="Preserve underscores or hyphens in tokenized inputs (anh_ta = anhᴬ¹_taᴬ¹)")
    args = parser.parse_args()

    wd = dirname(__file__)
    outputs_folder = join(dirname(wd), "outputs")
    with open(join(outputs_folder, "syllables.txt")) as f:
        lines = f.readlines()
        items = [line.strip() for line in lines]
        words = sorted(items, key=vietnamese_sort_key)
    results = ""
    i = 1
    for word in words:
        if word in set(["gĩữ", "by"]):
            continue
        ipa = to_ipa(word)
        results += f"{i},{word},{ipa}\n"
        i += 1
    with open(join(outputs_folder, "vphon_syllables_ipa.txt"), "w") as f:
        f.write(results)


if __name__ == '__main__':
    main()
