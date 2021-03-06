#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:49:38 2021

@author: Kirill Varchenko
"""

import re
from itertools import product
from collections import namedtuple, defaultdict
import iuliia
import pickle


eng2rus_dict = {"a": ["а"], "b": ["б"], "ch": ["ч"], "d": ["д"], "e": ["е", "э"], 
                "f": ["ф"], "g": ["г"], "i": ["и"], "k": ["к"], "kh": ["х"], 
                "l": ["л"], "m": ["м"], "n": ["н"], "o": ["о"], "p": ["п"], 
                "r": ["р"], "s": ["с"], "sh": ["ш"], "shch": ["щ", "шч"], 
                "t": ["т"], "ts": ["ц", "тс", "тьс"], "u": ["у"], "v": ["в"], 
                "y": ["й", "ы", "ый", "ий"], 
                "ya": ["я", "ья", "ъя", "йа", "ьа"], 
                "ye": ["е", "ье", "йе", "ъе", "ые", "ьэ"], 
                "yi": ["ьи", "ыи", "йи", "ъи"], 
                "yo": ["ё", "ьё", "ъё", "йо", "ьо", "ыо"], 
                "yu": ["ю", "ью", "ъю", "йу", "ьу", "ыу"], 
                "z": ["з"], "zh": ["ж"], 
                "ε": ["", "ь"]}

PositionalVariant = namedtuple('PositionalVariant', 'eng emits after before')

translate = lambda source: iuliia.translate(source, schema=iuliia.WIKIPEDIA)

class BackTransliterator:
    def __init__(self):
        """
        Back transliterates from English to Russian and determines (if fit)
        probabilities of each transliteration variant based on frequencies.
        If probabilities are not set uniform 1/n probability is used for each
        of n returned transliteration variant.
        During fitting for each subsequence e1-e2-e3 probability of emitting
        a r2 by e2 is estimated given e1-e3.

        Returns
        -------
        None.

        """

        self.eng2rus_dict = eng2rus_dict
        self.single_elements = set(k for k in self.eng2rus_dict.keys() if len(self.eng2rus_dict[k]) == 1)
        self.eng_split = re.compile(r'shch|[cskz]h|y[aoeui]|y|[aoeiu]|ts(?!h)|[skz](?!h)|[ngfvprldmtb]', 
                                    re.IGNORECASE)
        
        # Since after Rus->Eng transliteration ь (soft-sign) between consonants
        # disappiar, it has to be inserted back during Eng->Rus back process.
        # a & b determine between which pairs of consonants ь is possible
        # c determines after which consonant ь is possible at the end of word
        # ε will be inserted in all possible positions
        a = {'d','z','l','m','n','r','s','t','ch'}
        b = {'b','g','k','l','m','s','v','d','zh','z','n','p','r','t','f','kh','ts','ch','sh','shch'}
        self.ab = set(product(a, b))
        self.c = {'b','v','d','zh','z','l','m','n','p','r','s','t','f','ch','sh','shch'}
        
        # Probabilities
        self.probs = None
    
    def _list_all(self, word):
        """
        Lists all possible back transliterations as well as splitted form.

        Parameters
        ----------
        word : str
            Transliterated word.

        Returns
        -------
        splitted : list of str
            Splitted form of word.
        res : list of lists of str
            All possible transliterations
        """
        splitted = self.eng_split.findall(word)
        
        # Find possible positions for ε and insert
        possible_eps = []
        for i in range(1, len(splitted)):
            if (splitted[i-1], splitted[i]) in self.ab:
                possible_eps.append(i)
        if splitted[-1] in self.c:
            possible_eps.append(len(splitted))
        for shift, pos in enumerate(possible_eps):
            splitted.insert(pos + shift, 'ε')

        vowels = 'ёуеыаоэяию'
        consonants = 'цкнгшщзхфвпрлджчсмтб'

        res = []
        L = len(splitted)
        for variant in product(*[self.eng2rus_dict[i] for i in splitted]):
            for i, c in enumerate(variant):
                # ъ/ь cannot be in the beginning
                if i == 0 and c[0] in 'ъь':
                    break
                
                # ъ/ь/ы cannot be after a vowel
                if i > 0 and c != '' and c[0] in 'ьъы' and variant[i-1][-1] in vowels:
                    break
                
                # й cannot be after a consonant
                if i > 0 and c == 'й' and variant[i-1][-1] in consonants:
                    break
                
                # йо, йя... cannot be after a consonant in the last pos
                if i == L - 1 and c != '' and c[0] == 'й' and variant[i-1][-1] in consonants:
                    break
                
                # й cannot be in the beginning
                if i == 0 and c == 'й':
                    break
                
                # y -> ий/ый can be only in the end of the world
                if (c == 'ий' or c == 'ый') and i != L - 1:
                    break
                
                # ye -> е cannot be between two consonants
                if c == 'е' and splitted[i] == 'ye' and i > 0 and i < L-1 and \
                    (variant[i-1][-1] in consonants and variant[i+1][0] in consonants):
                    break
                
                # ye -> cannot be after a consonant before the end
                if i == L - 1 and c == 'е' and splitted[i] == 'ye' and variant[i-1][-1] in consonants:
                    break
            else:
                res.append(variant)
        return splitted, res

    def predict_proba(self, word):
        """
        Gets predicted back transliterations with probabilities of each.

        Parameters
        ----------
        word : str
            Transliterated word.

        Returns
        -------
        res : list of pairs (float, str)
            All possible back transliterations with probabilities sorted from
            the most probable to the least one.

        """
        splitted, all_possibilities = self._list_all(word)
        res = []
        L = len(splitted)
        for possibility in all_possibilities:
            restored = ''.join(possibility)
            if self.probs is None:
                prob = 1/L
            else:
                prob = 1
                for i, (e, r) in enumerate(zip(splitted, possibility)):
                    # ^ and $ are symbols for begin and end of the word
                    pv = PositionalVariant(eng=e, emits=r, 
                                           after=splitted[i-1] if i > 0 else '^', 
                                           before=splitted[i+1] if i < L-1 else '$')
                    prob *= self._probability(pv)
            
            if prob > 0:
                # list only varianst with non-zero probability
                res.append((prob, restored))
        
        res.sort(reverse=True)
        return res
    
    def predict(self, word):
        """
        Returns all possible back transliterations.

        Parameters
        ----------
        word : str
            Transliterated word.

        Returns
        -------
        list of str
            List of all possible back transliterations sorted by probabilities.

        """
        pp = self.predict_proba(word)
        return [p[1] for p in pp]
        
    def fit(self, words):
        """
        Learns probabilies of emissions from dictionary.

        Parameters
        ----------
        words : iterable of str
            Dictionary of words to learn probabilities from. Please take care
            to remove non-alph symbols and lower case before use.

        Returns
        -------
        None.

        """
        self.probs = defaultdict(int)
        count = defaultdict(int)
        
        # For all word count possible combinations of english terms e1-e2-e3
        # and possible emission variants for e2 depending on adjacent e1-e3
        for word in words:
            translated = translate(word)
            splitted, all_possibilities = self._list_all(translated)
            L = len(splitted)
            for possibility in all_possibilities:
                restored = ''.join(possibility)
                if restored == word:
                    for i, (e, r) in enumerate(zip(splitted, possibility)):
                        if e in self.single_elements:
                            continue
                        pv = PositionalVariant(eng=e, emits=r, 
                                               after=splitted[i-1] if i > 0 else '^', 
                                               before=splitted[i+1] if i < L-1 else '$')
                        self.probs[pv] += 1
                        count[(pv.after, pv.eng, pv.before)] += 1
                    break

        # Normalization
        for k, v in self.probs.items():
            pv = PositionalVariant(*k)
            self.probs[k] = v / count[(pv.after, pv.eng, pv.before)]
                
    def _probability(self, pv):
        """
        Returns probability of a particular positional variant.

        Parameters
        ----------
        pv : tuple (eng emits after before)
            tuple describing a positional variant.

        Returns
        -------
        float
            1 for elements that have only one possible emission.
            probability depending on given positional variant.
            0 if given positional variant was not found in pretrained probabilities.
        """
        if pv.eng in self.single_elements:
            return 1
        
        return self.probs.get(pv, 0)


    def save_probs(self, name):
        """
        Save trained probabilities

        Parameters
        ----------
        name : str
            Name for model.

        Returns
        -------
        None.

        """
        with open(f'{name}.pickle', 'wb') as fo:
            pickle.dump(self.probs, fo)
    
    def load_probs(self, name):
        """
        Load trained probabilities

        Parameters
        ----------
        name : str
            Name of model.

        Returns
        -------
        None.

        """
        with open(f'{name}.pickle', 'rb') as fi:
            self.probs = pickle.load(fi)
        

data = set()
with open('/home/kirill/sources/python/wordpaths/lop_list.txt', 'r') as fi:
    for line in fi:
        l = line.strip().lower()
        dirty_symbols = set(l) - set('ёйцукенгшщзхъфывапролджэячсмитьбю')
        if len(dirty_symbols) == 0:
            data.add(l)

bt = BackTransliterator()
bt.load_probs('lop')

m, n = 0, 0
k = 0
for word in data:
    translated = translate(word)
    x = bt.predict(translated)
    if x[0] == word:
        m += 1
    elif x[1] == word:
        k += 1
    n += 1

# [\d()-]