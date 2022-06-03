"""
To do:
* Add str and repr methods.
* Add method to print dataframes:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
* Could we optomise the get_possible_answers method, by only updating based
    on the previous guess, rather than looping through all guesses every
    time?
* Add methods to collect statistics?
* Optomise, slower than I expected.
"""


from random import choice as random_choice
from enum import Enum, auto as enum_auto
from functools import reduce

import numpy as np
import pandas as pd

INPUT_FILEPATH = r"filestore/words.txt"
NUM_CHARS = 5

# Could we make this a class variable?
dict_words = np.loadtxt(INPUT_FILEPATH, dtype='U')

def np_and(arrays, N=len(dict_words)):
    return reduce(np.logical_and, arrays, np.full(N, True))

class CharacterInfo(Enum):
    WRONG = enum_auto()
    MISSPLACED = enum_auto()
    CORRECT = enum_auto()


class Wordle:
    @staticmethod
    def _check_valid_word(word):
        assert isinstance(word, str), "{} is not a string".format(word)
        assert (len(word) == NUM_CHARS), "{} has {} characters, not {}".format(word, len(word), NUM_CHARS)
        assert word in dict_words, "{} is not in dictionary".format(word)

    def __init__(self, answer=None):
        self.guesses = list()
        self.infos = list()
        self.solved = False
        self.num_guesses_to_solve = None

        if answer is None:
            self.answer = ''.join(random_choice(dict_words))
        else:
            self.answer = answer
        
        self._check_valid_word(self.answer)

    def _get_guess_info_pairs(self):
        for g, i in zip(self.guesses, self.infos):
            yield tuple(zip(g, i))

    def _get_guess_info(self, guess):
        char_infos = [CharacterInfo.WRONG,]*NUM_CHARS
        
        exact_chars = list()
        answer_chars = list(self.answer)

        for i, (c1, c2) in enumerate(zip(self.answer, guess)):
            if c1 == c2:
                exact_chars.append(c1)
                char_infos[i] = CharacterInfo.CORRECT

        for c in exact_chars:
            answer_chars.remove(c)

        for i, (c, info) in enumerate(zip(guess, char_infos)):
            if (info != CharacterInfo.CORRECT) and (c in answer_chars):
                answer_chars.remove(c)
                char_infos[i] = CharacterInfo.MISSPLACED
        
        return char_infos


    def make_guess(self, guess):
        self._check_valid_word(guess)
        self.guesses.append(guess)

        guess_info = self._get_guess_info(guess)
        self.infos.append(guess_info)

        if all(info == CharacterInfo.CORRECT for info in guess_info):
            if not self.solved:
                self.num_guesses_to_solve = len(self.guesses)

            self.solved = True


    def _get_num_repeated_chars(self, char, drop_correct_chars=False):
        # First check if any guess has a grey char. That tells us exactly.
        # Are generators really necessary here? Nah.
        guess_infos = self._get_guess_info_pairs()

        def is_wrong(char_1, info):
            return (char_1 == char) and (info == CharacterInfo.WRONG)

        def is_right_or_missplaced(char_1, info):
            if drop_correct_chars:
                info_bool = info == CharacterInfo.MISSPLACED
            else:
                info_bool = (
                    (info == CharacterInfo.MISSPLACED) or
                    (info == CharacterInfo.CORRECT)
                )
            return (char_1 == char) and info_bool

        grey_rows = (
            l for l in guess_infos
            if any(is_wrong(c, i) for c, i in l)
        )

        # I don't think it matters which row we get, so just pick the
        # first, if it exists.
        grey_row = next(grey_rows, None)

        if grey_row is not None:
            num_chars = sum(is_right_or_missplaced(c, i) for c, i in grey_row)
            return (True, num_chars)
        
        else:
            # Need to create generator again:
            guess_infos = self._get_guess_info_pairs()
            
            # Loop and find the maximum number of chars.
            num_chars = max(
                sum(is_right_or_missplaced(c, i) for c, i in l)
                for l in guess_infos
            )

            return (False, num_chars)

    @staticmethod
    def filter_on_num_chars(char, exact, min_num_chars):
        char_count = np.char.count(dict_words, char)
        if exact:
            return (char_count == min_num_chars)
        else:
            return (char_count >= min_num_chars)
        
    def get_possible_answers(self, drop_correct_chars=False):
        # Need to modularize and comment this better.
        guessed_chars = sorted(list(set(
            e for l in self.guesses for e in l
        )))

        # 1. Filter words based on the number of characters.
        repeated_char_nums = [
            self._get_num_repeated_chars(c, drop_correct_chars)
            for c in guessed_chars
        ]

        num_chars_arrays = (
            self.filter_on_num_chars(char, exact, min_num_chars)
            for char, (exact, min_num_chars)
            in zip(guessed_chars, repeated_char_nums)
        )

        num_chars_array = np_and(num_chars_arrays)

        # 2. Filter words based on exact characters. 
        # Note that if there are duplicate greens, then this will duplicate
        # effort.
        correct_chars = set(
            (i, char)
            for l in self._get_guess_info_pairs()
            for i, (char, info) in enumerate(l)
            if info == CharacterInfo.CORRECT
        )

        if drop_correct_chars:
            correct_chars_array = np_and(
                dict_words.view('<U1')[i::len(dict_words[0])] != char
                for i, char in correct_chars
            )
        else:
            correct_chars_array = np_and(
                dict_words.view('<U1')[i::len(dict_words[0])] == char
                for i, char in correct_chars
            )
            

        # 3. Filter words based on wrong characters.
        def is_char_num_zero(char):
            for c, (exact, min_num_chars) in zip(guessed_chars, repeated_char_nums):
                if all([char == c, exact, min_num_chars == 0]):
                    return True
            return False
                        

        wrong_chars = set(
            (i, char)
            for l in self._get_guess_info_pairs()
            for i, (char, info) in enumerate(l)
            if ((info == CharacterInfo.WRONG) or (info == CharacterInfo.MISSPLACED))
            and not is_char_num_zero(char)
        )

        wrong_chars_array = np_and(
            dict_words.view('<U1')[i::len(dict_words[0])] != char
            for i, char in wrong_chars
        )

        possible_answers_array = np_and([
            num_chars_array,
            correct_chars_array,
            wrong_chars_array
        ])

        prob_array = possible_answers_array.astype('float')
        prob_array /= np.sum(prob_array)

        return prob_array

    def make_greedy_guess(self, verbose=False):
        prob_array = self.get_possible_answers()
        if sum(prob_array > 0) > 2:
            small_prob_array = self.get_possible_answers(drop_correct_chars=True)
            if sum(small_prob_array > 0) > 0:
                prob_array = small_prob_array
        guess = np.random.choice(dict_words, p=prob_array)
        if verbose:
            print('Guessing: {}'.format(guess))
        self.make_guess(guess)


    def greedy_strategy(self, verbose=False):
        while not self.solved:
            self.make_greedy_guess(verbose)
