from typing import Any, Iterator,Optional
import regex as re
from tqdm import tqdm
#from pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class MyTokenizer:
    merges: list[tuple[bytes, bytes]]
    vocabulary: dict[int, tuple[bytes,...]]
    words_counts_dict = dict[tuple[bytes, ...], int]
    pairs_counts_dict = dict[tuple[bytes, ...], int]

    def train(self, text, special_tokens, vocab_size: int) -> tuple[dict[int, tuple[bytes,...]], list[tuple[bytes, bytes]]]:
        words_counts_dict, pairs_counts_dict = self.make_counts_dicts(text, special_tokens)
        
        vocab = [bytes([i]) for i in range(256)] + [token.encode() for token in special_tokens]
        merges = []

        n_steps = vocab_size - len(vocab)

        for i in range(n_steps):
            # do naive implementation first
            # merge the words, make it into new word counts and completely new pairs counts
            max_pair = self._get_maximal_token_pair(pairs_counts_dict)
            merges.append(max_pair)
            vocab.append(b''.join(max_pair))

            new_words_counts_dict: dict = {}
            for old_word, cnt in words_counts_dict.items():
                new_word = self._merge_tokens(old_word, max_pair)
                new_words_counts_dict[new_word] = cnt

            words_counts_dict = new_words_counts_dict
            pairs_counts_dict = self._make_pairs_count_dict(words_counts_dict)

        vocab_dict = {i: token for i, token in enumerate(vocab)}

        return vocab_dict, merges

    def make_counts_dicts(
            self, 
            text, 
            special_tokens = Optional[list[str]]
            ) -> tuple[dict[tuple[bytes,...], int], dict[tuple[bytes, bytes], int]]:
        splits: list[str] = self._split_on_special_tokens(text, special_tokens)
        
        words_counts_dict_list: list[dict] = []
        for sentence in splits:
            pretokens: Iterator = self._pretokenize(sentence)
            words_counts_dict_list.append(self._make_word_cnt_dict(pretokens))

        words_counts_dict = self._merge_counts_dicts(words_counts_dict_list)

        pairs_counts_dict = self._make_pairs_count_dict(words_counts_dict)

        return words_counts_dict, pairs_counts_dict

    def _split_on_special_tokens(self, text: str, special_tokens: list[str]) -> list[str]:
        special_tokens = [re.escape(token) for token in special_tokens]
        pattern = "|".join(special_tokens)
        chunks = re.split(pattern, text)

        return chunks
    
    def _pretokenize(self, text: str) -> Iterator[re.Match[str]]:
        return re.finditer(PAT, text)

    def _make_word_cnt_dict(self, words: Iterator[re.Match[str]]) -> dict[tuple[bytes,...], int]:
        words_counts_dict: dict[tuple[bytes,...], int] = {}
        for word_match in words:
            word = word_match.group()
            word_as_byte_list = [bytes([byte]) for byte in word.encode()]
            word_as_byte_tuple = tuple(word_as_byte_list)
            words_counts_dict[word_as_byte_tuple] = 1 + words_counts_dict.get(word_as_byte_tuple, 0)

        return words_counts_dict

    def _merge_counts_dicts(self, dicts: list[dict[Any, int]]) -> dict[Any, int]:
        merged_dict: dict[Any, int] = {}
        for cnt_dict in dicts:
            for k, v in cnt_dict.items():
                merged_dict[k] = v + merged_dict.get(k, 0)

        return merged_dict
    
    def _make_pairs_count_dict(self, words_count_dict: dict[tuple[bytes, ...], int]):
        '''creates the dictionary with count of pairs of tokens'''
        pairs_count_dict: dict[tuple[bytes, bytes], int] = {}
        for word, word_count in words_count_dict.items():
            for pair in zip(word, word[1:]):
                pairs_count_dict[pair] = word_count + pairs_count_dict.get(pair, 0)

        return pairs_count_dict
    
    @staticmethod
    def _get_maximal_token_pair(pairs_counts_dict: dict[tuple, int]) -> tuple[bytes, bytes]:
        maximal_token_pair, _ = max(pairs_counts_dict.items(), key=lambda item: (item[1], item[0]))

        return maximal_token_pair
    
    @staticmethod
    def _merge_tokens(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes,...]:
        new_token = b''.join(pair)
        new_word = []
        l = len(word)

        i=0
        while i < l-1:
            if (word[i] == pair[0]) and (word[i+1] == pair[1]):
                new_word.append(new_token)
                i+=2
            else:
                new_word.append(word[i])
                i+=1

        if i == l-1:
            new_word.append(word[i])

        return tuple(new_word)

