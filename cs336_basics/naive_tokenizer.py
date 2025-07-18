import os
from typing import Dict, Iterable, Iterator, List, Tuple
import regex as re
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(text: str) -> Iterator[str]:
    return re.finditer(PAT, text)

def create_pretoken_count_dict(pretokens: Iterable[...]) -> Dict[Tuple[bytes], int]: # type: ignore
    tuples = {}
    for pretoken in pretokens:
        pretoken = pretoken.group()
        blist = [bytes([byte]) for byte in pretoken.encode()]
        t = tuple(blist)
        if t not in tuples:
            tuples[t] = 1
        else:
            tuples[t] += 1

    return tuples


def count_pairs(tuples_cnt_dict: Dict[Tuple, int]) -> Dict[Tuple, int]:
    pairs = {}

    for tuple in tuples_cnt_dict:
        for d in zip(tuple, tuple[1:]):
            if d not in pairs:
                pairs[d] = tuples_cnt_dict[tuple]
            else:
                pairs[d] += tuples_cnt_dict[tuple]

    return pairs

def get_maximal_token_pair(tuples: Dict[Tuple, int]) -> Dict[Tuple, int]:
    pairs = count_pairs(tuples)

    m = max([v for _, v in pairs.items()])
    maximal_token_pair = max([k for k,v in pairs.items() if v == m])

    return maximal_token_pair

def _merge_tokens_in_tuple(t: Tuple[bytes], token_pair: Tuple[bytes]) -> Tuple[bytes]:
    new_token = (b'').join(token_pair)

    new_t = []

    pos = 0
    while pos <= len(t)-1:
        if pos == (len(t)-1): # no pair to be taken, we are at the end
            new_t.append(t[pos])
            break

        if (t[pos], t[pos+1]) == token_pair:
            new_t.append(new_token)
            pos+=2

        else:
            new_t.append(t[pos])
            pos+=1

    return tuple(new_t)

def merge_tokens_in_count_dict(tuples: Dict[Tuple[bytes], int], token_pair: Tuple[bytes]):
    return {_merge_tokens_in_tuple(t, token_pair) : count for t,count in tuples.items() }

def merge_count_dicts(dicts: Iterable[Dict[Tuple[bytes], int]]):
    final_dict = {}
    for dict in dicts:
        for key, value in dict.items():
            if key in final_dict:
                final_dict[key] += value
            else:
                final_dict[key] = value

    return final_dict

def naive_tokenizer(
    text: str,
    vocab_size: int,
    special_tokens: list[str],
    use_tqdm: bool = False,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab = [bytes([i]) for i in range(256)] + [token.encode() for token in special_tokens]

    n_steps = vocab_size - len(vocab)

    merges = []

    if special_tokens:
        special_tokens = [re.escape(token) for token in special_tokens]
        pattern = "|".join(special_tokens)

        chunks = re.split(pattern, text)
    else:
        chunks = [text]

    chunks_pretokenized_count_dict = []
    for chunk in chunks:
        pretokens_iterator = pretokenize(chunk)
        count_dict = create_pretoken_count_dict(pretokens_iterator)
        chunks_pretokenized_count_dict.append(count_dict)

    count_dict = merge_count_dicts(chunks_pretokenized_count_dict)

    for _ in tqdm(range(n_steps), disable = not use_tqdm):
        pairs = count_pairs(count_dict)
        maximal_token_pair = get_maximal_token_pair(pairs)

        merges.append(maximal_token_pair)
        vocab.append((b'').join(maximal_token_pair))

        count_dict = merge_tokens_in_count_dict(count_dict, maximal_token_pair)

    vocab = {i: token for i, token in enumerate(vocab)}

    return vocab, merges


if __name__ == '__main__':
    
    text = '''low low low low low lower lower<|special|> widest widest widest newest newest<|special|> newest newest newest newest'''
    a, b = (naive_tokenizer(text, 269, ["<|special|>"]))
    print(a)
    print(b)
    # pretokens=re.findall(PAT, text)

    # pretokens = [pretoken.strip() for pretoken in pretokens]
    # tokens = ['<|endoftext|>'.encode()] + [chr(i) for i in range(256)]

    # # inicialize tuples as pretokens tokenized to byte level
    # tuples = {}
    # add_pretokens(pretokens, tuples)

    # for _ in range(6):
    #     pairs = count_pairs(tuples)
    #     max_token_pair = get_maximal_token_pair(tuples)
    #     tokens.append((b'').join(max_token_pair))

    #     tuples = merge_tokens(tuples, max_token_pair)
    
    # print(tokens)
