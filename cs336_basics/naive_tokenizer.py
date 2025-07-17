from typing import Dict, Tuple
import regex as re





def add_pretokens(pretokens, tuples):
    for pretoken in pretokens:
        blist = [bytes([byte]) for byte in pretoken.encode()]
        t = tuple(blist)
        if t not in tuples:
            tuples[t] = 1
        else:
            tuples[t] += 1


def count_pairs(tuples: Dict[Tuple, int]) -> Dict[Tuple, int]:
    pairs = {}

    for tuple in tuples:
        for d in zip(tuple, tuple[1:]):
            if d not in pairs:
                pairs[d] = tuples[tuple]
            else:
                pairs[d] += tuples[tuple]

    return pairs

def get_maximal_token_pair(tuples: Dict[Tuple, int]) -> Dict[Tuple, int]:
    pairs = count_pairs(tuples)

    m = max([v for _, v in pairs.items()])
    maximal_token_pair = max([k for k,v in pairs.items() if v == m])

    return maximal_token_pair

def merge_tuple_tokens(t: Tuple[bytes], token_pair: Tuple[bytes]) -> Tuple[bytes]:
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

def merge_tokens(tuples: Dict[Tuple[bytes], int], token_pair: Tuple[bytes]):
    return {merge_tuple_tokens(t, token_pair) : count for t,count in tuples.items() }



if __name__ == '__main__':
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    text = '''low low low low low lower lower widest widest widest newest newest newest newest newest newest'''
    pretokens=re.findall(PAT, text)

    pretokens = [pretoken.strip() for pretoken in pretokens]
    tokens = ['<|endoftext|>'.encode()] + [chr(i) for i in range(256)]

    # inicialize tuples as pretokens tokenized to byte level
    tuples = {}
    add_pretokens(pretokens, tuples)

    for _ in range(6):
        pairs = count_pairs(tuples)
        max_token_pair = get_maximal_token_pair(tuples)
        tokens.append((b'').join(max_token_pair))

        tuples = merge_tokens(tuples, max_token_pair)
    
    print(tokens)
