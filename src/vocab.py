class Vocab:
    def __init__(self):
        self._token_to_index = {"<s>": 0, "</s>": 1}
        self._index_to_token = ["<s>", "</s>"]

    def add(self, word):
        index = self._token_to_index.get(word, None)
        if index is None:
            index = len(self._index_to_token)
            self._index_to_token.append(word)
            self._token_to_index[word] = index
        return index

    def from_dataset(self, dataset):
        flatten_dataset = [word for sent in dataset for word in sent]
        dataset_unique = set(flatten_dataset)
        for word in dataset_unique:
            self.add(word)

    def encode_sentence(self, sent, pad_left=True, pad_right=True):
        sent = [self[word] for word in sent]
        if pad_left:
            sent = [0] + sent
        if pad_right:
            sent = sent + [1]
        return sent

    def all_tokens(self):
        return self._index_to_token

    def __len__(self):
        return len(self._index_to_token)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._index_to_token[key]
        else:
            return self._token_to_index[key]

    def __repr__(self):
        return " ".join(sorted(self._index_to_token))