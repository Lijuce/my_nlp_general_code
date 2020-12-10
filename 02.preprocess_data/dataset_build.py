
def build_corpus(split, data_dir):
    """读取数据"""
    assert split in ['train', 'dev', 'test']
    word_lists = []
    tag_lists = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != "\r\n":
                try:
                    word, tag = line.strip('\n').split()
                except Exception:
                    pass
                else:
                    word_list.append(word)
                    tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []
    return word_lists, tag_lists

def build_vocab(word_lists, tag_lists):
    """构建词典"""
    def build_map(lists):
        maps = {}
        for list_ in lists:
            for e in list_:
                if e not in maps:
                    maps[e] = len(maps)
        return maps
    word2id = build_map(word_lists)
    tag2id = build_map(tag_lists)
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    return word2id, tag2id

