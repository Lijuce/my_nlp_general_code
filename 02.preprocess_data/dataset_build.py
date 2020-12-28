import json


def load_ndjson_to_array(path_to_file):
    # 读取json格式的数据：dict->array
    data = []
    try:
        with open(path_to_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        raise e
    print("Reading {}".format(path_to_file))
    return data

def read_lines(path_to_file, split_str):
    # 读取普通文本格式的数据：multi->array
    data = []
    try:
        with open(path_to_file, 'r') as f:
            for line in f:
                tmp = [x for x in line.strip().split(split_str)]  # x类型视情况而定
                data.append(tmp)
    except Exception as e:
        raise e
    print("Reading {}".format(path_to_file))
    return data

def strlist_to_list(strlist):
    # 字符串格式的列表->列表格式的列表
    list_after = json.loads(strlist)
    return list_after

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

def Output_file(file_to_path, output_data):
    # 字符串列表内容->文本文件
    for out_line in output_data:
        with open(file_to_path, 'a') as f:
            f.writelines(out_line + '\n')
    print("Successfully output the data to the path: {}".format(file_to_path))

