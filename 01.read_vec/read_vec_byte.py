# 以字节方式读取词向量，高效且内存占用比例最低

import struct
import json
import time

def read_and_save():
    start = time.time()
    num = 0
    data = {}
    with open("en.300.multi.json","r") as f:
        line = f.readline().strip().split(" ")
        word_count,dim = int(line[0]),int(line[1])
        line = f.readline().strip()
        data = json.loads(line)
        if line:
            with open("vocal.vec.bin2","wb") as wf:
                wf.write(struct.pack('ii',word_count,dim))
                for k,v in data.items():
                    word = k.encode("utf-8")
                    word_len = len(word)
                    wf.write(struct.pack('i',word_len))
                    wf.write(word)
                    for vv in v:
                        wf.write(struct.pack("f",vv))
    print("Reading time: ", time.time()-start)
        

def read_bin():
    data = {}
    start = time.time()
    with open("/home/common/wzj/NLP通用代码/vocal.vec.bin2","rb") as f:
        record_struct = struct.Struct("ii")
        word_count, dim = struct.unpack("ii",f.read(record_struct.size))
        for i in range(word_count):
            record_struct = struct.Struct("i")
            word_len =  struct.unpack("i",f.read(record_struct.size))[0]
            # word = f.read(word_len).decode("utf-8")
            word = f.read(word_len)  # 可待到要使用的时候再进行解码
            record_struct = struct.Struct("f"*dim)
            vec = f.read(record_struct.size)
            # vec = struct.unpack("f"*dim,f.read(record_struct.size))
            data[word] = vec
    print("Reading time: ", time.time()-start)

# read_and_save()
read_bin()