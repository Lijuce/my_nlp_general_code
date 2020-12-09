import struct
import json
import time

def printf(line, file_name="en.300.json"):
  with open(file_name, "a") as f:
    f.writelines(line + '\n')

def read_and_save():
  data = {}
  num = 0
  with open("/home/common/wzj/NLP通用代码/01.read_vec/en.300.vec","r") as f:
    line = f.readline().strip().split(" ")
    word_count,dim = int(line[0]),int(line[1])  # 读取首行信息
    line = f.readline()
    while line:
      line = line.strip().split(' ')
      if len(line) < 2:
        line = f.readline()
        continue
      word = line[0]
      vec = [round(float(item), 3) for item in line[1:]]
      data[word] = vec
      line = f.readline()
      # num += 1
      # if num > 100:
      #   break

  # 构造json
  printf((str(word_count) + " " + str(dim)))
  for k,v in data.items():
    printf(json.dumps({k:v}))


def read_json():
  data = {}
  with open("en.300.json","r") as f:
    line = f.readline().strip().split(" ")
    word_count, dim = int(line[0]),int(line[1])
    line = f.readline()
    while line:
      line = line.strip()
      word_vec = json.loads(line)
      data.update(word_vec)
      line = f.readline()
  return data

def read_and_save_multijson():
  start = time.time()
  data = {}
  num = 0
  with open("/home/common/wzj/NLP通用代码/01.read_vec/en.300.vec","r") as f:
    line = f.readline().strip().split(" ")
    word_count, dim = int(line[0]),int(line[1])
    line = f.readline()
    while line:
      line = line.strip().split(" ")
      if len(line) < 2:
        line = f.readline()
        continue
      word = line[0]
      vec = [round(float(item), 3) for item in line[1:]]
      data[word] = vec
      line = f.readline()
      # num += 1
      # if num > 100:
      #   break
  # 生成多行json
  printf((str(word_count) + " " + str(dim)), file_name="en.300.multi.json")
  printf(json.dumps(data), file_name="en.300.multi.json")
  print("Reading time: ", time.time()-start)

def read_multi_json():
  start = time.time()
  data = {}
  with open("en.300.multi.json","r") as f:
    line = f.readline().strip().split(" ")
    word_count,dim = int(line[0]),int(line[1])
    line = f.readline().strip()
    data = json.loads(line)

  print("Reading time: ", time.time()-start)
# read_and_save()
read_and_save_multijson()
# d = read_json()
