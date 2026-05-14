create_dataset.py 使用方式：
1、merged_dataset.jsonl放到指定目录下（建议跟create_dataset.py同目录）
2、pip install matplotlib scipy numpy langid tqdm
3、python create_dataset.py --input_len 【输入token数】 --variance_scale 【方差（0~inf，默认为0无随机，1为标准正态分布）】 --max_lines 【产生数据条数】--min_length 【最小token长度】--max_length 【最大token长度】
                                            --distribution 【'{"input长度1": 比例1, "input长度2": 比例2, ...}' (比例总和1.0)，注意前后加单引号(')】（设置distribution参数后，自动忽略-input_len和--variance_scale）
                                            --input_filename 【输入数据集路径】 --output_filename 【输出数据集路径】
示例：python create_dataset.py --input_len 512 --variance_scale 1 --max_lines 100 --min_length 64 --max_length 1024 --input_filename "/data/merged_dataset.jsonl" --output_filename "/data/test_dataset.jsonl" （生成100条均值512，标准正态分布的数据，最小长度64，最大长度1024，指定输入输出路径）
          python create_dataset.py --input_len 1024 --variance_scale 0 --max_lines 300 （生成300条长度最接近1024的数据，其他默认：最小最大长度不限，输入路径"./merged_dataset.jsonl"，输出路径"./test_dataset.jsonl"）
          python create_dataset.py --max_lines 500 --min_length 512 --max_length 3584 --distribution '{"1024": 0.3, "2048": 0.7}' （生成500条数据，500*30%条长度1024, 500*70%条长度2048，最小长度512，最大长度3584）
          python create_dataset.py --input_len 512 --variance_scale 1 --max_lines 500 --distribution '{"1024": 0.3, "2048": 0.7}' （生成500条数据，500*30%条长度1024, 500*70%条长度2048，最小最大长度不限，自动忽略-input_len和--variance_scale）

data_augment.py 使用方式：
1、确保DeepSeekR1文件夹跟data_augment.py同目录
2、pip install matplotlib scipy numpy langid tqdm jionlp transformers
3、python data_augment.py --aug_scale 【数据扩充倍数】--min_length 【最小token长度】--max_length 【最大token长度】--input_filename 【输入数据集路径】 --output_filename 【输出数据集路径】
示例：python data_augment.py  --aug_scale 10 --input_filename "./test_dataset.jsonl" --output_filename "./test_dataset_aug.jsonl" （数据扩充10倍）
          python data_augment.py  --aug_scale 10 --min_length 512 --max_length 3584 （数据扩充10倍，最小长度512，最大长度3584，其他默认：输入路径"./test_dataset.jsonl"，输出路径"./test_dataset_aug.jsonl"）

shuffle.py 使用方式：
1、python shuffle.py  --input_filename 【输入数据集路径】 --output_filename 【输出数据集路径】
示例：python shuffle.py  --input_filename "./test_dataset.jsonl" --output_filename "./test_dataset_shuffle.jsonl"

data_dist.py 使用方式：
1、pip install matplotlib scipy numpy langid
2、python data_dist.py --input_filename 【输入数据集路径】 --output_filename 【输出分布图路径】
示例：python data_dist.py  --input_filename "./test_dataset.jsonl" --output_filename "./test_dataset.png"

更新记录：
v1.3
data_augment.py 性能优化，新增设置最小最大token长度，自动将数据随机洗牌保存，输出输出token长度分布与语种分布
data_dist.py 能力增强，自动输出token长度分布与语种分布

v1.2
reate_dataset.py 新增设置最小最大token长度
自带DeepSeekR1分词器，无需联网下载
新增统计数据分布能力，见data_dist.py
数据扩充能力增强，可设置扩充倍数

v1.1
新增数据扩充能力，通过邻近汉字换位/同音词替换/随机增删字符扩充数据集，见data_augment.py
新增数据集随机洗牌能力，见shuffle.py
create_dataset.py 完成数据筛选后，自动将数据随机洗牌保存

v1.0
初始版本