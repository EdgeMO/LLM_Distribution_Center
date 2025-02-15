# 基于本地数据，生成算法适配的随机批输入, 并且提供数据查询接口，返回基于 task id 的具体数据

import os
import sys
import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import wordnet

# 加载必要的模型和资源
nlp = spacy.load("en_core_web_sm")
current_working_directory = os.getcwd()
sys.path.append(current_working_directory)
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from config.type import TaskType
class InputProcessor:
    """
    regulate the input data from different sources and generate output data compatible for the algorithm input
    """
    def __init__(self, file_path = "/mnt/data/workspace/LLM_Distribution_Center/data/example.csv"):
        """
        Args:
            file_path (_type_): local file path
            
        """
        self.file_path = file_path
        self.random_state = 930
        self.df = pd.read_csv(file_path)
        self.common_words = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I'])
    def generate_task_set_for_each_timestamp(self, task_num):
        """
        
        randomly select task_num tasks from data iterator

        Args:
            task_num (_type_): task_set size
        """
        sampled_task_set = self.df.sample(n = task_num, random_state = self.random_state)
        task_dicts = sampled_task_set.to_dict(orient = 'records')
                
        return task_dicts
    
    def generate_query_word_from_task_type(self, TASK_TYPE):
        """
        generate query word from task type, fixed query mode, without considering the output format
        TASK_TYPE : 0 : TC, 1 : NER, 2 : QA, 3 : TL, 4 : SG
        """
        TASK_TYPE  = TaskType(TASK_TYPE)
        if TASK_TYPE == TaskType.TC:
            res = f"Please classify the sentiment of the following text into one of the categories: sad, happy, love, angry, fear, surprise \n\n Text:"
            return res
        elif TASK_TYPE == TaskType.NER:
            res = f"Please identify the named entities in the following text. Classify entities into categories such as Person, Location, Organization, Miscellaneous \n\n Text:"
            return res
        elif TASK_TYPE == TaskType.QA:
            res = f"please answer the following question based on the text provided \n\n Question:"
            return res
        elif TASK_TYPE == TaskType.TL:
            res = "please translate the following text into English \n\n text:"
            return res
        elif TASK_TYPE == TaskType.SG:
            res = "Please summarize the following text \n\n Text:"
            return res
        else:
            return ""
    
    def calculate_input_features(self, text):
        
        # 1. 文本长度
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(nltk.sent_tokenize(text))

        # 2. 词汇复杂度
        avg_word_length = char_count / word_count
        rare_words_ratio = len([w for w in text.split() if w.lower() not in self.common_words]) / word_count

        # 3. 句法复杂度
        doc = nlp(text)
        avg_sentence_length = word_count / sentence_count
        max_depth = max(len(list(token.ancestors)) for token in doc)

        # 6. 上下文依赖性
        context_words = ['it', 'this', 'that', 'these', 'those', 'he', 'she', 'they']
        context_dependency = len(re.findall(r'\b(' + '|'.join(context_words) + r')\b', text.lower())) / word_count

        # 7. 歧义程度
        ambiguity_score = sum(len(wordnet.synsets(word)) for word in text.split()) / word_count

        # 8. 信息密度
        content_words = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
        information_density = len(content_words) / len(doc)

        # 9. 特殊符号和数字比例
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / char_count
        digit_ratio = len(re.findall(r'\d', text)) / char_count
        
        vocab_complexity = 0.5 * rare_words_ratio + 0.5 * avg_sentence_length
        ambiguity = ambiguity_score
        info_density = information_density
        special_char_ratio_value =  0.5 * special_char_ratio + 0.5 * digit_ratio
        return {
            'vocab_complexity': vocab_complexity,
            'syntax_complexity': max_depth,
            'context_dependency': context_dependency,
            'ambiguity': ambiguity,
            'info_density': info_density,
            'special_char_ratio': special_char_ratio_value
        }


if __name__ == "__main__":
    input = InputProcessor('/mnt/data/workspace/LLM_Distribution_Center/data/example.csv')
    res = input.generate_task_set_for_each_timestamp(task_num = 10)
    res2 = input.generate_query_word_from_task_type(2)
    test_text = "What is the impact of deep learning on modern AI applications?"
    print(input.calculate_input_features(test_text))
    pass