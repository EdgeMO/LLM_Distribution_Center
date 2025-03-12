# 基于本地数据，生成算法适配的随机批输入, 并且提供数据查询接口，返回基于 task id 的具体数据

import os
import sys

import pandas as pd
import re
current_working_directory = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(current_working_directory)
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from config.type import TaskType
class InputProcessor:
    """
    regulate the input data from different sources and generate output data compatible for the algorithm input
    """
    def __init__(self, file_path = "/home/wu/workspace/LLM_Distribution_Center/data/example.csv"):
        """
        Args:
            file_path (_type_): local file path
            
        """
        self.file_path = file_path
        self.random_state = 930
        self.df = pd.read_csv(file_path)
        self.common_words = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I'])
        self.call_count = 0 # generate random seed
        
    def generate_task_set_for_each_timestamp(self, task_num):
        """
        randomly select task_num tasks from data iterator
        
        Args:
            task_num (_type_): task_set size
        """
        # 使用递增的随机种子
        current_seed = self.random_state + self.call_count
        self.call_count += 1
        
        sampled_task_set = self.df.sample(n=task_num, random_state=current_seed)
        task_dicts = sampled_task_set.to_dict(orient='records')
                
        return task_dicts
    
    def generate_query_word_from_task_type(self, TASK_TYPE):
        """
        generate query word from task type, fixed query mode, without considering the output format
        TASK_TYPE : 0 : TC, 1 : NER, 2 : QA, 3 : TL, 4 : SG
        """
        TASK_TYPE = int(TASK_TYPE)
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
        """
        使用Python原生方法计算中英文文本的各项特征指标
        """
        # 检测文本主要语言（简单判断是否主要为中文）
        is_mainly_chinese = len(re.findall(r'[\u4e00-\u9fff]', text)) > len(text) / 3
        
        # 预处理：移除多余空格
        text = ' '.join(text.split())
        
        # 1. 基本统计
        char_count = len(text)
        
        # 中英文分词和句子分割不同处理
        if is_mainly_chinese:
            # 中文分词（简单实现，实际应用可能需要更复杂的分词）
            words = self.chinese_word_segmentation(text)
            word_count = len(words)
            
            # 中文句子通常以句号、问号、感叹号、分号结束
            sentences = re.split(r'[。！？；!?;.]+', text)
        else:
            # 英文分词
            words = text.split()
            word_count = len(words)
            
            # 英文句子分割
            sentences = re.split(r'[.!?]+', text)
        
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences) if sentences else 1
        
        # 避免除零错误
        if word_count == 0:
            return {
                'vocabulary_complexity': 0,
                'syntactic_complexity': 0,
                'context_dependency': 0,
                'ambiguity_level': 0,
                'information_density': 0,
                'special_symbol_ratio': 0
            }
        
        # 2. 词汇复杂度
        if is_mainly_chinese:
            # 中文词汇复杂度：平均词长和罕见词比例
            avg_word_length = sum(len(w) for w in words) / word_count
            
            # 中文常用词表
            common_chinese_words = set([
                '的', '是', '不', '了', '在', '人', '有', '我', '他', '这', 
                '个', '们', '中', '来', '上', '大', '为', '和', '国', '地', 
                '到', '以', '说', '时', '要', '就', '出', '会', '可', '也', 
                '你', '对', '生', '能', '而', '子', '那', '得', '于', '着', 
                '下', '自', '之', '年', '过', '发', '后', '作', '里', '用'
            ])
            rare_words_ratio = len([w for w in words if w not in common_chinese_words]) / word_count
        else:
            # 英文词汇复杂度
            avg_word_length = sum(len(w) for w in words) / word_count
            
            # 英文常用词表
            common_english_words = set([
                'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
                'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
                'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
                'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
                'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me'
            ])
            rare_words_ratio = len([w.lower() for w in words if w.lower() not in common_english_words]) / word_count
        
        # 3. 句法复杂度
        avg_sentence_length = word_count / sentence_count
        
        # 中英文标点符号和连词
        if is_mainly_chinese:
            # 中文标点符号和连词
            punctuation_count = len(re.findall(r'[，、；：""''（）【】《》]', text))
            chinese_connectives = ['和', '与', '而', '但', '但是', '不过', '然而', '所以', '因此', '因为', 
                                '如果', '虽然', '尽管', '即使', '无论', '只要', '除非', '以及']
            connective_count = sum(1 for word in words if word in chinese_connectives)
        else:
            # 英文标点符号和连词
            punctuation_count = len(re.findall(r'[,;:()$$$${}]', text))
            english_connectives = ['and', 'but', 'or', 'so', 'because', 'if', 'when', 'although', 'while', 'since']
            connective_count = sum(1 for word in words if word.lower() in english_connectives)
        
        # 句法复杂度：句子平均长度、标点符号和连词的加权和
        syntactic_complexity = (0.5 * avg_sentence_length + 
                            0.3 * (punctuation_count / sentence_count) + 
                            0.2 * (connective_count / sentence_count))
        
        # 6. 上下文依赖性
        if is_mainly_chinese:
            # 中文上下文依赖词
            context_words = ['这', '那', '它', '他', '她', '他们', '她们', '它们', '这些', '那些', 
                            '此', '该', '其', '之', '前者', '后者']
        else:
            # 英文上下文依赖词
            context_words = ['it', 'this', 'that', 'these', 'those', 'he', 'she', 'they', 
                            'him', 'her', 'them', 'his', 'hers', 'their', 'theirs']
        
        if is_mainly_chinese:
            context_word_count = sum(1 for word in words if word in context_words)
        else:
            context_word_count = sum(1 for word in words if word.lower() in context_words)
        
        context_dependency = context_word_count / word_count
        
        # 7. 歧义程度
        if is_mainly_chinese:
            # 中文歧义估计：单字词更可能有歧义
            ambiguity_scores = []
            for word in words:
                if len(word) == 1 and re.match(r'[\u4e00-\u9fff]', word):
                    ambiguity_scores.append(2.0)  # 单字词通常有更多歧义
                elif word in common_chinese_words:
                    ambiguity_scores.append(1.5)  # 常用词有一定歧义
                elif len(word) == 2:
                    ambiguity_scores.append(1.0)  # 双字词有中等歧义
                else:
                    ambiguity_scores.append(0.5)  # 多字词通常歧义较少
        else:
            # 英文歧义估计
            ambiguity_scores = []
            for word in words:
                word = word.lower()
                # 短词且常见的词可能有更多歧义
                if len(word) <= 4:
                    ambiguity_scores.append(1.5)
                elif word in common_english_words:
                    ambiguity_scores.append(1.0)
                else:
                    ambiguity_scores.append(0.5)
        
        ambiguity_level = sum(ambiguity_scores) / word_count
        
        # 8. 信息密度
        if is_mainly_chinese:
            # 中文内容词：非常用词且长度大于1的词
            content_words = [w for w in words if len(w) > 1 and w not in common_chinese_words]
        else:
            # 英文内容词：长度大于3且不在常见词列表中的词
            content_words = [w for w in words if len(w) > 3 and w.lower() not in common_english_words]
        
        information_density = len(content_words) / word_count
        
        # 9. 特殊符号和数字比例
        # 对中英文都通用
        special_char_count = len(re.findall(r'[^\w\s\u4e00-�]', text))
        digit_count = len(re.findall(r'\d', text))
        
        special_char_ratio = special_char_count / char_count if char_count > 0 else 0
        digit_ratio = digit_count / char_count if char_count > 0 else 0
        
        # 计算最终指标
        vocab_complexity = 0.5 * rare_words_ratio + 0.5 * avg_word_length
        special_char_ratio_value = 0.5 * special_char_ratio + 0.5 * digit_ratio
        
        return {
            'vocabulary_complexity': vocab_complexity,
            'syntactic_complexity': syntactic_complexity,
            'context_dependency': context_dependency,
            'ambiguity_level': ambiguity_level,
            'information_density': information_density,
            'special_symbol_ratio': special_char_ratio_value
        }

    def chinese_word_segmentation(self, text):
        """
        简单的中文分词实现，使用最大正向匹配算法
        实际应用中可以替换为更复杂的分词算法
        """
        # 一个简单的中文词典，实际应用需要更完整的词典
        dictionary = [
            '中国', '人民', '共和国', '北京', '上海', '广州', '深圳', '天津', '重庆',
            '我们', '你们', '他们', '这个', '那个', '什么', '为什么', '怎么样', '如何',
            '经济', '文化', '科技', '发展', '社会', '政治', '历史', '未来', '现在',
            '工作', '学习', '生活', '家庭', '朋友', '同事', '老师', '学生', '公司',
            '计算机', '互联网', '人工智能', '大数据', '云计算', '软件', '硬件', '编程'
        ]
        # 按词长排序，优先匹配长词
        dictionary.sort(key=len, reverse=True)
        
        result = []
        i = 0
        text_len = len(text)
        
        while i < text_len:
            matched = False
            # 尝试匹配词典中的词
            for word in dictionary:
                word_len = len(word)
                if i + word_len <= text_len and text[i:i+word_len] == word:
                    result.append(word)
                    i += word_len
                    matched = True
                    break
            
            # 如果没有匹配到词典中的词，将单个字符作为一个词
            if not matched:
                result.append(text[i])
                i += 1
        
        return result
        
    def generate_offloading_message(self,edge_id_to_task_id_dict, task_set):
        output = []
        
        for key, task_id_list in edge_id_to_task_id_dict.items():
            res = {}
            res['edge_id'] = key
            task_set_for_edge = []
            for task_id in task_id_list:
                temp_res = {}
                matching_tasks = [task for task in task_set if task.get("task_id") == task_id]
                task_type = matching_tasks[0].get("task_type")
                query_word = self.generate_query_word_from_task_type(task_type)
                task_token = matching_tasks[0].get("task_token")
                reference_value = matching_tasks[0].get("reference_value")
                temp_res['task_id'] = task_id
                temp_res['task_type'] = task_type
                temp_res['task_token'] = query_word + task_token
                temp_res['reference_value'] = reference_value
                task_set_for_edge.append(temp_res)
            res['task_set'] = task_set_for_edge
            output.append(res)
        return output
    
if __name__ == "__main__":
    input = InputProcessor('/home/wu/workspace/LLM_Distribution_Center/data/example.csv')
    res = input.generate_task_set_for_each_timestamp(task_num = 10)
    res2 = input.generate_query_word_from_task_type(2)
    test_text = "What is the impact of deep learning on modern AI applications?"
    print(input.calculate_input_features(test_text))
    pass