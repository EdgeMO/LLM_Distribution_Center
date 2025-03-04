
# QA true value enum
from collections import Counter
import math

class Metrics:
    def __init__(self):
        
        pass
    
    def TC_Metric(self,prediction, reference):
        """
        0 1 accuracy metric for TC task

        Args:
            prediction (_type_): token generated from model
            reference (_type_): TC true value type from dataset

        Returns:
            _type_: 0 or 1 
        """
        if type(reference) != str:
            reference = str(reference)
        # store all the categories if example csv
        EMOTION_CATEGORIES = {
            "0": "sad",
            "1": "happy",
            "2": "love",
            "3": "angry",
            "4": "scared",
            "5": "surprise"
        }
        # store all the keywords for single category
        EMOTION_KEYWORDS = {
            "sad": [
                "sad", "unhappy", "depressed", "gloomy", "heartbroken", "melancholy", "sorrowful", 
                "downcast", "miserable", "grieving", "dejected", "despondent", "dismal", "woeful", 
                "tearful", "upset", "distressed", "down", "blue", "troubled", "lonely", "hopeless",
                "crestfallen", "doleful", "mournful", "wistful", "pessimistic", "forlorn", "desolate",
                "heavy-hearted", "somber", "downhearted", "glum", "morose", "sullen", "lugubrious",
                "inconsolable", "anguished", "dreary", "weeping", "sobbing", "crying", "wretched",
                "feeling low", "in the dumps", "down in the mouth", "under the weather",
                "broken-hearted", "crushed", "defeated", "discouraged", "dispirited", "bummed out",
                "soul-crushing", "gut-wrenching", "world-weary", "melancholic", "lachrymose",
                "not a happy camper", "like the weight of the world on my shoulders"
            ],
            "happy": [
                "happy", "joyful", "delighted", "cheerful", "glad", "elated", "ecstatic", "merry", 
                "jolly", "gleeful", "thrilled", "upbeat", "jubilant", "excited", "pleased", "content", 
                "satisfied", "blissful", "euphoric", "overjoyed", "radiant", "beaming", "grinning",
                "exuberant", "buoyant", "chipper", "blithe", "jovial", "mirthful", "lighthearted",
                "sunny", "breezy", "peppy", "zestful", "vivacious", "exhilarated", "in high spirits",
                "walking on air", "on cloud nine", "over the moon", "tickled pink", "on top of the world",
                "giddy", "chirpy", "perky", "zippy", "bouncy", "lively", "animated", "effervescent",
                "in seventh heaven", "on a high", "flying high", "in good cheer", "full of beans",
                "cock-a-hoop", "jubilant", "rapturous", "in raptures", "blissed out", "stoked",
                "chuffed", "like a kid in a candy store", "happy as a clam", "grinning from ear to ear"
            ],
            "love": [
                "love", "adore", "affection", "passion", "fondness", "devotion", "admiration", "cherish", 
                "infatuation", "enamored", "smitten", "caring", "tender", "warm", "attachment", "attracted", 
                "romantic", "intimate", "desire", "longing", "yearning", "compassion", "beloved",
                "amorous", "ardent", "besotted", "doting", "enchanted", "enthralled", "fascinated",
                "head over heels", "idolize", "infatuated", "lovestruck", "moonstruck", "passionate",
                "puppy love", "starry-eyed", "swooning", "taken with", "twitterpated", "worshipful",
                "crazy about", "wild about", "mad about", "carried away", "captivated", "charmed",
                "fallen for", "fancying", "fond of", "gone on", "keen on", "pining for", "sweet on",
                "love-struck", "lovesick", "heart aflutter", "weak at the knees", "butterflies in stomach",
                "heartthrob", "paramour", "sweetheart", "significant other", "better half", "soulmate"
            ],
            "angry": [
                "angry", "furious", "annoyed", "irritated", "enraged", "irate", "livid", "incensed", 
                "outraged", "fuming", "seething", "mad", "bitter", "resentful", "exasperated", "indignant", 
                "agitated", "hostile", "antagonistic", "provoked", "infuriated", "cross", "vexed",
                "aggravated", "apoplectic", "ballistic", "boiling", "choleric", "enraged", "foaming at the mouth",
                "frenzied", "heated", "hot under the collar", "huffy", "in a huff", "in a rage", "incandescent",
                "irascible", "ireful", "losing it", "miffed", "on the warpath", "peeved", "piqued", "raging",
                "ranting", "raving", "seeing red", "spitting mad", "steamed", "storming", "teed off", "ticked off",
                "up in arms", "worked up", "wrath", "wrathful", "bristling", "indignant", "ferocious",
                "beside oneself", "fit to be tied", "hopping mad", "blowing a fuse", "hitting the roof",
                "flying off the handle", "losing one's cool", "blowing a gasket", "having a conniption"
            ],
            "scared": [
                "scared", "frightened", "terrified", "anxious", "fearful", "afraid", "panicked", "alarmed", 
                "startled", "nervous", "uneasy", "worried", "apprehensive", "dreadful", "horrified", "petrified", 
                "spooked", "timid", "shaken", "jittery", "trembling", "phobic", "intimidated",
                "aghast", "antsy", "chicken", "cowed", "daunted", "disquieted", "edgy", "fidgety", "frantic",
                "freaked out", "fretful", "hair-raising", "heart-pounding", "hysterical", "jumpy", "nailbiting",
                "nervy", "on edge", "overwrought", "quaking", "quivering", "rattled", "restless", "shaky",
                "shot", "skittish", "spineless", "sweating bullets", "wary", "worked up", "yellow",
                "goosebumps", "hair standing on end", "blood running cold", "paralyzed with fear",
                "scared stiff", "scared witless", "shaking like a leaf", "white as a sheet", "weak-kneed",
                "teeth chattering", "breaking out in a cold sweat", "heart in one's mouth", "knees knocking"
            ],
            "surprise": [
                "surprised", "amazed", "astonished", "shocked", "stunned", "startled", "dumbfounded", 
                "bewildered", "astounded", "awestruck", "flabbergasted", "thunderstruck", "taken aback", 
                "dazed", "speechless", "goggle-eyed", "open-mouthed", "agape", "wonder-struck", "staggered", 
                "nonplussed", "baffled", "perplexed",
                "gobsmacked", "floored", "knocked for a loop", "blindsided", "caught off guard",
                "caught unawares", "dazed and confused", "discombobulated", "disoriented", "bowled over",
                "blown away", "jaw-dropping", "eye-opening", "mind-blowing", "head-spinning",
                "didn't see that coming", "out of the blue", "bolt from the blue", "never saw it coming",
                "stopped in one's tracks", "couldn't believe one's eyes", "rubbing one's eyes",
                "doing a double-take", "struck dumb", "at a loss for words", "knocked my socks off",
                "blown my mind", "hit me like a ton of bricks", "left me reeling", "took my breath away"
            ]
        }
        
        prediction = prediction.lower()
        true_category = EMOTION_CATEGORIES[reference]
        
        # 检查每个类别的关键词
        for category, keywords in EMOTION_KEYWORDS.items():
            if any(keyword in prediction for keyword in keywords):
                return int(category == true_category)
        return 0
    
    def NER_Metric(self, prediction, reference):
        """

        Args:
            prediction (_type_): _description_
            reference (_type_): LOC PER MISC
        """
        prediction = prediction.lower()
        if type(reference) != str:
            reference = str(reference)
        EMOTION_CATEGORIES = {
            "MISC": "miscellaneous",
            "LOC": "location",
            "PER": "person"
        }
        reference_type = EMOTION_CATEGORIES.get('reference',"location")
        reference_type = reference_type.lower()
        if reference_type in prediction:
            return 1
        return 0
    
    def TL_Metric(self, prediction, reference):
        """
        with given reference data format, we need to switch the reference and prediction to calculate the BLEU score
        because of the data format
        Args:
            prediction (_type_): _description_
            reference (_type_): _description_
        """
        def count_ngrams(segment, n):
            """计算 n-gram 的出现次数"""
            return Counter(zip(*[segment[i:] for i in range(n)]))

        def modified_precision(candidate, reference, n):
            """计算修正后的精确度"""
            candidate_ngrams = count_ngrams(candidate, n)
            reference_ngrams = count_ngrams(reference, n)
            
            if not candidate_ngrams:
                return 0
            
            clipped_counts = {ngram: min(count, reference_ngrams.get(ngram, 0)) 
                            for ngram, count in candidate_ngrams.items()}
            
            return sum(clipped_counts.values()) / sum(candidate_ngrams.values())

        def brevity_penalty(candidate, reference):
            """计算简短惩罚"""
            c = len(candidate)
            r = len(reference)
            
            if c > r:
                return 1
            else:
                return math.exp(1 - r/c)

        def calculate_bleu(prediction, reference, max_n=4):
            """
            计算 BLEU 分数
            
            :param prediction: 预测的翻译（字符串）
            :param reference: 参考翻译（字符串）
            :param max_n: 考虑的最大 n-gram（默认为4）
            :return: BLEU 分数（0到1之间的浮点数）
            """
            prediction_tokens = prediction.split()
            reference_tokens = reference.split()
            
            bp = brevity_penalty(prediction_tokens, reference_tokens)
            
            precisions = [modified_precision(prediction_tokens, reference_tokens, n) 
                        for n in range(1, max_n+1)]
            
            if all(p > 0 for p in precisions):
                log_avg = sum(math.log(p) for p in precisions) / len(precisions)
                bleu = bp * math.exp(log_avg)
            else:
                bleu = 0
            
            return bleu
        # with given reference data format, we need to switch the reference and prediction to calculate the BLEU score
        # because of the data format
        res = calculate_bleu(reference, prediction)
        return res  
    
    def SG_Metric(self,prediction, reference):
        def length_adaptability_score(summary, reference_summary):
            """
            计算长度适配性得分
            
            :param summary: 生成的摘要
            :param reference_summary: 参考摘要
            :return: 0到1之间的分数，1表示长度完全匹配
            """
            summary_length = len(summary.split())
            reference_length = len(reference_summary.split())
            
            # 使用指数衰减函数来计算得分
            # 当两个长度相等时，得分为1
            # 长度差异越大，得分越接近0
            score = math.exp(-abs(summary_length - reference_length) / reference_length)
            
            return score
        def cosine_similarity(text1, text2):
            """计算两段文本的余弦相似度"""
            words1 = text1.lower().split()
            words2 = text2.lower().split()
            
            counter1 = Counter(words1)
            counter2 = Counter(words2)
            
            all_words = set(counter1.keys()) | set(counter2.keys())
            
            dot_product = sum(counter1[word] * counter2[word] for word in all_words)
            magnitude1 = math.sqrt(sum(counter1[word]**2 for word in counter1))
            magnitude2 = math.sqrt(sum(counter2[word]**2 for word in counter2))
            
            if magnitude1 * magnitude2 == 0:
                return 0
            
            return dot_product / (magnitude1 * magnitude2)
        def rouge_n(summary, reference, n=1):
            """计算ROUGE-N分数"""
            summary_ngrams = Counter(zip(*[summary.split()[i:] for i in range(n)]))
            reference_ngrams = Counter(zip(*[reference.split()[i:] for i in range(n)]))
            
            overlap = sum((summary_ngrams & reference_ngrams).values())
            total = sum(reference_ngrams.values())
            
            return overlap / total if total > 0 else 0
        rouge_score = rouge_n(prediction, reference, n=1)
        cosine_score = cosine_similarity(prediction, reference)
        length_adaptability_score_value = length_adaptability_score(prediction, reference)
        
        overall_score = (rouge_score * 0.48 + cosine_score* 0.48 + length_adaptability_score_value * 0.04)
        return overall_score
    
    
    def QA_Metric(self, prediction, reference ):
        return self.SG_Metric(prediction, reference)
    
    def process(self, type, prediction, reference):
        """
        generate the score of single task request

        Args:
            type (_type_): the number of specific task type 
            prediction (_type_): model output
            reference (_type_): true value from dataset
        """
        score = 0
        if type == 0:
            score = self.TC_Metric(prediction, reference)
        if type == 1:
            score = self.NER_Metric(prediction, reference)
        if type == 2:
            score = self.QA_Metric(prediction, reference)
        if type == 3:
            score = self.TL_Metric(prediction, reference)
        if type ==4:
            score = self.SG_Metric(prediction, reference)
        return score
        
    
        
        
if __name__ == "__main__":
    metrics_calculator = Metrics()
    