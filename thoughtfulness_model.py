import re

import numpy as np
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel


class ThoughtfulnessModel:
    def __init__(self, final_keywords, bert_model="bert-base-uncased", max_length=512, bert_threshold=0.6):
        self.final_keywords = final_keywords
        self.max_length = max_length
        self.bert_threshold = bert_threshold
        self.sia = SentimentIntensityAnalyzer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载 BERT 作为文本嵌入模型（去掉分类头）
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.model = BertModel.from_pretrained(bert_model).to(self.device)

        # 代表性高质量评论（可以扩展）
        self.representative_reviews = [
            # Gameplay
            "This game has amazing gameplay with deep mechanics and great storytelling.",
            "The controls are smooth, and the difficulty curve is just right.",
            "The combat system is incredibly engaging and rewarding.",

            # Graphics
            "The graphics are stunning, and the combat system is well-balanced.",
            "The attention to detail in the textures and lighting is amazing.",
            "The game offers breathtaking visuals and smooth animations.",

            # Sound and Music
            "The soundtrack perfectly complements the mood of the game.",
            "The audio effects are immersive and enhance the gameplay experience.",
            "The voice acting is top-notch, adding depth to the characters.",

            # Storyline
            "I love the character development and the immersive world-building.",
            "The story is captivating, with well-written dialogues and plot twists.",
            "The narrative keeps you engaged from start to finish.",

            # User Experience
            "The interface is intuitive and easy to navigate.",
            "The game is well-optimized, with no noticeable lag or bugs.",
            "The tutorials are clear, making it easy to get started."
        ]
        self.rep_vectors = np.array([self.get_bert_embedding(review) for review in self.representative_reviews])

    def get_bert_embedding(self, text):
        """使用 BERT 提取文本嵌入"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.max_length).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()  # 取 CLS token 作为表示

    def bert_similarity_score(self, text):
        """计算评论与高质量评论之间的相似度"""
        text_vector = self.get_bert_embedding(text)
        similarity_scores = cosine_similarity([text_vector], self.rep_vectors)
        return np.mean(similarity_scores)  # 取均值作为分数

    def clean_text(self, text):
        """清洗评论文本"""
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.lower().strip()

    def level_of_detail(self, text):
        """计算 Level of Detail（细节程度）"""
        sentences = text.split(".")
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0
        return avg_sentence_length

    def ttr(self, text):
        """计算 Type-Token Ratio（词汇丰富度）"""
        words = text.split()
        return len(set(words)) / len(words) if len(words) > 0 else 0

    def sentiment_intensity(self, text):
        """计算情感强度"""
        if not text.strip():
            return 0  # 如果文本为空，返回情感分数为 0
        return self.sia.polarity_scores(text)['compound']

    def relevance_score(self, text):
        """计算主题相关性"""
        words = text.split()
        relevant_words = [word for word in words if word in self.final_keywords]
        return len(relevant_words) / len(words) if len(words) > 0 else 0

    def hybrid_tier_prediction(self, text):
        cleaned_text = self.clean_text(text)

        # 规则过滤
        if len(cleaned_text) <= 10:
            return 0
        if self.level_of_detail(cleaned_text) <= 5 or self.ttr(cleaned_text) <= 0.4:
            return 1
        if abs(self.sentiment_intensity(cleaned_text)) <= 0.5 or self.relevance_score(cleaned_text) <= 0.2:
            return 2

        # BERT 相似度评分
        bert_score = self.bert_similarity_score(cleaned_text)
        if bert_score > self.bert_threshold:  # 0.6 是一个经验值，可以调整
            return 3
        return 2  # 如果 BERT 评分较低，则降为 Tier 2


# 手动定义关键词
manual_keywords = list(
    {"gameplay", "mechanics", "levels", "difficulty", "balance", "graphics", "resolution", "textures", "lighting",
     "animation", "soundtrack", "soundtracks", "audio", "music", "effects", "story", "plot", "characters", "narrative",
     "controls", "input", "interface", "ak", "m4a1", "sniper", "headshot", "weapon", "recoil", "quests", "loot",
     "skills", "character", "dialogue", "leveling", "ranked", "matchmaking", "strategy", "heroes", "abilities"})

# 初始化模型
model = ThoughtfulnessModel(manual_keywords)

# 示例评论
review = "The gameplay was amazing, with smooth controls and stunning graphics!"
tier_labels = {
    0: "Tier 0: Not Thoughtful (too short or irrelevant)",
    1: "Tier 1: Basic (lacks detail or uniqueness)",
    2: "Tier 2: Intermediate (relevant but not deep)",
    3: "Tier 3: Thoughtful (detailed, unique, and relevant)"
}
tier = model.hybrid_tier_prediction(review)
print(f"The review belongs to: {tier_labels[tier]}")
