import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, Counter
import jieba
import re
import gradio as gr
import os
import zipfile
from tqdm import tqdm
import random
import platform
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import logging
import time
from threading import Thread
from typing import List, Tuple, Optional, Deque, Dict, Any


# -------------------------- 1. 基础配置 --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")


# -------------------------- 2. 基础组件类 --------------------------
class LeakyIntegrateFire(nn.Module):
    """
    脉冲神经网络模块，模拟生物神经元的漏积分发放机制

    参数:
        input_size (int): 输入特征维度
        hidden_size (int): 隐藏层维度
        beta (float): 膜电位衰减系数 (0.0-1.0)
        threshold (float): 神经元发放阈值
        reset_value (float): 发放后重置电位值
    """

    def __init__(self, input_size: int, hidden_size: int, beta: float = 0.9,
                 threshold: float = 1.0, reset_value: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.beta = beta
        self.threshold = threshold
        self.reset_value = reset_value
        self.fc = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None,
                spike: Optional[torch.Tensor] = None, num_steps: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播过程

        参数:
            x: 输入张量 (batch_size, input_size) 或 (batch_size, num_steps, input_size)
            mem: 初始膜电位 (batch_size, hidden_size)
            spike: 初始脉冲状态 (batch_size, hidden_size)
            num_steps: 模拟时间步数

        返回:
            spikes: 脉冲序列 (batch_size, num_steps, hidden_size)
            final_mem: 最终膜电位 (batch_size, hidden_size)
        """
        batch_size = x.size(0)

        # 处理输入形状
        if x.dim() == 3 and x.size(1) == num_steps:
            time_series = x
        else:
            time_series = x.unsqueeze(1).repeat(1, num_steps, 1)

        # 初始化状态
        if mem is None:
            mem = torch.full((batch_size, self.hidden_size), self.reset_value, device=x.device)
        if spike is None:
            spike = torch.zeros((batch_size, self.hidden_size), device=x.device)

        spikes = torch.zeros((batch_size, num_steps, self.hidden_size), device=x.device)

        for step in range(num_steps):
            current = self.fc(time_series[:, step, :])
            mem = self.beta * mem + (1 - self.beta) * current
            spike = (mem >= self.threshold).float()
            mem = mem * (1 - spike) + self.reset_value * spike
            spikes[:, step, :] = spike

        return spikes, mem


class NeuroModulator:
    """
    神经调节模块，模拟多巴胺系统对学习的影响

    参数:
        initial_dopamine (float): 初始多巴胺水平
        learning_rate (float): 学习率，控制多巴胺更新速度
    """

    def __init__(self, initial_dopamine: float = 0.5, learning_rate: float = 0.01):
        self.dopamine = initial_dopamine
        self.novelty_history: Deque[float] = deque(maxlen=100)
        self.learning_rate = learning_rate

    def update_dopamine(self, novelty: float) -> float:
        """
        根据新颖性更新多巴胺水平

        参数:
            novelty: 当前刺激的新颖性评分 (0.0-1.0)

        返回:
            更新后的多巴胺水平
        """
        self.novelty_history.append(novelty)
        avg_novelty = np.mean(self.novelty_history) if self.novelty_history else 0.5

        # 使用更稳定的sigmoid风格更新
        difference = novelty - avg_novelty
        update = self.learning_rate * np.tanh(difference * 2)  # 限制更新幅度

        self.dopamine += update
        self.dopamine = np.clip(self.dopamine, 0.1, 0.9)
        return self.dopamine

    def get_noise_intensity(self, creativity_level: float = 1.0) -> float:
        """
        获取噪声强度，用于促进探索

        参数:
            creativity_level: 创造力调节因子

        返回:
            噪声强度值
        """
        return creativity_level * (0.5 + self.dopamine) / 1.5


# -------------------------- 3. TextDataset（强制统一样本长度） --------------------------
class TextDataset(Dataset):
    """
    文本数据集类，处理变长文本并统一为固定长度

    参数:
        texts: 文本列表
        tokenizer: 分词器实例
        max_length: 最大序列长度
    """

    def __init__(self, texts: List[str], tokenizer: 'ImprovedTokenizer', max_length: int = 50):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_length = max_length - 1
        self.filter_short_texts()

    def filter_short_texts(self):
        """过滤过短的文本"""
        filtered = []
        for text in self.texts:
            tokens = self.tokenizer.tokenize(text, self.max_length)
            if len(tokens) >= 2:
                filtered.append(text)
        self.texts = filtered
        if not self.texts:
            raise ValueError(f"没有有效文本（需单段文本token数≥2，当前max_length={self.max_length}）")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        tokens = self.tokenizer.tokenize(text, self.max_length)

        # 确保填充后长度正确
        if len(tokens) < self.max_length:
            pad_num = self.max_length - len(tokens)
            tokens += [self.tokenizer.word2idx['<PAD>']] * pad_num
        else:
            tokens = tokens[:self.max_length]  # 截断

        input_tokens = tokens[:self.input_length]
        target_tokens = tokens[1:self.max_length]

        # 确保输入和目标长度一致
        if len(input_tokens) != self.input_length:
            input_tokens = input_tokens[:self.input_length] + [self.tokenizer.word2idx['<PAD>']] * (
                        self.input_length - len(input_tokens))

        if len(target_tokens) != self.input_length:
            target_tokens = target_tokens[:self.input_length] + [self.tokenizer.word2idx['<PAD>']] * (
                        self.input_length - len(target_tokens))

        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)


# -------------------------- 4. 其他核心类 --------------------------
class DataProcessor:
    """
    数据处理类，负责ZIP文件解压和文本加载

    参数:
        data_dir: 数据存储目录
    """

    def __init__(self, data_dir: str = 'user_data'):
        self.data_dir = data_dir
        self.extracted_dir = None
        os.makedirs(self.data_dir, exist_ok=True)
        self.progress_callback = None

    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback

    def _update_progress(self, percent: float, message: str):
        """更新进度信息"""
        if self.progress_callback:
            try:
                self.progress_callback(percent, message)
            except Exception as e:
                logging.warning(f"进度更新失败: {str(e)}")
        logging.info(f"[数据处理] {percent:.1f}% - {message}")

    def unzip_file(self, zip_path: str) -> Tuple[str, List[str]]:
        """
        解压ZIP文件

        参数:
            zip_path: ZIP文件路径

        返回:
            解压目录和TXT文件列表
        """
        try:
            if not zipfile.is_zipfile(zip_path):
                raise ValueError(f"{zip_path} 不是有效的ZIP文件")
            filename_base = os.path.basename(zip_path).split('.')[0]
            self.extracted_dir = os.path.join(self.data_dir, f"extracted_{filename_base}")
            os.makedirs(self.extracted_dir, exist_ok=True)

            self._update_progress(10, "开始解压ZIP文件...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                txt_files = [f for f in file_list if f.lower().endswith('.txt')]
                if not txt_files:
                    raise ValueError("ZIP文件中未找到任何TXT文本文件")

                total_files = len(txt_files)
                for i, file in enumerate(txt_files):
                    zip_ref.extract(file, self.extracted_dir)
                    progress = 10 + (i / total_files) * 30
                    self._update_progress(progress, f"解压完成: {os.path.basename(file)}")

            self._update_progress(40, f"ZIP解压完成，共 {len(txt_files)} 个TXT文件")
            return self.extracted_dir, txt_files
        except Exception as e:
            logging.error(f"解压失败: {str(e)}")
            raise

    def load_from_extracted(self, min_para_length: int = 20) -> List[str]:
        """
        从解压目录加载文本

        参数:
            min_para_length: 最小段落长度

        返回:
            有效文本列表
        """
        if not self.extracted_dir or not os.path.exists(self.extracted_dir):
            raise FileNotFoundError(f"解压目录不存在: {self.extracted_dir}")

        self._update_progress(45, "开始读取TXT文本...")
        texts = []
        txt_files = []
        for root, _, files in os.walk(self.extracted_dir):
            for f in files:
                if f.lower().endswith('.txt'):
                    txt_files.append(os.path.join(root, f))

        if not txt_files:
            raise ValueError("解压目录中未找到TXT文件")

        total_files = len(txt_files)
        for i, file_path in enumerate(txt_files):
            try:
                # 尝试UTF-8编码
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                except UnicodeDecodeError:
                    # 回退到GBK编码
                    try:
                        with open(file_path, 'r', encoding='gbk') as f:
                            content = f.read().strip()
                    except Exception as e:
                        logging.warning(f"无法读取文件 {file_path}: {str(e)}")
                        continue

                if not content:
                    logging.warning(f"跳过空文件: {file_path}")
                    continue

                cleaned_text = self.clean_text(content)
                paragraphs = re.split(r'[\n。？！;；]+', cleaned_text)
                valid_paragraphs = [p.strip() for p in paragraphs if len(p.strip()) >= min_para_length]
                texts.extend(valid_paragraphs)

                progress = 45 + (i / total_files) * 25
                self._update_progress(progress, f"读取完成: {os.path.basename(file_path)}")
            except Exception as e:
                logging.error(f"处理文件 {file_path} 出错: {str(e)}")
                continue

        if not texts:
            raise ValueError(f"未加载到有效文本（单段文本需≥{min_para_length}字符）")

        self._update_progress(70, f"文本加载完成，共 {len(texts)} 段有效文本")
        return texts

    def clean_text(self, text: str) -> str:
        """文本清洗"""
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。,、；;：:‘’"\'()（）\[\]{}!?？！.·\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class ImprovedTokenizer:
    """
    改进的分词器类，支持自定义词典和并行分词

    参数:
        max_vocab_size: 最大词汇表大小
        custom_dict: 自定义词典路径
        min_word_freq: 最小词频阈值
    """

    def __init__(self, max_vocab_size: int = 10000, custom_dict: Optional[str] = None, min_word_freq: int = 1):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.max_vocab_size = max_vocab_size
        self.vocab_size = 2
        self.word_counter = Counter()
        self.min_word_freq = min_word_freq
        self.progress_callback = None

        if custom_dict and os.path.exists(custom_dict):
            try:
                jieba.load_userdict(custom_dict)
                logging.info(f"已加载自定义词典: {custom_dict}")
            except Exception as e:
                logging.warning(f"加载自定义词典失败: {str(e)}")

    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback

    def _update_progress(self, percent: float, message: str):
        """更新进度信息"""
        if self.progress_callback:
            try:
                self.progress_callback(percent, message)
            except Exception as e:
                logging.warning(f"分词进度更新失败: {str(e)}")
        logging.info(f"[分词器] {percent:.1f}% - {message}")

    def fit(self, texts: List[str]):
        """构建词汇表"""
        self._update_progress(75, "开始分词与词汇统计...")
        use_parallel = False
        total_chars = sum(len(text) for text in texts)
        if platform.system() != "Windows" and total_chars >= 10000:
            try:
                jieba.enable_parallel()
                use_parallel = True
                logging.info(f"启用并行分词（文本总字符数: {total_chars}）")
            except Exception as e:
                logging.warning(f"启用并行失败，降级为单线程: {str(e)}")
        else:
            logging.info(f"使用单线程分词（系统: {platform.system()}, 文本总字符数: {total_chars}）")

        all_words = []
        total_texts = len(texts)
        for i, text in enumerate(texts):
            if not text or not isinstance(text, str):
                continue
            try:
                words = jieba.cut(text, cut_all=False)
                valid_words = [w.strip() for w in words if w.strip() and len(w.strip()) >= 1]
                all_words.extend(valid_words)
            except Exception as e:
                logging.warning(f"文本分词失败: {str(e)}")
                continue

            if i % max(1, total_texts // 20) == 0:
                progress = 75 + (i / total_texts) * 20
                self._update_progress(progress, f"处理文本 {i + 1}/{total_texts}")

        if use_parallel:
            try:
                jieba.disable_parallel()
            except:
                pass

        if not all_words:
            raise ValueError("未统计到任何词汇（请检查文本有效性）")
        self.word_counter = Counter(all_words)
        sorted_words = self.word_counter.most_common(self.max_vocab_size - 2)

        self.vocab_size = 2
        for word, count in sorted_words:
            if count >= self.min_word_freq and self.vocab_size < self.max_vocab_size:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

        if self.vocab_size <= 2:
            raise ValueError(f"词汇表构建失败（仅 {self.vocab_size} 个默认token），请增加文本多样性")

        self._update_progress(100, f"词汇表构建完成，词汇量: {self.vocab_size}")

    def tokenize(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """文本分词"""
        if not text or not isinstance(text, str):
            return [self.word2idx['<PAD>']] * (max_length if max_length else 1)
        words = jieba.cut(text, cut_all=False)
        tokens = []
        for word in words:
            token_idx = self.word2idx.get(word, self.word2idx['<UNK>'])
            tokens.append(token_idx)
            if max_length and len(tokens) >= max_length:
                break
        if max_length and len(tokens) < max_length:
            tokens += [self.word2idx['<PAD>']] * (max_length - len(tokens))
        return tokens

    def detokenize(self, tokens: List[int]) -> str:
        """token序列转文本"""
        if not tokens:
            return ""
        words = []
        for token in tokens:
            if token == self.word2idx['<PAD>']:
                continue
            word = self.idx2word.get(token, '<UNK>')
            words.append(word)
        return ''.join(words)

    def batch_tokenize(self, texts: List[str], max_length: Optional[int] = None, padding: bool = True) -> List[
        List[int]]:
        """批量分词"""
        tokenized = [self.tokenize(text, max_length) for text in texts]
        if max_length and padding:
            tokenized = [
                t[:max_length] if len(t) >= max_length
                else t + [self.word2idx['<PAD>']] * (max_length - len(t))
                for t in tokenized
            ]
        return tokenized


class BrainTextModel(nn.Module):
    """
    类脑文本生成模型，结合LSTM和脉冲神经网络

    参数:
        vocab_size: 词汇表大小
        embedding_dim: 词嵌入维度
        hidden_dim: LSTM隐藏层维度
        concept_dim: 概念空间维度
        memory_size: 记忆缓冲区大小
        max_length: 最大序列长度
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128,
                 concept_dim: int = 64, memory_size: int = 50, max_length: int = 50):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.concept_dim = concept_dim
        self.max_length = max_length
        self.input_length = max_length - 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=1, dropout=0.1)
        self.snn = LeakyIntegrateFire(hidden_dim, hidden_dim)
        self.concept_projection = nn.Linear(hidden_dim, concept_dim)
        self.modulator = NeuroModulator()
        self.memory = deque(maxlen=memory_size)
        self.output_layer = nn.Linear(concept_dim, vocab_size)
        self.adaptation_layer = nn.Linear(concept_dim, embedding_dim)
        self.generator_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=1, dropout=0.1)

    def forward(self, x: torch.Tensor, num_steps: int = 5, return_sequence: bool = False) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入token序列 (batch_size, seq_len)
            num_steps: SNN模拟步数
            return_sequence: 是否返回完整序列

        返回:
            概念空间表示
        """
        batch_size, seq_len = x.size()
        x_emb = self.embedding(x)
        lstm_out, _ = self.lstm(x_emb)

        if return_sequence:
            batch_seq_out = lstm_out.reshape(-1, self.hidden_dim)
            spikes, _ = self.snn(batch_seq_out, num_steps=num_steps)
            spike_rate = spikes.mean(dim=1)
            spike_rates = spike_rate.reshape(batch_size, seq_len, self.hidden_dim)
            concepts = self.concept_projection(spike_rates)
            return concepts
        else:
            last_out = lstm_out[:, -1, :]
            spikes, mem = self.snn(last_out, num_steps=num_steps)
            spike_rate = spikes.mean(dim=1)
            concept = self.concept_projection(spike_rate)
            return concept

    def generate_concept(self, input_text: str, tokenizer: ImprovedTokenizer,
                         creativity_level: float = 1.0, num_steps: int = 5) -> torch.Tensor:
        """
        生成概念向量

        参数:
            input_text: 输入文本
            tokenizer: 分词器实例
            creativity_level: 创造力水平
            num_steps: SNN模拟步数

        返回:
            概念空间向量
        """
        self.eval()
        tokens = tokenizer.tokenize(input_text, self.max_length)
        input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            seq_concepts = self.forward(input_tensor, num_steps=num_steps, return_sequence=True).squeeze(0)
            weights = torch.exp(torch.linspace(-1, 0, seq_concepts.size(0))).to(seq_concepts.device)
            weights = weights / weights.sum()
            base_concept = (seq_concepts * weights.unsqueeze(1)).sum(dim=0)

        noise_intensity = self.modulator.get_noise_intensity(creativity_level)
        noise = torch.randn_like(base_concept) * noise_intensity
        memory_influence = torch.zeros_like(base_concept)

        if self.memory:
            num_memories = min(3, len(self.memory))
            selected_memories = random.sample(self.memory, num_memories)
            similarities = [
                torch.nn.functional.cosine_similarity(base_concept.unsqueeze(0), mem.unsqueeze(0)).item()
                for mem in selected_memories
            ]
            memory_weights = torch.softmax(torch.tensor(similarities) + 0.1, dim=0).to(base_concept.device)
            for i, mem in enumerate(selected_memories):
                memory_influence += memory_weights[i] * mem

        levy_step = self.levy_flight(base_concept.size(), creativity_level)
        new_concept = base_concept + noise * 0.3 + memory_influence * 0.2 + levy_step * 0.1
        self.memory.append(new_concept.detach())
        return new_concept

    def levy_flight(self, size: Tuple[int], intensity: float = 1.0) -> torch.Tensor:
        """
        列维飞行随机游走

        参数:
            size: 输出张量形状
            intensity: 强度因子

        返回:
            随机步长张量
        """
        alpha = 1.5
        sigma = (math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
                 (math.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (1 / alpha)
        u = torch.randn(size) * sigma
        v = torch.randn(size)
        step = u / torch.abs(v) ** (1 / alpha)
        return intensity * 0.1 * step.to(device)

    def concept_to_text(self, concept: torch.Tensor, tokenizer: ImprovedTokenizer,
                        max_length: int = 30, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        概念向量转文本 - 改进版，加入重复惩罚和Top-p采样

        参数:
            concept: 概念向量
            tokenizer: 分词器实例
            max_length: 最大生成长度
            temperature: 采样温度 (0.5-1.5)
            top_p: Top-p采样阈值 (0.7-0.95)

        返回:
            生成文本
        """
        self.eval()
        start_token = tokenizer.word2idx.get('<PAD>', 0)
        current_token = torch.tensor([[start_token]], dtype=torch.long).to(device)
        generated_tokens = []
        adapted_concept = self.adaptation_layer(concept).unsqueeze(0)
        lstm_hidden = (torch.zeros(1, 1, self.hidden_dim).to(device),
                       torch.zeros(1, 1, self.hidden_dim).to(device))

        unk_idx = tokenizer.word2idx['<UNK>']
        pad_idx = tokenizer.word2idx['<PAD>']
        punctuation = {tokenizer.word2idx.get(c, -1) for c in ['。', '！', '？', '.', '!', '?', ';', '；']}
        consecutive_unk = 0

        # 应用重复惩罚函数
        def apply_repetition_penalty(logits, generated_tokens, penalty=1.2):
            """
            应用重复惩罚，降低已生成token的概率
            """
            # 只考虑最近生成的token以避免过度惩罚
            recent_tokens = generated_tokens[-10:] if len(generated_tokens) > 10 else generated_tokens
            for token in set(recent_tokens):
                logits[0, token] /= penalty
            return logits

        for _ in range(max_length):
            emb = self.embedding(current_token)
            combined = emb + adapted_concept.unsqueeze(1) * 0.5
            lstm_out, lstm_hidden = self.generator_lstm(combined, lstm_hidden)
            logits = self.output_layer(self.concept_projection(lstm_out.squeeze(1)))

            # 应用重复惩罚
            logits = apply_repetition_penalty(logits, generated_tokens, penalty=1.2)

            # 降低未知词和填充词的概率
            logits[:, unk_idx] *= 0.01
            logits[:, pad_idx] = -float('inf')

            # 应用温度调整
            logits = logits / temperature

            # 使用Top-p采样
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # 移除累积概率大于top_p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0

            # 重新归一化概率
            probs = probs / probs.sum(dim=-1, keepdim=True)

            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)

            next_token_val = next_token.item()
            generated_tokens.append(next_token_val)

            if next_token_val == unk_idx:
                consecutive_unk += 1
                if consecutive_unk >= 2:
                    break
            else:
                consecutive_unk = 0

            current_token = next_token

            # 提前终止条件：如果生成了标点符号，有50%的概率提前结束
            if len(generated_tokens) > max_length // 2 and next_token_val in punctuation:
                if random.random() < 0.5:
                    break

        # 清理生成结果：移除末尾的未知词、填充词和标点符号
        while generated_tokens and (generated_tokens[-1] in [unk_idx, pad_idx] or generated_tokens[-1] in punctuation):
            generated_tokens.pop()

        return tokenizer.detokenize(generated_tokens)

    def calculate_novelty(self, concept: torch.Tensor) -> float:
        """
        计算概念新颖性

        参数:
            concept: 概念向量

        返回:
            新颖性评分 (0.0-1.0)
        """
        if not self.memory:
            return 0.5
        similarities = [
            torch.nn.functional.cosine_similarity(concept.unsqueeze(0), mem.unsqueeze(0)).item()
            for mem in self.memory
        ]
        avg_similarity = np.mean(similarities)
        return 1 - avg_similarity

    def train_model(self, dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None,
                    epochs: int = 5, lr: float = 0.001, patience: int = 2):
        """
        模型训练

        参数:
            dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            patience: 早停耐心值
        """
        device = next(self.parameters()).device
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5)
        best_val_loss = float('inf')
        patience_counter = 0

        # 修复：添加训练模式标记
        self.optimizer = optimizer

        for epoch in range(epochs):
            # 确保模型处于训练模式
            self.train()  # 添加这行关键代码

            total_loss = 0
            accumulation_steps = 4
            optimizer.zero_grad()

            for i, (inputs, targets) in enumerate(dataloader):
                assert inputs.size(1) == self.input_length, f"输入序列长度错误: {inputs.size(1)}≠{self.input_length}"
                assert targets.size(1) == self.input_length, f"目标序列长度错误: {targets.size(1)}≠{self.input_length}"

                inputs = inputs.to(device)
                targets = targets.to(device)
                batch_size, seq_len = inputs.size()

                concepts = self.forward(inputs, return_sequence=True)
                adapted = self.adaptation_layer(concepts)
                lstm_out, _ = self.generator_lstm(adapted)
                logits = self.output_layer(self.concept_projection(lstm_out))
                loss = criterion(logits.transpose(1, 2), targets) / accumulation_steps

                total_loss += loss.item() * accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            avg_loss = total_loss / len(dataloader)
            val_loss = 0

            if val_dataloader:
                self.eval()
                with torch.no_grad():
                    for inputs, targets in val_dataloader:
                        assert inputs.size(
                            1) == self.input_length, f"验证输入长度错误: {inputs.size(1)}≠{self.input_length}"
                        assert targets.size(
                            1) == self.input_length, f"验证目标长度错误: {targets.size(1)}≠{self.input_length}"

                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        concepts = self.forward(inputs, return_sequence=True)
                        adapted = self.adaptation_layer(concepts)
                        lstm_out, _ = self.generator_lstm(adapted)
                        logits = self.output_layer(self.concept_projection(lstm_out))
                        loss = criterion(logits.transpose(1, 2), targets)
                        val_loss += loss.item()
                val_loss /= len(val_dataloader)
                logging.info(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}, 验证损失: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"验证损失连续 {patience} 次未改善，提前停止训练")
                        break
            else:
                logging.info(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

            scheduler.step(avg_loss if not val_dataloader else val_loss)
            logging.info(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")


# -------------------------- 5. Gradio界面（修复tqdm进度条步数获取） --------------------------
def build_interface():
    logger.info("开始构建界面...")
    processor = DataProcessor()
    tokenizer = None
    model = None

    def upload_zip_file(file, progress=gr.Progress()):
        nonlocal processor
        if file is None:
            return "请上传包含TXT文件的ZIP包（支持任意大小）"
        try:
            processor.set_progress_callback(lambda p, d: progress(p, desc=d))
            extract_dir, txt_files = processor.unzip_file(file.name)
            return f"ZIP处理成功！\n解压目录: {extract_dir}\n包含 {len(txt_files)} 个TXT文件"
        except Exception as e:
            return f"ZIP处理失败: {str(e)}"

    def prepare_data(custom_dict_file, progress=gr.Progress()):
        nonlocal processor, tokenizer
        if not processor.extracted_dir:
            return "请先上传并处理ZIP文件"
        try:
            texts = processor.load_from_extracted(min_para_length=20)
            custom_dict_path = custom_dict_file.name if custom_dict_file else None
            tokenizer = ImprovedTokenizer(
                max_vocab_size=10000,
                custom_dict=custom_dict_path,
                min_word_freq=1
            )
            tokenizer.set_progress_callback(lambda p, d: progress(p, desc=d))
            tokenizer.fit(texts)
            return f"数据准备完成！\n有效文本段数: {len(texts)}\n词汇表大小: {tokenizer.vocab_size}"
        except Exception as e:
            logger.error(f"数据准备失败: {str(e)}")
            return f"数据准备失败: {str(e)}"

    def train_model_interface(epochs, embedding_dim, hidden_dim, max_length, progress=gr.Progress()):
        nonlocal tokenizer, model, processor
        if not tokenizer:
            return "请先上传ZIP并准备数据"
        try:
            progress(0, desc="准备训练数据...")
            texts = processor.load_from_extracted()
            dataset = TextDataset(texts, tokenizer, max_length=int(max_length))
            if len(dataset) < 5:
                return f"训练数据不足（仅 {len(dataset)} 个样本），至少需要5个样本"

            test_size = 0.2 if len(dataset) >= 10 else 0.1 if len(dataset) >= 5 else 0
            val_loader = None
            if test_size > 0:
                train_dataset, val_dataset = train_test_split(dataset, test_size=test_size, random_state=42)
                val_loader = DataLoader(val_dataset, batch_size=min(8, len(val_dataset)), shuffle=False)
            else:
                train_dataset = dataset

            batch_size = min(16, len(train_dataset)) if len(train_dataset) >= 16 else min(4, len(train_dataset))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            progress(0.3, desc="初始化模型...")
            model = BrainTextModel(
                vocab_size=tokenizer.vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                max_length=int(max_length)
            ).to(device)

            progress(0.5, desc="开始训练模型...")
            original_tqdm_update = tqdm.update

            # -------------------------- 修复：通过进度条实例获取已迭代步数 --------------------------
            def tqdm_update_wrapper(*args, **kwargs):
                # 调用原始update方法（保留所有参数）
                original_tqdm_update(*args, **kwargs)
                # 通过进度条实例的n属性获取已迭代步数（tqdm内部维护）
                current_step = progress_bar.n
                total_steps = len(train_loader)
                # 计算进度（确保不超过0.99）
                current_progress = 0.5 + (current_step / total_steps) * 0.5
                progress(min(current_progress, 0.99), desc=f"训练中: {current_step}/{total_steps} 步")

            # 创建tqdm进度条实例并保存引用
            progress_bar = tqdm(train_loader, desc="训练进度")
            tqdm.update = tqdm_update_wrapper  # 替换tqdm的update方法

            model.train_model(train_loader, val_loader, epochs=epochs, lr=0.001)

            # 恢复原始update方法
            tqdm.update = original_tqdm_update

            progress(1.0, desc="训练完成！")
            val_count = len(val_dataset) if 'val_dataset' in locals() else 0
            return f"模型训练完成！\n训练样本数: {len(train_dataset)}\n验证样本数: {val_count}\n序列长度: {model.input_length}\n最终学习率: {model.optimizer.param_groups[0]['lr']:.6f}"
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            return f"模型训练失败: {str(e)}"

    def generate_text_interface(input_text, creativity_level, gen_length, top_p_value):
        nonlocal tokenizer, model
        if not model or not tokenizer:
            return "请先完成数据准备和模型训练"
        if not input_text or input_text.strip() == "":
            input_text = "请输入有意义的提示文本"
        try:
            concept = model.generate_concept(
                input_text,
                tokenizer,
                creativity_level=creativity_level
            )
            novelty = model.calculate_novelty(concept)
            dopamine_level = model.modulator.update_dopamine(novelty)

            # 根据创新程度调整采样参数
            # 高创新度：更高的温度，更低的Top-p值（增加随机性）
            # 低创新度：更低的温度，更高的Top-p值（减少随机性）
            temp = 0.7 + (creativity_level - 1.0) * 0.3
            top_p = 0.95 - (creativity_level - 1.0) * 0.2  # 范围0.7-0.95

            generated_text = model.concept_to_text(
                concept,
                tokenizer,
                max_length=int(gen_length),
                temperature=temp,
                top_p=top_p_value  # 使用传入的top_p值
            )

            generated_text = generated_text.strip()
            if not generated_text:
                return "生成文本为空，请调整创新度或提示文本"
            if generated_text[-1] not in ['。', '！', '？', '.', '!', '?']:
                generated_text += '。'
            info = f"\n\n=== 生成信息 ===\n新颖性评分: {novelty:.2f}\n多巴胺水平: {dopamine_level:.2f}\n采样温度: {temp:.2f}\nTop-p值: {top_p:.2f}\n提示文本: {input_text[:50]}..."
            return generated_text + info
        except Exception as e:
            logger.error(f"文本生成失败: {str(e)}")
            return f"文本生成失败: {str(e)}"

    with gr.Blocks(title="通用类脑文本生成系统") as demo:
        gr.Markdown("# 🧠 通用类脑文本生成系统")
        gr.Markdown("修复DataLoader无n属性问题，支持任意大小文本")

        with gr.Tab("1. 上传ZIP文件"):
            zip_file = gr.File(label="上传包含TXT的ZIP包（支持任意大小）", file_types=[".zip"])
            upload_btn = gr.Button("处理ZIP文件", variant="primary")
            upload_output = gr.Textbox(label="处理结果", lines=3)

        with gr.Tab("2. 准备训练数据"):
            custom_dict = gr.File(label="可选：自定义词典（TXT格式）", file_types=[".txt"])
            prepare_btn = gr.Button("准备数据（分词+词汇表）", variant="primary")
            prepare_output = gr.Textbox(label="准备结果", lines=3)

        with gr.Tab("3. 训练模型"):
            with gr.Row():
                epochs = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="训练轮次")
                embedding_dim = gr.Slider(minimum=32, maximum=256, value=64, step=32, label="嵌入维度")
                hidden_dim = gr.Slider(minimum=64, maximum=512, value=128, step=64, label="隐藏层维度")
                max_length = gr.Slider(minimum=10, maximum=100, value=50, step=5, label="序列最大长度（输入+目标）")
            train_btn = gr.Button("开始训练模型", variant="primary")
            train_output = gr.Textbox(label="训练结果", lines=3)

        with gr.Tab("4. 生成文本"):
            input_text = gr.Textbox(
                label="输入提示文本（任意主题）",
                value="人工智能的发展对人类认知有何影响？",
                lines=2,
                placeholder="请输入任意主题的提示文本..."
            )
            with gr.Row():
                creativity = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="创新程度")
                gen_length = gr.Slider(minimum=30, maximum=300, value=100, step=10, label="生成文本长度")
                top_p = gr.Slider(minimum=0.7, maximum=0.95, value=0.9, step=0.05, label="Top-p采样值")
            generate_btn = gr.Button("生成文本", variant="primary")
            output_text = gr.Textbox(label="生成文本", lines=8)

        # 更新事件绑定
        generate_btn.click(generate_text_interface, inputs=[input_text, creativity, gen_length, top_p],
                           outputs=output_text)
        # 绑定ZIP上传按钮
        upload_btn.click(upload_zip_file, inputs=[zip_file], outputs=upload_output)

        # 绑定数据准备按钮
        prepare_btn.click(prepare_data, inputs=[custom_dict], outputs=prepare_output)

        # 绑定训练模型按钮
        train_btn.click(
            train_model_interface,
            inputs=[epochs, embedding_dim, hidden_dim, max_length],
            outputs=train_output
        )

    logger.info("界面构建完成")
    return demo


# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    logger.info("进入主程序执行...")
    demo = build_interface()
    logger.info("启动Gradio界面...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        debug=False,
        quiet=False,
        share=False
    )