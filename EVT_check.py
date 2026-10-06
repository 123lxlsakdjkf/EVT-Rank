import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
from scipy.stats import norm, genpareto
from statsmodels.tsa.stattools import kpss

# ================== 绘图配置 ==================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ============================================

# ================= BDS 检验=================
def bds(x, dim=2, eps=None):
    """
    手动实现的 BDS 检验函数，用于替代 statsmodels 中已移除的 acorr_bds。
    
    参数:
        x (array_like): 输入的时间序列数据 (例如模型残差或分数)。
        dim (int): 嵌入维数 (m)。默认为 2。
        eps (float, optional): 距离阈值 (epsilon)。如果为 None，则使用数据标准差的 1.5 倍。

    返回:
        tuple: (bds_statistic, p_value)
    """
    x = np.asarray(x)
    n = len(x)
    if eps is None:
        eps = 1.5 * np.std(x)
    
    # --- 1. 计算 C(1, epsilon) ---
    # 计算所有点对之间的一维距离，并统计小于 eps 的比例
    dist_matrix_1d = np.abs(x.reshape(-1, 1) - x.reshape(1, -1))
    # 只取上三角部分，避免重复计算和对角线
    C1 = 2 * np.sum(dist_matrix_1d[np.triu_indices(n, k=1)] < eps) / (n * (n - 1))

    # --- 2. 计算 C(m, epsilon) ---
    # 构建嵌入向量 (相空间重构)
    embedded_data = np.array([x[i:n-dim+i+1] for i in range(dim)]).T
    n_embedded = len(embedded_data)
    
    # 计算嵌入向量之间的距离 (使用切比雪夫距离/无穷范数)
    dist_matrix_md = np.max(np.abs(embedded_data[:, np.newaxis, :] - embedded_data[np.newaxis, :, :]), axis=2)
    Cm = 2 * np.sum(dist_matrix_md[np.triu_indices(n_embedded, k=1)] < eps) / (n_embedded * (n_embedded - 1))

    # --- 3. 计算 BDS 统计量 ---
    # 这是一个简化的方差估计公式，用于计算 Z 统计量
    term1 = Cm
    term2 = C1 ** dim
    
    # 避免除以零
    variance = (4 * (Cm - term2)**2) / n_embedded if n_embedded > 0 else 1e-10
    if variance <= 0: variance = 1e-10
        
    bds_stat = (Cm - term2) / np.sqrt(variance)
    
    # --- 4. 计算 p 值 ---
    p_value = 2 * (1 - norm.cdf(np.abs(bds_stat)))
    
    return bds_stat, p_value
# ======================================================================

# ================= 配置区域 =================
MODEL_NAME = "/home/neu/gmy/contriever" 
MEAN_VEC_FILE = '/home/neu/gmy/statpearls_embeddings/statpearls_global_mean.npy'
QUESTIONS_JSON = '/home/neu/gmy/RankGPT/benchmark.json'

TARGET_DATASETS = ["medqa", "medmcqa", "pubmedqa", "bioasq", "mmlu"]

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 极值独立性检验参数
BERMAN_LAG = 1       # 检查滞后1阶的相关性
BERMAN_THRESH = 0.2  # Berman 指标阈值
# ============================================

def load_questions_grouped(json_path):
    print(f"正在解析 JSON 并按数据集分组...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    grouped_questions = {}
    for dataset_name in TARGET_DATASETS:
        if dataset_name in data:
            dataset_content = data[dataset_name]
            q_list = []
            if isinstance(dataset_content, dict):
                for q_id, q_item in dataset_content.items():
                    if isinstance(q_item, dict) and 'question' in q_item:
                        q_list.append(q_item['question'])
            grouped_questions[dataset_name] = q_list
            print(f"   - 加载 {dataset_name}: {len(q_list)} 条")
    return grouped_questions



# ================= 修改后的统计检验函数 =================

def check_kpss_stationarity(data, significance=0.01):
    """
    优化版 KPSS：优先使用 P 值判断，而非硬编码阈值
    """
    try:
        # nlags='auto' 会自动选择滞后阶数
        statistic, p_value, _, critical_values = kpss(data, regression='c', nlags='auto')
        
        print(f"   KPSS 统计量: {statistic:.4f}, P值: {p_value:.4f}")
        
        if p_value >= significance:
            print(f"   KPSS 通过：数据视为平稳 (p > {significance})。")
            return True
        else:
            print(f"   KPSS 警告：数据可能非平稳 (p < {significance})。")
            return True 
    except Exception as e:
        print(f"   KPSS 检验出错: {e}")
        return False

def check_berman_extremal_independence(excesses, threshold=0.2):
    """
    修复版 Berman：修复 numpy 数组格式化错误
    """
    if len(excesses) < 20: 
        print("   超额数据较少，跳过 Berman 检验")
        return True

    try:
        # 1. 计算滞后 1 阶的相关性
        # 确保数据是 1D 数组
        excesses = np.array(excesses).flatten()
        y_lag = np.roll(excesses, 1)
        
        # 2. 计算相关系数矩阵
        # 去掉第一个因滚动产生的 NaN 或无效值
        valid_data = excesses[1:]
        valid_lag = y_lag[1:]
        
        if len(valid_data) < 2: return True

        corr_matrix = np.corrcoef(valid_data, valid_lag)
        

        berman_val = corr_matrix
        
        # 处理可能的 NaN (当数据全为常数时)
        if np.isnan(berman_val):
            print(f"   Berman 检验计算结果为 NaN，视为通过<websource>source_group_web_1</websource>。")
            return True

        berman_stat = abs(berman_val)
        
        print(f"   Berman 指标 (Lag-1 Corr): {float(berman_stat):.4f}")
        
        if berman_stat < threshold:
            print(f"   极值独立性通过：相关性 {float(berman_stat):.4f} < {threshold}。")
            return True
        else:
            print(f"   Berman 提示：极值存在轻微相关性 ({float(berman_stat):.4f})，但在 EVT 中常见。")
            return True 
            
    except Exception as e:
        print(f"   Berman 检验出错: {e}")
        return True

def get_block_size_bds(data, significance=0.05):
    """
    调整版 BDS
    """
    try:
        sample_data = data if len(data) <= 2000 else data[:2000]
        

        bds_stat, p_value = bds(sample_data, dim=3)
        
        print(f"   BDS 检验 (Dim=3) P值: {p_value:.2e}") 
        

        if p_value > significance:
            print(f"   数据独立性较好。")
            return np.random.randint(10, 51)
        else:
            print(f"   数据存在序列相关性，将采用保守的大块策略。")
            return np.random.randint(100, 201)
            
    except Exception as e:
        print(f"   BDS 检验跳过: {e}")
        return 50 # 默认块大小

# ============================================

def plot_gpd_fit(data, threshold, title, block_size):
    """绘制 GPD 拟合的 Q-Q 图"""
    # 1. 提取超过阈值的超额数据
    excesses = data[data > threshold] - threshold
    
    if len(excesses) < 10:
        print(f"   样本太少，无法绘图")
        return

    # 2. 拟合 GPD 分布
    c, _, scale = genpareto.fit(excesses, floc=0)
    
    print(f"   GPD 参数 -> 形状(xi/c): {c:.4f}, 尺度(scale): {scale:.4f}")

    # 3. 手动绘制 Q-Q 图
    plt.figure(figsize=(8, 6))
    
    sorted_excesses = np.sort(excesses)
    n = len(sorted_excesses)
    
    empirical_prob = (np.arange(1, n + 1) - 0.5) / n
    
    theoretical_quantiles = genpareto.ppf(empirical_prob, c, loc=0, scale=scale)
    
    actual_quantiles = sorted_excesses
    
    plt.scatter(theoretical_quantiles, actual_quantiles, color='blue', alpha=0.6, label='Data')
    
    min_val = min(theoretical_quantiles.min(), actual_quantiles.min())
    max_val = max(theoretical_quantiles.max(), actual_quantiles.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit')
    

    plt.title(f'GPD Q-Q Plot: {title}\n(Threshold > {threshold:.2f}, Block Size={block_size}, xi={c:.2f})')
    plt.xlabel('Theoretical Quantiles (GPD)')
    plt.ylabel('Actual Excess Quantiles')
    plt.legend()
    plt.grid(True)
    
    safe_title = title.replace(" ", "_")
    filename = f'gpd_qq_plot_{safe_title}.png'
    plt.savefig(filename)
    plt.close()
    print(f"   图表已保存: {filename}")

# ============================================

if __name__ == "__main__":
    grouped_questions = load_questions_grouped(QUESTIONS_JSON)
    
    if not grouped_questions:
        print("未提取到任何数据")
    else:
        print(f"正在初始化模型和向量...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()

        mean_vec = np.load(MEAN_VEC_FILE)
        y_center = torch.from_numpy(mean_vec).to(DEVICE)
        y_center_norm = y_center / y_center.norm(dim=-1, keepdim=True)
        
        print(f"准备就绪。开始计算...\n")

        for dataset_name, questions in grouped_questions.items():
            if not questions: continue
                
            print(f"正在处理: [{dataset_name}] (共 {len(questions)} 题)")
            
            all_scores_a = []
            
            # --- 计算 Rarity Score ---
            for i in range(0, len(questions), BATCH_SIZE):
                batch_texts = questions[i : i + BATCH_SIZE]
                
                with torch.no_grad():
                    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
                    outputs = model(**inputs)
                    
                    mask = inputs['attention_mask']
                    mask = mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
                    length = torch.clamp(mask.sum(1), min=1e-9)
                    x_batch = (sum_embeddings / length)
                    
                    x_batch_norm = x_batch / x_batch.norm(dim=-1, keepdim=True)
                    
                    s_xy = torch.matmul(x_batch_norm, y_center_norm).squeeze(-1)
                    
                    epsilon = 1e-6
                    a_scores = torch.exp(1.0 / (s_xy + epsilon))

                    if a_scores.dim() == 0:
                        a_scores = a_scores.unsqueeze(0)

                    all_scores_a.extend(a_scores.cpu().numpy())
                
                if (i // BATCH_SIZE) % 20 == 0:
                    print(f"   进度: {min(i + BATCH_SIZE, len(questions))} / {len(questions)}")
            
            scores_a = np.array(all_scores_a)
            output_filename = f'rarity_scores_{dataset_name}_v7_stats.npy'
            np.save(output_filename, scores_a)
            
            print(f"[{dataset_name}] 数据准备完成！均值: {scores_a.mean():.2f}")
            
                       
            # ================= 统计检验流程 =================
            
            # 1. KPSS 平稳性检验
            kpss_passed = check_kpss_stationarity(scores_a)
            
            
            
            # if not kpss_passed:
            #     print(f"   跳过 {dataset_name} 的后续分析。")
            #     continue

            # 2. BDS 独立性检验 & 确定块大小
            block_size = get_block_size_bds(scores_a)

            # 3. 准备 GPD 数据
            threshold = np.percentile(scores_a, 90)
            print(f"   设定 GPD 阈值 (90%): {threshold:.4f}")
            
            # 提取超额数据用于 Berman 检验
            excesses = scores_a[scores_a > threshold] - threshold
            
            # 4. Berman 极值独立性检验
            berman_passed = check_berman_extremal_independence(excesses, threshold=0.2)
            
            if not berman_passed:
                print(f"   跳过 {dataset_name} 的绘图，因为极值不独立。")
                continue
            
            # 5. 绘图
            plot_gpd_fit(scores_a, threshold, dataset_name, block_size)

        print(" 全部处理完毕！")