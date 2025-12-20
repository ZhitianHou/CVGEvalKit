import json
import os.path
import re
from sklearn.metrics import f1_score, accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from rouge_score import rouge_scorer
import sys
sys.setrecursionlimit(7000)


def evaluation(dataset_name, result_path, metrics_path, charge_only: bool = False):
    responses = []
    labels = []
    charges = []
    charge_in_responses = []

    # 正则匹配 <view> 和 <charge>
    # view_pattern = re.compile(r"<view>(.*?)</view>", re.DOTALL)
    # charge_pattern = re.compile(r"<charge>(.*?)</charge>", re.DOTALL)
    view_pattern = re.compile(
        r"<view>(.*?)(</view>|<charge>)",
        re.DOTALL
    )

    charge_pattern = re.compile(
        r"<charge>(.*?)(</charge>|$)",
        re.DOTALL
    )
    with open(result_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                resp = obj.get("response", "")
                label = obj.get("label", "")
                charge = obj.get("charge", "").strip() if obj.get("charge") else None

                # 提取 <view> 内容
                resp_view_match = view_pattern.search(resp)
                label_view_match = view_pattern.search(label)
                resp_view = resp_view_match.group(1).strip() if resp_view_match else ""
                label_view = label_view_match.group(1).strip() if label_view_match else ""

                responses.append(" ".join(list(resp_view.replace(" ", ""))) if resp_view else "我无法回答这个问题")
                labels.append(" ".join(list(label_view.replace(" ", ""))))

                # responses.append(resp_view.replace(" ", "") if resp_view else "我无法回答这个问题")
                # labels.append(label_view.replace(" ", ""))

                # charge 对比
                charges.append(charge)
                resp_charge_match = charge_pattern.search(resp)
                resp_charge = resp_charge_match.group(1).strip() if resp_charge_match else ""
                if charge and charge in resp_charge:
                    charge_in_responses.append(1)
                else:
                    charge_in_responses.append(0)

            except json.JSONDecodeError:
                continue

    if os.path.exists(metrics_path):
        mode = "a"
    else:
        mode = "w"

    def tee_print(text, file):
        print(text)          # 打印到屏幕
        file.write(text + "\n")   # 写入文件

    with open(metrics_path, mode, encoding="utf-8") as f:
        if not charge_only:
            rouge = Rouge()
            rouge_scores = rouge.get_scores(responses, labels, avg=True)

            # ========== BLEU ==========
            smoothie = SmoothingFunction().method4

            def safe_bleu(ref, hyp, n):
                try:
                    weights = tuple([1.0 / n] * n)
                    return sentence_bleu([list(ref)], list(hyp), weights=weights, smoothing_function=smoothie)
                except:
                    return 0.0

            bleu_1 = sum([safe_bleu(r, h, 1) for r, h in zip(labels, responses)]) / len(responses)
            bleu_2 = sum([safe_bleu(r, h, 2) for r, h in zip(labels, responses)]) / len(responses)
            bleu_4 = sum([safe_bleu(r, h, 4) for r, h in zip(labels, responses)]) / len(responses)

            tee_print(f"📊 {dataset_name} 文本生成指标 (ROUGE)：", f)
            tee_print(f"ROUGE-1: {rouge_scores['rouge-1']['f']:.4f}", f)
            tee_print(f"ROUGE-2: {rouge_scores['rouge-2']['f']:.4f}", f)
            tee_print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}", f)

            tee_print(f"📘 {dataset_name} 文本生成指标 (BLEU)：", f)
            tee_print(f"BLEU-1: {bleu_1:.4f}", f)
            tee_print(f"BLEU-2: {bleu_2:.4f}", f)
            tee_print(f"BLEU-4: {bleu_4:.4f}", f)

        # ========== charge F1 和 Accuracy ==========
        y_true = [1 if charge else 0 for charge in charges]
        y_pred = charge_in_responses

        charge_f1 = f1_score(y_true, y_pred, zero_division=0)
        charge_acc = accuracy_score(y_true, y_pred)

        tee_print(f"🧾 {dataset_name} charge 标签评估：", f)
        tee_print(f"Accuracy: {charge_acc:.4f}", f)
        tee_print(f"F1 Score: {charge_f1:.4f}", f)

