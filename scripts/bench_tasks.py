"""Decode speed by task type against any OpenAI-compatible endpoint: chat, code, math, 5 prompts each, greedy.

Speculative decoding depends heavily on how predictable the text is: on this setup MTP yields 3.3 tokens per verify
window on code and math but 2.1 on chat, so measuring one kind of prompt misjudges the speed by 25% or more.

    python bench_tasks.py http://127.0.0.1:20062 [tag]

tokens/window = generated / (generated - accepted drafts), from the server's timings (Strata and llama-server both
report draft_n / draft_n_accepted).
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1].rstrip('/') + '/v1/chat/completions'
TAG = sys.argv[2] if len(sys.argv) > 2 else 'run'
CATS = {
    'chat': [
        '請用繁體中文介紹台灣的夜市文化、代表小吃與歷史，至少寫 300 字。',
        'Explain how a refrigerator works to a curious 12-year-old, with an everyday analogy.',
        '寫一封婉拒工作邀約的信，語氣誠懇、感謝對方，並保留未來合作的可能。',
        'What are the trade-offs between living in a big city and a small town? Give a balanced view.',
        '用三個段落說明為什麼睡眠對學習很重要，並給出具體建議。',
    ],
    'code': [
        '用 Python 實作一個 LRU cache 類別，支援 get 與 put，兩者都要 O(1)，並附上 pytest 單元測試。',
        'Write a C function that parses an IPv4 address string into a uint32_t with full error handling, then list the edge cases it covers.',
        '用 TypeScript 寫 debounce 與 throttle 兩個函式，含完整型別定義、取消功能與使用範例。',
        'Implement generic quicksort and mergesort in Rust, with unit tests and a short comparison of their complexity.',
        '寫一個 bash 腳本：遞迴找出目錄中超過 100MB 的檔案，依大小由大到小輸出，並支援以參數排除特定路徑。',
    ],
    'math': [
        '小明有 3 盒蘋果，每盒 12 顆。他送出全部的四分之一後，又買了 15 顆。現在有幾顆？請逐步計算。',
        'A train travels 180 km at 60 km/h, then 120 km at 80 km/h. What is its average speed for the whole trip? Show every step.',
        '解方程組：2x + 3y = 13，5x - y = 7。請寫出完整步驟並驗算。',
        '一件商品先打八折，再減 50 元，最後售價 350 元。原價是多少？請逐步說明。',
        'Compute the sum of all integers from 1 to 1000 that are divisible by 3 or 5. Show the reasoning.',
    ],
}


def ask(content, n):
    body = json.dumps({'model': 'x', 'messages': [{'role': 'user', 'content': content}], 'max_tokens': n,
                       'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=3600)).get('timings', {})


ask(CATS['chat'][0], 200)   # warm up
for cat, prompts in CATS.items():
    n = a = 0
    speeds = []
    for q in prompts:
        t = ask(q, 400)
        n += t.get('predicted_n') or 0
        a += t.get('draft_n_accepted') or 0
        speeds.append(t.get('predicted_per_second') or 0)
    tpw = n / (n - a) if n > a else 0
    print(f"{TAG}\t{cat}\tgenerated={n}\taccepted={a}\ttokens/window={tpw:.2f}\tdecode={sum(speeds) / len(speeds):.1f} tok/s",
          flush=True)
