import sys
from collections import defaultdict

texts, cnt = defaultdict(int), 0
while True:
    text = sys.stdin.readline().rstrip()
    if not text: break
    texts[text] += 1
    cnt += 1
    
species = sorted(texts.items())

for sort, num in species:
    print(f'{sort} {num/cnt*100:.4f}')

'''
import sys

texts = []
while True:
    text = sys.stdin.readline().rstrip()
    if not text: break
    texts.append(text)
    
lentexts = len(texts)
species = sorted(set(texts))

for s in species:
    print(f'{s} {texts.count(s)/lentexts*100:.4f}')
'''