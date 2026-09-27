import sys
import math

def solve(sentences,query):
    st_idx=query[0]
    target_word=query[1]
    target_sentence=sentences[st_idx].split()
    word_cnt=target_sentence.count(target_word)
    total_words=len(target_sentence)
    tf=word_cnt/total_words

    N=len(sentences)
    Nw=0
    for st in sentences:
        words=list(st.split())
        if target_word in words:
            Nw+=1
    idf=math.log(N/(Nw+1)+1)
    # print(f'[debug] tf: {tf}, idf: {idf}')
    return tf*idf

if __name__=='__main__':
    sentences=eval(input())
    query=eval(input())
    tf_idf=solve(sentences,query)
    print(f'{tf_idf:.3f}')

