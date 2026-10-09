import re


if __name__=='__main__':
    url=input()
    prefix='https://www'
    pattern='^(https?://(www)?)?'
    match_len=0
    for i in range(len(prefix)+1):
        current_pattern='^'+re.escape(prefix[:i])
        match=re.match(current_pattern,url)
        if match:
            match_len=i
        else:
            break
    print(f'(0, {match_len})')
