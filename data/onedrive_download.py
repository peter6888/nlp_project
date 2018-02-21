'''
<iframe src="https://onedrive.live.com/embed?cid=06DDD24D87B0D23B&resid=6DDD24D87B0D23B%21551&authkey=AMcY3zcjDtjgLZI" width="98" height="120" frameborder="0" scrolling="no"></iframe>
<iframe src="https://onedrive.live.com/embed?cid=06DDD24D87B0D23B&resid=6DDD24D87B0D23B%21552&authkey=AB4pRDUbtaDWAKA" width="98" height="120" frameborder="0" scrolling="no"></iframe>
<iframe src="https://onedrive.live.com/embed?cid=06DDD24D87B0D23B&resid=6DDD24D87B0D23B%21554&authkey=AJBblyzOf0-3lAI" width="98" height="120" frameborder="0" scrolling="no"></iframe>
'''
'''
wget --no-check-certificate "https://onedrive.live.com/download?cid=06DDD24D87B0D23B&resid=6DDD24D87B0D23B%21489&authkey=APj2E8TI_O5xjWg"
mv "download?cid=06DDD24D87B0D23B&resid=6DDD24D87B0D23B%21489&authkey=APj2E8TI_O5xjWg" 2.264
'''
def embed2wget(url, filename):
    '''
    :param url: url string like <iframe>...cid=..." ...</iframe>
    :param filename: downloaded filename
    :return: the wget command
    '''
    s_cid = 'cid='
    c_index = url.find(s_cid)
    e_index = url.find('" width')
    cid = url[c_index+len(s_cid):e_index]
    return ['wget --no-check-certificate "https://onedrive.live.com/download?cid={}"'.format(cid), \
            'mv "download?cid={}" {}'.format(cid, filename)]

def main():
    sh_cmds = []
    with open('onedrive_share.txt', 'r') as f:
        f_lines = f.readlines()
    for i in range(0, len(f_lines), 2):
        sh_cmds.extend(embed2wget(f_lines[i+1], f_lines[i]))
    #print(sh_cmds)
    with open('download.sh', 'w') as shfile:
        shfile.write("\n".join(sh_cmds))

if __name__ == "__main__":
    main()