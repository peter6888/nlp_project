'''
read the rouge scores and do stastics
'''
import numpy as np
def read_scores():
    '''
    read rouge scores from the file
    :return: list(list)
    '''
    rouges = [[], [], [], []]
    search = ['Rouge-1: ', 'Rouge-2: ', 'Rouge-L (Sentence Level):', 'Rouge-L (Summary Level):']
    with open("ranks.txt","r") as f:
        ranks = f.read()
    for l in ranks.split('\n'):
        for i in range(len(search)):
            if l.startswith(search[i]):
                rouges[i].append(float(l[len(search[i]):]))

    for i in range(len(rouges)):
        print("Median of {}{}".format(search[i], \
                                      np.median(np.array(rouges[i]))))

if __name__ == '__main__':
    read_scores()

#A sample output is
'''
(.env) Peters-MBP:nlp_project peli$ python rouge_score.py
Median of Rouge-1: 0.5
Median of Rouge-2: 0.313131
Median of Rouge-L (Sentence Level):0.230138
Median of Rouge-L (Summary Level):0.0201045
(.env) Peters-MBP:nlp_project peli$
'''