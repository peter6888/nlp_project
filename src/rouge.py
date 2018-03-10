import os
import sys

from decode import rouge_eval

folder = sys.argv[1]
# os.chdir(os.path.dirname(__file__))
root = os.getcwd()
print(os.path.join(root, folder))
ref_dir = os.path.join(root, folder, 'reference')
dec_dir = os.path.join(root, folder, 'decoded')
print(rouge_eval(ref_dir, dec_dir))
