from os.path import exists
import shutil

abstracts = open('abstract.txt', 'r')
missing_abstracts = open('missing_abstract.txt', 'w')
lines = abstracts.readlines()
for line in lines:
	word = line.strip()
	word_file = 'sents_500/' + word + '.txt'
	if exists(word_file):
		shutil.copyfile(word_file, 'mcrae_sents/' + word + '.txt')
	else:
		missing_abstracts.write(word + '\n')
		
print('Done !')
