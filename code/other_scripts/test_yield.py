import time
start_time = time.time()
print(start_time)
from typing import *
stamp2 = time.time()
print(stamp2-start_time)
def f(n):
	for i in range(n):
		if i % 2 == 0:
			yield i
stamp3 = time.time()
print(stamp3-start_time)
for i in f(10):
	print(str(i)+'i')

stamp4 = time.time()
print(stamp4-start_time)
def _concat_terms(lemmas: List[str],
              term_indices: List[List[int]]
              ) -> List[str]:
	"""Concatenate multiword terms by '_'.

	Args:
	    lemmas: A list of lemmatized words
	    term_indices: A list of lists. Each list is a term and
	        contains the indices belonging to that term.
	Return:
	    List of lemmas with concatenated multiword terms.
	"""
	for ti in term_indices[::-1]:
	    joined = '_'.join(lemmas[ti[0]:ti[-1] + 1])
	    lemmas[ti[0]:ti[-1] + 1] = [joined]

	return lemmas

lemmas = ['Ich', 'gehen', 'an', 'der', 'Bahnhof']
term_indices = [[1, 2], [3, 4]]
print(_concat_terms(lemmas, term_indices))
