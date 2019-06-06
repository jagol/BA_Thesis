import multiprocessing as mp
import random
import string

random.seed(123)

# Define an output queue
output = mp.Queue()

# define a example function
def rand_string(length, output):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    rand_str = ''.join(random.choice(
                        string.ascii_lowercase
                        + string.ascii_uppercase
                        + string.digits)
                   for i in range(length))
    output.put(rand_str)

def fac(n, m, o):
    x = n*m*o
    with open(str(n)+str(m)+str(o)+'.txt', 'w', encoding='utf8') as f:
        print(x)
        f.write(str(x))
    output.put('w')

# Setup a list of processes that we want to run
processes = [mp.Process(target=fac, args=(x)) for x in [(2, 3, 4), (4, 6, 8), (7, 9, 13)]]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]

print(results)
