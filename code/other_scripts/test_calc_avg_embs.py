import time
from embeddings import ElmoE

def main():
    print('Instatiating ELMo...')
    elmo = ElmoE()
    start_time = time.time()
    print('Calculate avg embeddings...')
    elmo.calc_avg_embs_per_doc('./output/dblp/', 'l')
    time_stamp = time.time()
    time_passed = time_stamp - start_time
    print(time_passed)
    exp_time = time_passed*400000
    minutes = exp_time/60
    hours = minutes/60
    days = hours/24
    print(exp_time)
    print(minutes)
    print(hours)
    print(days)


if __name__ == '__main__':
    main()