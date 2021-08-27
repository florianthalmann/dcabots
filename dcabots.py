import os, dateutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from joblib import parallel_backend
import optuna

def load_opens(file, start_date, end_date):
    data = pd.read_csv(file)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date')
    if len(data[data['date'] == start_date]) == 1:
        return data['open'][(start_date <= data['date']) & (data['date'] <= end_date)].to_numpy()

def load_data(path, start_date, end_date):
    filenames = [os.path.join(path, f)
        for f in os.listdir(path) if 'DS_Store' not in f]
    data = [load_opens(f, start_date, end_date) for f in filenames]
    return zip(*[(d, filenames[i]) for i,d in enumerate(data) if d is not None])

class Bot:
    #take profit, base, safety, volume scale, step scale, so dev, max so count
    #TP: 1.0%, BO: 10, SO: 20, OS: 1.05, SS: 1.0, SOS: 2.5, MSTC: 20
    #TP: 1.0%, BO: 10, SO: 20, OS: 2.0, SS: 2.0, SOS: 2.0, MSTC: 5
    def __init__(self, tp, bo, so, os, ss, sos, mstc):
        self.tp, self.bo = 0.01*tp, bo
        steps = np.arange(mstc, dtype='float')
        self.so_pos = -0.01*np.cumsum(sos*(ss**steps))
        self.so_fiat = so*(os**steps)
        self.started = False
    
    def start(self, price, stock):
        if self.bo+self.so_fiat[0] <= stock:
            self.started = True
            self.avg_price = price
            self.next_so = 0
            self.stuck = False
            #print('bo', self.bo, self.so_fiat[0], self.avg_price, stock-self.locked_amount())
            return stock-self.locked_amount()
        return stock
    
    def step(self, price, stock):
        profit = 0
        if not self.started:
            return self.start(price, stock), profit
        deviation = (price / self.avg_price)-1
        if deviation >= self.tp:
            profit = self.current_investment() * deviation#self.tp
            stock += self.locked_amount()
            #print('profit', deviation, profit, stock)
            stock = self.start(price, stock)
        elif not self.stuck and self.next_so < len(self.so_pos) and deviation <= self.so_pos[self.next_so]:
            if self.next_so == len(self.so_pos)-1 or self.so_fiat[self.next_so+1] <= stock:
                self.avg_price = ((self.current_investment() * self.avg_price) \
                    + (self.so_fiat[self.next_so] * price)) \
                    / (self.current_investment() + self.so_fiat[self.next_so])
                if self.next_so < len(self.so_fiat)-1:
                    stock -= self.so_fiat[self.next_so+1]
                    #print(self.so_fiat[self.next_so+1])
                #print('so', deviation, self.avg_price, stock)
                self.next_so += 1
            else:
                self.stuck = True
                #print('stuck', self.avg_price)
        #else: print('...', price, deviation)
        return stock, profit
    
    def current_investment(self):
        return self.bo+sum(self.so_fiat[:self.next_so]) if self.started else 0
    
    def open_order(self):
        if self.started and self.next_so < len(self.so_fiat):
            return self.so_fiat[self.next_so]
        return 0
    
    def locked_amount(self):
        return self.current_investment()+self.open_order()
    
    def close_out(self, price):
        if self.started:
            #print('close', self.current_investment(), (price / self.avg_price), self.open_order())
            return self.current_investment() * (price / self.avg_price) + self.open_order()
        return 0

def run_multi_bots(capital, data, tp, bo, so, os, ss, sos, mstc, profit_to_stock, close_out):
    bots = [Bot(tp, bo, so, os, ss, sos, mstc) for d in data]
    stock = capital
    total_profit = 0
    locked_amount = []
    num_deals = 0
    for t in range(len(data[0])):
        locked_amount.append(0)
        for i in range(len(bots)):
            stock, profit = bots[i].step(data[i][t], stock)
            if profit > 0: num_deals += 1
            if profit_to_stock:
                stock += profit
            else:
                total_profit += profit
            locked_amount[-1] += bots[i].locked_amount()
        #print(stock, locked_amount[-1])
    #print([bots[i].close_out(data[i][-1]) for i in range(len(bots))])
    if close_out:
        stock += sum([bots[i].close_out(data[i][-1]) for i in range(len(bots))])
    else:
        stock += sum([bots[i].locked_amount() for i in range(len(bots))])
    profit = stock-capital + total_profit
    return profit, np.mean(locked_amount), np.max(locked_amount), profit/np.mean(locked_amount), num_deals


#1h all '2021-04-27'
#minute '2021-06-07'
#crash '2021-05-13'
#after '2021-05-20'
data_path = 'data/usd_1m'#usd_1h_all'#'data/usd_1h_all'#'data/usd_1m'
buffer_path = 'data/1h'#'data/usd_1h_2021'#'data/usd_1m')
# data, filenames = load_data(data_path,
#    dateutil.parser.parse('2021-01-01'), dateutil.parser.parse('2021-05-12'))
# np.save(buffer_path, ([d[::1] for d in data], filenames))
data, filenames = np.load(buffer_path+'.npy', allow_pickle=True)
#data = [d[:11053] for d in data][::-1]
#print(data[0][:10], data[1][:10])
#data = [data[0][:100]]
#data = [data[i] for i in [0,2,4,5,6,9,10,13]]
#data = [data[i] for i in range(len(data)) if i not in [1,3,8,9,13,14,16,21,22]]#worst: 3,8,9,13,16,22, bad: 1,14,21
#filenames = [filenames[i] for i in range(len(filenames)) if i not in [1,3,8,9,13,14,16,21,22]]
#data, filenames = data[14:15], filenames[14:15]
# data, filenames = data[-1:], filenames[-1:]
print([len(d) for d in data])
print(filenames)

# for i in range(len(data)):
#         print(run_multi_bots(2000, data[:i+1], 1.5, 10, 10, 1.4, 1, 2.9, 25))

#avg best for 600 per bot: os 1.12, sos 2.2-2.3, 800: 1.14, 2.4
#1h_all: 600: (1, 11.2, 16.8, 2, 1, 2.5, 20) 245.17681406617635 80.39513261436494
# result = [run_multi_bots(600, [data[i]], 0.6, 10, 13, 3.8, 3, 2.7, 20, False)
#     for i in range(len(data))]
# [print(filenames[i], r) for i,r in enumerate(result)]
# print(np.max([r[0] for r in result]), np.mean([r[0] for r in result]))

def run_n_bots(capital, data, n, tp, bo, so, os, ss, sos, mstc, profit_to_stock, close_out):
    # choice = np.random.choice(len(data), n)
    # data = [data[i] for i in choice]
    #data = list(reversed(data))
    return run_multi_bots(capital, data, tp, bo, so, os, ss, sos, mstc, profit_to_stock, close_out)


#tp, bo, so, os, ss, sos, mstc
def objective(trial):
    #n = trial.suggest_int('n', 3, 10)
    c = trial.suggest_int('c', 2000, 2000, 100)
    tp = trial.suggest_float('tp', 2, 2, step=1)
    bo = trial.suggest_int('bo', 10, 10, step=5)
    so = trial.suggest_int('so', 10, 15, step=1)
    os = trial.suggest_float('os', 1, 1.2, step=0.05)
    ss = trial.suggest_float('ss', 1, 1, step=0.1)
    sos = trial.suggest_float('sos', 1, 1.2, step=0.05)
    #mstc = trial.suggest_int('mstc', 1, 3)
    # result = [run_multi_bots(1000, [data[i]], tp, bo, so, os, ss, sos, 100)
    #     for i in range(len(data[:]))]
    # result = [run_n_bots(2000, data, 100, tp, bo, so, os, ss, sos, 100, False)
    #     for i in range(len(data))]
    # return np.mean([r[0] for r in result])
    results = run_multi_bots(c, data, tp, bo, so, os, ss, sos, 30, False, False)
    return results[3] + results[1]/c
    #return run_multi_bots(2000, data, tp, bo, so, os, ss, sos, 30, False, False)[0]

study = optuna.create_study(direction='maximize')#, sampler=optuna.samplers.GridSampler())
#with parallel_backend('multiprocessing'):  # Overrides `prefer="threads"` to use multi-processing.
study.optimize(objective, n_trials=100)#, n_jobs=6)
print(study.best_params)

#optuna.visualization.plot_param_importances(study).write_image("*params.png")
#optuna.visualization.plot_contour(study, params=['bo','so']).write_image("*contour.png")
