import numpy as np
from .model import Model

class Solver(object):
    def __init__(self, data, model, metric=None, bigger_is_better=True, **kwargs):
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.model = model
        self.metric = metric
        self.batch_size = kwargs.pop('batch_size',100)
        self.iterations_per_epoch = max(self.X_train.shape[0] // self.batch_size, 1)
        self.train_subsample_size = kwargs.pop('train_subsample_size',1000)
        self.batch_report_every = kwargs.pop('batch_report_every',10)
        self.verbose = kwargs.pop('verbose',True)
        self.order = 2*int(bigger_is_better)-1
        self.best_result = None
        self.best_settings = None
        self.history = {}
            
    def _sample(self,X,y,sample_size):
        indices = np.random.choice(X.shape[0],sample_size)
        X_sample = X[indices]
        y_sample = y[indices]
        return X_sample,y_sample

    def reset(self):
        self.model.reset()
        self.history = {}
        self.best_result = None
        self.best_settings = None
    
    def train(self,num_epochs=10,**kwargs_train):
        num_iterations = num_epochs*self.iterations_per_epoch
        for it in range(num_iterations+1):
            X_batch, y_batch = self._sample(self.X_train,self.y_train,self.batch_size)
            batch_report = self.model.train_step(X_batch,y_batch,**kwargs_train)
            if it % self.batch_report_every == 0 and self.verbose:
                print("(Iteration {0}/{1})".format(it,num_iterations),", ".join([a+':'+str(round(b,4)) for a,b in batch_report]))
            if it % self.iterations_per_epoch == 0:
                X_tr_sample, y_tr_sample = self._sample(self.X_train,self.y_train,self.train_subsample_size)
                epoch_report = self.model.epoch_step(X_tr_sample,y_tr_sample,self.X_val,self.y_val)
                for a,b in epoch_report:
                    self.history.setdefault(a,list()).append(b)
                if self.metric is not None and (self.best_result is None or self.order*(self.history[self.metric][-1]-self.best_result[self.metric])>=0):
                    self.best_result = {h:self.history[h][-1] for h in self.history}
                    self.best_settings = self.model.save()
                    print("Improved",self.metric+':',self.best_result[self.metric])
                if self.verbose:
                    print("(Epoch {0}/{1})".format(it//self.iterations_per_epoch,num_epochs),", ".join([a+':'+str(round(b,4)) for a,b in epoch_report]))
        if self.metric is not None:
            self.model.load(self.best_settings)
            for h in self.history:
                self.history[h][-1]=self.best_result[h]
