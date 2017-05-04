class Model():
    def __init__(self,train_step,epoch_step,predict,reset,save,load):
        self.train_step = train_step
        self.epoch_step = epoch_step
        self.predict = predict
        self.reset = reset
        self.save = save
        self.load = load
