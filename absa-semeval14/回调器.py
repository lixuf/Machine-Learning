
class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0
    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，不warmup有不收敛的可能。
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * 1e-3
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
    def on_epoch_end(self, epoch, logs=None):
        EMAer.apply_ema_weights()
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            train_model.save_weights('best_model.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))
        EMAer.reset_old_weights()
        if epoch + 1 == 50 or (
            self.stage == 0 and epoch > 10 and
            (f1 < 0.5 or np.argmax(self.F1) < len(self.F1) - 8)
        ):
            self.stage = 1
            train_model.load_weights('best_model.weights')
            EMAer.initialize()
            K.set_value(self.model.optimizer.lr, 1e-4)
            K.set_value(self.model.optimizer.iterations, 0)
            opt_weights = K.batch_get_value(self.model.optimizer.weights)
            opt_weights = [w * 0. for w in opt_weights]
            K.batch_set_value(zip(self.model.optimizer.weights, opt_weights))
    def evaluate(self):
        orders = ['subject']
        A, B, C = 1e-10, 1e-10, 1e-10
        F = open('dev_pred.json', 'w')
        for d in test_data.keys():
            R = set(extract_items(test_data[d]["text"],test_data_un[d]["text"]))
            T=[]
            for i in test_data[d]["aspect"]:
                T.append(tuple(i[3]))
            T = set(tuple(T))
            A += len(R & T)
            B += len(R)
            C += len(T)
            s = json.dumps({
                'text': d['text'],
                'aspect': [
                    dict(zip(orders, spo)) for spo in T
                ],
                'aspect_pred': [
                    dict(zip(orders, spo)) for spo in R
                ],
                'new': [
                    dict(zip(orders, spo)) for spo in R - T
                ],
                'lack': [
                    dict(zip(orders, spo)) for spo in T - R
                ]
            }, ensure_ascii=False, indent=4)
            F.write(s.encode('utf-8') + '\n')
        F.close()
        return 2 * A / (B + C), A / B, A / C