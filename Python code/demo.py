from mynn.optimizers.adam import Adam
from Model import Model

model = Model()

optim = Adam(model.parameters, learning_rate=1e-04, weight_decay=5e-04)


#notebook with noggin
#plotter, fig, ax = create_plot(('loss', 'reg_loss', 'cls_loss', 'Precision', 'Recall'),
              #                 1, figsize=(8, 12))