from mynn.optimizers.adam import Adam
from Model import Model

model = Model()

optim = Adam(model.parameters, learning_rate=1e-04, weight_decay=5e-04)


#notebook with noggin
#plotter, fig, ax = create_plot(('loss', 'reg_loss', 'cls_loss', 'Precision', 'Recall'),
              #                 1, figsize=(8, 12))


path = r"C:\Users\khoui\Nutrient_Detection\Food_Data"
train_dir = r"C:\Users\khoui\Nutrient_Detection\Food_Data\food-11\training"
test_dir = r"C:\Users\khoui\Nutrient_Detection\Food_Data\food-11\validation"

x_train, y_train, x_test, y_test  = load_draw_data(path)

generate_plate_set(x_train, y_train)

#load plate data

generate


for _ in range(5):
    train_epoch(train_data, train_boxes, train_labels, anchor_boxes, model, optim,
                val_data, val_boxes, val_labels)
    #plotter.plot()

#write something to display