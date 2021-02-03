from functions_and_classes import *
from nnfs.datasets import spiral_data

# CREATE SAMPLE DATA SET
X, y = spiral_data(samples=100, classes=3)

# CREATE HIDDEN LAYER WITH 2 INPUT VALUES (X Y POS OF EVERY SINGLE POINT IN ALL 3 SPIRALS)
# NUMBER OF NEURONS IN EACH HIDDEN LAYER IS COMPLETELY ARBITRARY
dense1 = Layer_Dense(2, 64)

# RECTIFIED LINEAR ACTIVATION FOR LAYER 1
activation1 = Activation_ReLU()

# INITALIZE SECOND LAYER (WHICH WILL BE OUR OUTPUT LAYER) AND PASS THE INPUT FROM THE FIRST LAYER
# 64 INPUT VALUES FOR 64 NEURONS IN PREVIOUS LAYER

dense2 = Layer_Dense(64, 3)
# INITALIZE SOFTMAX/LOSS OBJECT FOR LAYER 2
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# INITIALIZE OPTIMIZER
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)

# ITERATE OVER LOOP (EPOCH IS FANCY TERM FOR LOOPING OVER ENTIRE DATA SET FORWARD AND BACKWARDS ONCE)
for epoch in range(10001):

    # LAYER 1 FOWARD PASS
    dense1.forward(X)

    # OUTPUT OF LAYER ONE FORWARD PASS TO RELU ACTIVATION FUNCTION
    activation1.forward(dense1.output)

    # LAYER 2 FORWARD PASS
    dense2.forward(activation1.output)

    # LAYER 2 SOFTMAX ACTIVATION AND LOSS FUNCTION (WITH OUTPUT FROM PREVIOUS LAYER)
    loss = loss_activation.forward(dense2.output, y)

    # CALCULATE AND PRINT ACCURACY FROM OUTPUT VALUES COMPARED TO TARGET VALUES
    # AXIS = 1 BECAUSE (with literally everthing else in the NN) WE WANT THE CALCULATIONS FROM EACH ROW INSTEAD OF EACH COLUMN OF THE OUTPUT MATRIX
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    # PERFORM BACK-PROPOGATION
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # UPDATE WEIGHTS AND BIASES
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
