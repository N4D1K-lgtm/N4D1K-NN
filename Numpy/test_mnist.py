from functions_and_classes import *
from sklearn.utils import shuffle
from mnist import MNIST

print("Loading data...")
mndata = MNIST('./bin/data')
trainImages, trainLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

trainImages = np.array(trainImages)[:60000]
trainLabels = np.array(trainLabels)[:60000]

testImages = np.array(testImages)[:10000]
testLabels = np.array(testLabels)[:10000]

print("trainImages shape: " + str(trainImages.shape))
print("trainLabels shape: " + str(trainLabels.shape))

# PRINT A SAMPLE DIGIT
print_sample(trainImages[0], trainLabels[0]);

trainImages = trainImages / 255
testImages = testImages / 255


# CREATE SAMPLE DATA SET

# CREATE HIDDEN LAYER WITH 2 INPUT VALUES (X Y POS OF EVERY SINGLE POINT IN ALL 3 SPIRALS)
# NUMBER OF NEURONS IN EACH HIDDEN LAYER IS COMPLETELY ARBITRARY
dense1 = Layer_Dense(784, 16, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

# RECTIFIED LINEAR ACTIVATION FOR LAYER 1
activation1 = Activation_ReLU()
activation2 = Activation_ReLU()

# INITALIZE SECOND LAYER (WHICH WILL BE OUR OUTPUT LAYER) AND PASS THE INPUT FROM THE FIRST LAYER
# 64 INPUT VALUES FOR 64 NEURONS IN PREVIOUS LAYER

dense2 = Layer_Dense(16, 16)

dense3 = Layer_Dense(16, 10)

# INITALIZE SOFTMAX/LOSS OBJECT FOR LAYER 2
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()


# INITIALIZE OPTIMIZER
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)

# ITERATE OVER LOOP (EPOCH IS FANCY TERM FOR LOOPING OVER ENTIRE DATA SET FORWARD AND BACKWARDS ONCE)
for epoch in range(300):

    # LAYER 1 FOWARD PASS
    dense1.forward(trainImages)

    # OUTPUT OF LAYER ONE FORWARD PASS TO RELU ACTIVATION FUNCTION
    activation1.forward(dense1.output)

    # LAYER 2 FORWARD PASS
    dense2.forward(activation1.output)

    activation2.forward(dense2.output)

    dense3.forward(activation2.output)

    # LAYER 2 SOFTMAX ACTIVATION AND LOSS FUNCTION (WITH OUTPUT FROM PREVIOUS LAYER)
    data_loss = loss_activation.forward(dense3.output, trainLabels)

    # CALCULATE REGULARIZATION PENALTY
    regularization_loss = \
        loss_activation.loss.regularization_loss(dense1) + \
        loss_activation.loss.regularization_loss(dense2) + \
        loss_activation.loss.regularization_loss(dense3)

    # CALCULATE AND PRINT ACCURACY FROM OUTPUT VALUES COMPARED TO TARGET VALUES
    # AXIS = 1 BECAUSE (with literally everthing else in the NN) WE WANT THE CALCULATIONS FROM EACH ROW INSTEAD OF EACH COLUMN OF THE OUTPUT MATRIX
    loss = data_loss + regularization_loss

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(trainLabels.shape) == 2:
        y = np.argmax(trainLabels, axis=1)
    accuracy = np.mean(predictions==trainLabels)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    # PERFORM BACK-PROPOGATION
    loss_activation.backward(loss_activation.output, trainLabels)
    dense3.backward(loss_activation.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # UPDATE WEIGHTS AND BIASES
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()



# VALIDATE MODEL (ACTUALLY TESTING IT) (RUN THROUGH THIS AFTER ITS DONE TRAINING ^^^^)

# TEST DATA
X_test = testImages
y_test = testLabels

# PERFORM FORWARD PASS THROUGH THE LAYER
dense1.forward(X_test)

# FORWARD PASS THROUGH ACTIVATION FUNCTION
activation1.forward(dense1.output)

# FORWARD PASS THROUGH SECOND HIDDEN LAYER
dense2.forward(activation1.output)

activation2.forward(dense2.output)

dense3.forward(activation2.output)

# LOSS ACTIVATION FUNCTION
loss = loss_activation.forward(dense3.output, y_test)

#CALCULATE ACCURACY
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
