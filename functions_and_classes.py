import numpy as np



# CLASS FOR HIDDEN LAYERS TAKES NUMBER OF INPUTS AND NUMBER OF NEURONS
class Layer_Dense:
	def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):

		# INTIALIZE RANDOM WEIGHTS BASED OFF GAUSSIAN DISTRUBITION CENTERED AROUND 0 AND MULTIPLY BY .1 TO SCALE WEIGHTS CLOSER TO 0 AND 1
		# THERE IS PROBABLY A BETTER WAY TO DO THIS
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

		# INITIALIZE BIASES (IN THIS CASE 1D ARRAY FILLED WITH ZEROS FOR THE AMOUNT OF NUERONS IN THE LAYER)
		self.biases = np.zeros((1, n_neurons))

		# REGULARIZATON STRENGTH
		self.weight_regularizer_l1 = weight_regularizer_l1
		self.weight_regularizer_l2 = weight_regularizer_l2
		self.bias_regularizer_l1 = bias_regularizer_l1
		self.bias_regularizer_l2 = bias_regularizer_l2
	
	# FORWARD FEED
	def forward(self, inputs):

		# REMEMBER INPUTS
		self.inputs = inputs
		# GET OUTPUT VALUES FROM INPUTS, BIASES AND WEIGHTS
		self.output = np.dot(inputs, self.weights) + self.biases

	# BACK-PROPOGATION
	# I understand sort of what this is doing but the underlying math is way over my head, (multivariable calculus and linear algebra)
	def backward(self, dvalues):
		
		# GRADIENT DESCENT ON PARAMATERS
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

		# GRADIENTS DESCENT ON REGULARIZATION
		
		#L1 ON WEIGHTS
		if self.weight_regularizer_l1 > 0:
			dL1 = np.ones_like(self.weights)
			dL1[self.weights < 0] = -1
			self.dweights += self.weight_regularizer_l1 * dL1
		
		# L2 ON WEIGHTS
		if self.weight_regularizer_l2 > 0:
			self.dweights += 2 * self.weight_regularizer_l2 * \
			self.weights
		
		# L1 ON BIASES
		if self.bias_regularizer_l1 > 0:
			dL1 = np.ones_like(self.biases)
			dL1[self.biases < 0] = -1
			self.dbiases += self.bias_regularizer_l1 * dL1
		
		# L2 ON BIASES
		if self.bias_regularizer_l2 > 0:
			self.dbiases += 2 * self.bias_regularizer_l2 * \
			self.biases

		# Gradient on values
		self.dinputs = np.dot(dvalues, self.weights.T)

# SETS ACTIVATION FUNCTION IF X < -1 OUTPUT 0 ELSE OUTPUT X (RECTIFIED LINEAR FUNCTION)
class Activation_ReLU:

	# FORWARD FEED
	def forward(self, inputs):

		# REMEMBER INPUTS
		self.inputs = inputs

		self.output = np.maximum(0, inputs)

	# BACK-PROPAGATION
	def backward(self, dvalues):

		self.dinputs = dvalues.copy()

		# SETS GRADIENT TO 0 WHEN INPUTS ARE NEGATIVE
		self.dinputs[self.inputs <= 0] = 0

'''
SOFTMAX IS CALLED ON THE FINAL OUTPUT LAYER OF THE NEURAL NETWORK
THE INPUT MATRIX IS SUBTRACTED BY THE HIGHEST VALUE IN THE MATRIX TO BRING EVERTHING BELOW -1
THE INPUT MATRIX IS THEN EXPONENTIATED BY ROW (AXIS 1)
EXPONENTIATED INPUT MATRIX IS DIVIDED BY THE SUM OF EVERY VALUE IN THE MATRIX
EVERY BATCH (ROW IN INPUT MATRIX) ADDS TO 1 BECAUSE OF NORMALIZATION FUNCTION
'''

class Activation_Softmax:

	# FORWARD FEED
	def forward(self, inputs):

		#DONT CHOP OFF LEFT ARM
		self.inputs = inputs

		#GET UNORMALIZED PROBABILITIES
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True))
		
		#NORMALIZE PROBABILITIEES
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims = True)

		self.output = probabilities

	# BACK-PROPAGATION (tired of me saying that yet?)
	def backward(self, dvalues):

		# CREATE UN-INITIALIZED ARRAY OF THE SAME SIZE AS INPUT
		self.dinputs = np.empty_like(dvalues)

		# ITERATE OUTPUTS AND GRADIENTS
		for index, (single_output, single_dvalues) in \
			enumerate(zip(self.output, dvalues)):
			
			# FLATTEN OUTPUT ARRAY
			single_output = single_output.reshape(-1, 1)
			
			'''
			CALCULATES JACOBIAN MATRIX FROM FLATTENED ARRAY
			(this is the math thats over my head)
			I know (if i understand it correctly) its somehow doing matrix math to find the most desirable direction of weights and biases
			while taking into account every single neurons magnitude and direction
			'''
			
			jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

			# GET FINAL GRADIENT AND ADD IT TO THE ARRAY OF SAMPLE GRADIENTS
			self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# ADAM OPTIMIZER, best for the type of digit recognition that Im planning on doing, just google it if your curious
# FYI this is copied I dont really know how it works mathematically
class Optimizer_Adam:

    #initialize
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # call once
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2

        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                             self.epsilon)
        layer.biases += -self.current_learning_rate * \
                         bias_momentums_corrected / \
                         (np.sqrt(bias_cache_corrected) +
                             self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


class Loss:


	# CALCULATE REGULARIZATION LOSS 
	def regularization_loss(self, layer):

		# DEFAULT
		regularization_loss = 0

		# L1 REGULIZATION ON WEIGTHS
		# CALCULATE WHEN FACTOR IS MORE THAN 0
		if layer.weight_regularizer_l1 > 0:
			regularization_loss += layer.weight_regularizer_l1 * \
			np.sum(np.abs(layer.weights))

		# L2 REGULARIZATION ON WEIGHTS
		if layer.weight_regularizer_l2 > 0:
			regularization_loss += layer.weight_regularizer_l2 * \
			np.sum(layer.weights *
			      layer.weights)


		# L1 REGULARIZATION ON BIASES
		# CALCULATE WHEN FACTOR IS GREATER THAN 0
		if layer.bias_regularizer_l1 > 0:
			regularization_loss += layer.bias_regularizer_l1 * \
			np.sum(np.abs(layer.biases))

		# L2 REGULARIZATION BIASES
		if layer.bias_regularizer_l2 > 0:
			regularization_loss += layer.bias_regularizer_l2 * \
			np.sum(layer.biases *
			      layer.biases)

		return regularization_loss



	# CALCULATES DATA AND NORMALIZED VALUES GIVEN MODEL OUPUT AND GROUND TRUTH VALUES
	
	def calculate(self, output, y):

		# CALCULATE SAMPLE LOSSES
		sample_losses = self.forward(output, y)

		# GET AVERAGE LOSS FROM SAMPLE LOSSES
		data_loss = np.mean(sample_losses)

		return data_loss


# CROSS-ENTROPY (REALLY WE'RE CHEATING BY JUST DOING NEGATIVE LOG BECAUSE WE ONLY INPUT ONE HOT ENCODED ARRAYS)
class Loss_CategoricalCrossentropy(Loss):

	# FORWARD FEED
	def forward(self, y_pred, y_true):

		# FINDS NUMBER OF ARRAYS IN THE INPUT ARRAY, SO IN THIS CASE 3 BECAUSE OF THE 3 SETS OF SPIRALS
		samples = len(y_pred)

		# GET RID OF 0 AND EXTREME OUTLIERS
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)


		# PROBABILITIES FOR TARGET VALUES IF IN CATEGORICAL VARIABLES
		if len(y_true.shape) == 1:
		    correct_confidences = y_pred_clipped[
		        range(samples),
		        y_true
		    ]

		# MASK VALUES FOR ONE HOT ENCODED LABELS (ONLY ONE OUTPUT IN ARRAY IS 1 AND REST ARE 0)
		elif len(y_true.shape) == 2:
		    correct_confidences = np.sum(
		        y_pred_clipped*y_true,
		        axis=1
		    )

		# CALCULATE LOSS BASED ON NEG LOG, GOOGLE CROSS ENTROPY LOSS AND YOU CAN SIMPLIFY THE FORMULA TO NEG LOG IF INPUT IS ONE HOT ENCODED
		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods

	# BACK-PROPAGATION
	def backward(self, dvalues, y_true):

		# NUMBER OF SAMPLES
		samples = len(dvalues)
		# SET THE NUMBER OF LABELS FOR THE ENTIRE BATCH BASED ON THE NUMBER OF LABELS ATTACHED TO THE FIRST SAMPLE
		labels = len(dvalues[0])

		# IF LABELS ARE SPARCE TURN THEM INTO A ONE HOT VECTOR
		if len(y_true.shape) == 1:
		    y_true = np.eye(labels)[y_true]

		# CALCULATE GRADIENT
		self.dinputs = -y_true / dvalues
		# NORMALIZE GRADIENT
		self.dinputs = self.dinputs / samples

# THIS IS JUST COMBINING SOFTMAX ACTIVATION AND CROSS ENTROPY FOR FASTER BACKWARDS STEP
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # INITALIZE OBJECTS OF BOTH CLASSES
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # FORWARD PASS
    def forward(self, inputs, y_true):
        # OUTPUT LAYER ACTIVATION FUNCTION
        self.activation.forward(inputs)
        # SET OUTPUT
        self.output = self.activation.output
        # CALCULATE AND RETURN VALUE
        return self.loss.calculate(self.output, y_true)

    # BACK-PROPAGATION
    def backward(self, dvalues, y_true):

        # NUMBER OF SAMPLES
        samples = len(dvalues)

        # IF LABELS ARE ONE HOT ENCODED TURN THEM INTO DISCRETE VALUES
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # STORE PARAMETERS SO I DON'T CHOP OFF MY LEFT ARM
        self.dinputs = dvalues.copy()
        # CALCULATE GRADIENT
        self.dinputs[range(samples), y_true] -= 1
        # NORMALIZE GRADIENT
        self.dinputs = self.dinputs / samples

def print_sample(image, label):
    print("Sample image: should be " + str(label));
    s = '';
    for i in range (784):
        if ((i % 28) == 0 and i != 0):
            print(s)
            s = ''
        tmp = image[i]
        if (tmp > 127):
            s += "1 "
        else:
            s += "0 "


def random_test_sample(X, y, Max):
	randInt = np.random.randint(0, Max)
	self.X = np.array(X)[randInt]
	self.y = np.array(y)[randInt]
