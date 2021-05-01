import java.util.Arrays;

public class NeuralNetwork {
    public final double[][] hidden_layer_weights;
    public final double[][] output_layer_weights;
    private final int num_inputs;
    private final int num_hidden;
    private final int num_outputs;
    private final double learning_rate;

    public NeuralNetwork(int num_inputs, int num_hidden, int num_outputs, double[][] initial_hidden_layer_weights, double[][] initial_output_layer_weights, double learning_rate) {
        //Initialise the network
        this.num_inputs = num_inputs;
        this.num_hidden = num_hidden;
        this.num_outputs = num_outputs;

        this.hidden_layer_weights = initial_hidden_layer_weights;
        this.output_layer_weights = initial_output_layer_weights;

        this.learning_rate = learning_rate;
    }


    //Calculate neuron activation for an input
    public double sigmoid(double input) {
        double output = 1 / (1 + Math.exp(-input));
        return output;
    }

    //Feed forward pass input to a network output
    public double[][] forward_pass(double[] inputs) {
        double[] hidden_layer_outputs = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output.
            //For each z = weight[0] * O[0] + weight[1] * O[1] + weight[2] * O[2] + weight[3] * O[3]
            //Each O[i] is the real values from the penguin data
            double weighted_sum = 0;
            for(int j = 0; j < num_inputs; j++){
                weighted_sum += hidden_layer_weights[j][i] * inputs[i];
            }
            //Sigmoid function on weight_sum
            double output = sigmoid(weighted_sum);
            hidden_layer_outputs[i] = output;
        }

        double[] output_layer_outputs = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output.
            double weighted_sum = 0;
            for(int j = 0; j < num_hidden; j++){
                weighted_sum += output_layer_weights[j][i] * hidden_layer_outputs[j];
            }
            double output = sigmoid(weighted_sum);
            output_layer_outputs[i] = output;
        }
        return new double[][]{hidden_layer_outputs, output_layer_outputs};
    }

    public double[][][] backward_propagate_error(double[] inputs, double[] hidden_layer_outputs,
                                                 double[] output_layer_outputs, int desired_outputs) {

        double[] output_layer_betas = new double[num_outputs];
        // TODO! Calculate output layer betas.
        //Beta is the difference between desired output and actual output
        //Go over this - desired outputs (one integer?)
        for(int i = 0; i < num_outputs; i++){
            if(i == desired_outputs) output_layer_betas[i] = 1 - output_layer_outputs[i];
            else output_layer_betas[i] = 0 - output_layer_outputs[i];
//            System.out.println("OL desired: " + desired_outputs);
//            System.out.println("OL Output: " + output_layer_outputs[i]);
        }
//        System.out.println("OL betas: " + Arrays.toString(output_layer_betas));

        double[] hidden_layer_betas = new double[num_hidden];
        // TODO! Calculate hidden layer betas.
        //Betah5 = k output layer 1 through 4 (Wh5 -> k * outputk * (1 -outputk) * betak
        for(int i = 0; i < num_hidden; i++){
            double beta = 0;
            for(int j = 0; j < num_outputs; j++){
                //Changed hidden_layer_weights[][] to j,i
                beta += /*weight from hidden layer node to output layer node*/
                        hidden_layer_weights[j][i] * output_layer_outputs[j] * (1 - output_layer_outputs[j]) * output_layer_betas[j];
            }
            hidden_layer_betas[i] = beta;
        }
//        System.out.println("HL betas: " + Arrays.toString(hidden_layer_betas));

        // This is a HxO array (H hidden nodes, O outputs)
        double[][] delta_output_layer_weights = new double[num_hidden][num_outputs];
        // TODO! Calculate output layer weight changes.
        //Delta Weight i->j = learning_rate0.2 * output[i] * output[j] * (beta[j])
        //Weights from hidden layer to output layer (i think)
        for(int i = 0; i < num_hidden; i++){
            for(int j = 0; j < num_outputs; j++){
                delta_output_layer_weights[i][j] =
                        learning_rate * hidden_layer_outputs[i] * output_layer_outputs[j] * (1 - output_layer_outputs[j]) * output_layer_betas[j];
            }
        }

        // This is a IxH array (I inputs, H hidden nodes)
        double[][] delta_hidden_layer_weights = new double[num_inputs][num_hidden];
        // TODO! Calculate hidden layer weight changes.
        //Weights from input layer to hidden layer
        for(int i = 0; i < num_inputs; i++){
            for(int j = 0; j < num_hidden; j++){
                delta_hidden_layer_weights[i][j] =
                        learning_rate * inputs[i] * hidden_layer_outputs[j] * (1 - hidden_layer_outputs[j]) * hidden_layer_betas[j];
            }
        }

        // Return the weights we calculated, so they can be used to update all the weights.
        return new double[][][]{delta_output_layer_weights, delta_hidden_layer_weights};
    }

    public void update_weights(double[][] delta_output_layer_weights, double[][] delta_hidden_layer_weights) {
        // TODO! Update the weights
        //Hidden layer to output layer
        //Output layer weights
        for(int i = 0; i < num_hidden; i++){
            for(int j = 0; j < num_outputs; j++){
                output_layer_weights[i][j] += delta_output_layer_weights[i][j];
//                System.out.println("Updated Output Weights: " + this.output_layer_weights[i][j]);
            }
        }
        //Input layer to hidden layer
        //Hidden layer weights
        for(int i = 0; i < num_inputs; i++){
            for(int j = 0; j < num_hidden; j++){
                hidden_layer_weights[i][j] += delta_hidden_layer_weights[i][j];
//                System.out.println("Updated Hidden Weights: " + this.hidden_layer_weights[i][j]);
            }
        }
    }

    public void train(double[][] instances, int[] desired_outputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("epoch = " + epoch);
            int[] predictions = new int[instances.length];
            for (int i = 0; i < instances.length; i++) {
                double[] instance = instances[i];
                double[][] outputs = forward_pass(instance);
                double[][][] delta_weights = backward_propagate_error(instance, outputs[0], outputs[1], desired_outputs[i]);
//                int predicted_class = -1; // TODO!
                int predicted_class = findMaxIndex(outputs[1]);
                predictions[i] = predicted_class;
                /****/
                //For testing only print the final epoch
                if(epoch == epochs - 1) {
                    System.out.println("Instance: " + i);
                    System.out.println("Desired output: " + desired_outputs[i]);
                    System.out.println("Predicted output: " + predicted_class);
                    System.out.println("Output Values: " + Arrays.toString(outputs[1]) + '\n');
                }
                /****/
                //We use online learning, i.e. update the weights after every instance.
                update_weights(delta_weights[0], delta_weights[1]);
            }

            // Print new weights
//            System.out.println("Hidden layer weights \n" + Arrays.deepToString(hidden_layer_weights));
//            System.out.println("Output layer weights  \n" + Arrays.deepToString(output_layer_weights));

            // TODO: Print accuracy achieved over this epoch
            int accuracyCount = 0;
            for(int i = 0; i < predictions.length; i++){
                if(predictions[i] == desired_outputs[i]) accuracyCount++;
            }
            double acc = ((double)accuracyCount / instances.length) * 100;
            System.out.println("Accuracy = " + acc + "\n");
        }
    }

    public int[] predict(double[][] instances) {
        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);
            int predicted_class = -1;  // TODO !Should be 0, 1, or 2.
            //Predicted class determined by which output layer value is biggest
            double[] output_layer_outputs = outputs[1];
            predicted_class = findMaxIndex(output_layer_outputs);
            predictions[i] = predicted_class;
        }
        return predictions;
    }

    /**
     * Find the max double index in an array
     */
    private int findMaxIndex(double[] values){
        double max = -10.0;
        int index = -1;
        for(int i = 0; i < values.length; i++){
            if(values[i] > max){
                max = values[i];
                index = i;
            }
        }
        return index;
    }

}
