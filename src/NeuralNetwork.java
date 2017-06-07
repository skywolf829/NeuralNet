import java.util.ArrayList;

public class NeuralNetwork {

	// These hold the sizes for each stage of the neural network
	private int hiddenLayerSize, resultsSize;
	// Meant to change how quickly the network changes it's weights
	private double learningConstant = 0.01;
	// Bias for how much to offset each node
	private double[] biases;
	// This represents the change in biases
	private double[] dBiases;
	// Arrays holding the weights from each layer to the next
	private double[] weightsToHidden, weightsToFinal;
	// These represent the change in value to each weight
	private double[] dWeightsToHidden, dWeightsToFinal;
	// Necessary variables
	private double cost, totalDistance;
	// Boolean telling the network if this is a winner take all system
	private boolean winnerTakeAll = false;
	
	
	/**
	 * Creates a neural network that has an input layer size of input + 1 and an output 
	 * layer size of output
	 */
	public NeuralNetwork(int input, int output){
		// Initialize the size of each layer
		// Assume hidden layer size is the same as input
		resultsSize = output;
		
		// Create the arrays for the weights based on sizes
		weightsToHidden = new double[(input) * (input)];
		dWeightsToHidden = new double[input * input];
		weightsToFinal = new double[(input) * output];
		dWeightsToFinal = new double[input * input];
		biases = new double[input];
		dBiases = new double[input];
		RandomizeWeightsAndBiases();
	}
	/**
	 * Creates a neural network that has an input layer size of input + 1,
	 * a hidden layer size of hidden, and an output layer size of output
	 */
	public NeuralNetwork(int input, int hidden, int output){
		// Initialize the size of each layer
		hiddenLayerSize = hidden;
		resultsSize = output;
		
		// Create the arrays for the weights based on sizes
		weightsToHidden = new double[(input) * hidden];
		dWeightsToHidden = new double[input * input];
		weightsToFinal = new double[hidden * output];
		dWeightsToFinal = new double[input * input];
		biases = new double[input];
		dBiases = new double[input];
		RandomizeWeightsAndBiases();
	}
	
	/**
	 * Setter for winnerTakeAll
	 */
	public void SetWinnerTakeAll(boolean b){
		winnerTakeAll = b;
	}	
	
	/**
	 * Activate the neural network by feeding it inputs and getting the 
	 * final result in an array
	 */
	public double[] Activate(double[] initInputs){
		
		//Create an array of doubles for the results		 
		double[] results = new double[resultsSize];
		
		// Create an array of doubles for the new input array		 
		double[] inputs = new double[initInputs.length + 1];
	
		// Copy all input values into the new array		 
		for(int i = 0; i < initInputs.length; i++){
			inputs[i] = initInputs[i];
		}
		// Then add a bias with a value of 1 in the last spot.
		inputs[inputs.length - 1] = 1;
		
		// Hold the values of each node as it calculates based on weights
		double[] hiddenNodeValues = new double[hiddenLayerSize];
		
		// Move from the inputs into the hidden layer
		for(int i = 0; i < inputs.length; i++){
			hiddenNodeValues[i] = 0;
			for(int j = 0; j < hiddenLayerSize; j++){
				hiddenNodeValues[i] += inputs[i] * weightsToHidden[hiddenLayerSize * i + j];
			}
			hiddenNodeValues[i] = 1 / (1 + Math.exp(-hiddenNodeValues[i]) - biases[i]);
		}
		
		// Keep track of the sum of the results
		double resultsSum = 0;
		// Move from the hidden layer to the output nodes
		for(int i = 0; i < hiddenLayerSize; i++){
			results[i] = 0;
			for(int j = 0; j < resultsSize; j++){
				results[i] += hiddenNodeValues[j] * weightsToFinal[resultsSize * i + j];
			}
			results[i] = 1 / (1 + Math.exp(-results[i]) - biases[i]);
			resultsSum += results[i];
		}	
		// Normalize the output
		for(int i = 0; i < resultsSize; i++){
			results[i] /= resultsSum;
		}
		// Return the final results
		return results;
	}
	
	/**
	 * Begin training with randomizing the weights on each transition
	 */
	public void RandomizeWeightsAndBiases(){
		for(int i = 0; i < weightsToHidden.length; i++){
			weightsToHidden[i] = Math.random();
		}
		for(int i = 0; i < weightsToFinal.length; i++){
			weightsToFinal[i] = Math.random();
		}
		for(int i = 0; i < hiddenLayerSize; i++){
			biases[i] = Math.random();
		}
	}
	
	public int WinnerTakeAll(double[] results){
		double max = results[0];
		int maxSpot = 0;
		for(int i = 1; i < results.length; i++){
			if(results[i] > max){
				max = results[i];
				maxSpot = i;
			}
		}
		return maxSpot;
	}
	public void TrainNeuralNet(ArrayList<double[]> allInputs, ArrayList<double[]> correctResults){
		for(int i = 0; i < allInputs.size(); i++){
			double oldCost = cost;
			double[] oldWeightsToHidden = weightsToHidden.clone();
			double[] oldWeightsToFinal = weightsToFinal.clone();
			double[] oldBiases = biases.clone();
			
			Train(allInputs.get(i), correctResults.get(i), i);
			
			double changeInCost = cost - oldCost;
			
		}
	}
	public void Train(double[] inputs, double[] correct, int sampleNumber){
		double[] results = Activate(inputs);
		totalDistance += distance(results, correct);
		cost = (1 / (2 * sampleNumber)) * totalDistance;
		
	}
	
	private double distance(double[] vector1, double[] vector2){
		double d = 0;
		for(int i = 0; i < vector1.length; i++){
			d += Math.pow(vector1[i] - vector2[i], 2);
		}
		d = Math.sqrt(d);		
		return d;
	}
}
