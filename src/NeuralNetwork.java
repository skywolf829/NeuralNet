import java.util.ArrayList;

import Jama.*;

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
		biases = new double[hidden];
		dBiases = new double[hidden];
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
	public double[] Activate(double[] inputs){
		
		//Create an array of doubles for the results		 
		double[] results = new double[resultsSize];
		
		// Hold the values of each node as it calculates based on weights
		double[] hiddenNodeValues = new double[hiddenLayerSize];
		
		// Move from the inputs into the hidden layer
		for(int i = 0; i < inputs.length; i++){			
			for(int j = 0; j < hiddenLayerSize; j++){
				hiddenNodeValues[j] += inputs[i] * weightsToHidden[hiddenLayerSize * i + j];
			}
		}
		
		// Update the node with its bias and Sigmoid the node's value.
		for(int j = 0; j < hiddenLayerSize; j++){
			hiddenNodeValues[j] += biases[j];
			hiddenNodeValues[j] = 1 / (1 + Math.exp(-hiddenNodeValues[j]));
		}
		
		// Keep track of the sum of the results
		double resultsSum = 0;
		results = new double[resultsSize];
		// Move from the hidden layer to the output nodes
		for(int i = 0; i < hiddenLayerSize; i++){			
			for(int j = 0; j < resultsSize; j++){
				results[j] += hiddenNodeValues[i] * weightsToFinal[resultsSize * i + j];
			}
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
		
		// Get the correct results into a matrix
		double[][] resultsArray = new double[correctResults.size()][correctResults.get(0).length];
		correctResults.toArray(resultsArray);
		Matrix correctResultsMatrix = new Matrix(resultsArray);
		
		// Run the network on the training data and make a matrix with the estimated results
		Matrix estimatedResultsMatrix = 
				new Matrix(correctResults.size(), correctResults.get(0).length);
		for(int i = 0; i < allInputs.size(); i++){
			double[] answer = Activate(allInputs.get(i));
			for(int j = 0; j < answer.length; j++){
				estimatedResultsMatrix.set(i, j, answer[j]);
			}
		}
		
		// Create a matrix representing the weights from the inputs to the hidden layer
		Matrix inputToHidden = new Matrix(allInputs.get(0).length, hiddenLayerSize);
		for(int i = 0; i < allInputs.get(0).length; i++)
		{
			for(int j = 0; j < hiddenLayerSize; j++){
				inputToHidden.set(i, j, weightsToHidden[i * allInputs.get(0).length + j]);
			}
		}
		
		// Create a matrix representing the weights from the hidden layer to the outputs
		Matrix hiddenToFinal = new Matrix(hiddenLayerSize, correctResults.get(0).length);
		for(int i = 0; i < hiddenLayerSize; i++)
		{
			for(int j = 0; j < correctResults.get(0).length; j++){
				inputToHidden.set(i, j, weightsToFinal[i * hiddenLayerSize + j]);
			}
		}
		
		Matrix diff = estimatedResultsMatrix.minus(correctResultsMatrix);
	}
	
	private double[] allWeights(){
		double[] array = new double[weightsToHidden.length 
		                            + biases.length + weightsToFinal.length];
		int i = 0;
		for(i = 0; i < weightsToHidden.length; i++){
			array[i] = weightsToHidden[i];
		}
		for(; i < biases.length + i; i++){
			array[i] = biases[i];
		}
		for(; i < weightsToFinal.length + i; i++){
			array[i] = weightsToFinal[i];
		}
		return array;
	}
	private double[] difference(double[] vector1, double[] vector2){
		double[] d = new double[vector1.length];
		for(int i = 0; i < vector1.length; i++){
			d[i] = vector1[i] - vector2[i];
		}
		return d;
	}
	
	/**
	 * Find the distance from any size vector vector1 to the same sized vector vector2
	 */
	private double distance(double[] vector1, double[] vector2){
		double d = 0;
		for(int i = 0; i < vector1.length; i++){
			d += Math.pow(vector1[i] - vector2[i], 2);
		}
		d = Math.sqrt(d);		
		return d;
	}
}
