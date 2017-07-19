import Jama.*;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class NeuralNetwork implements Runnable{

	private Thread t;
	private boolean stopThread = false;
	private boolean training = false, initialized = false;
	
	private double[][] inputs, correctResults;
	private int iterations;
	private double minCost;
	private ArrayList<Matrix> minHiddenLayerWeights;
	
	private int numInputNodes;
	private int numHiddenLayers;
	private int[] nodesPerLayer;
	private int numOutputNodes;
	
	private double learningRate = 0.1;
	
	private ArrayList<Matrix> hiddenLayerWeightsMatrices;
	
	public NeuralNetwork(){	
	}
	public void SetInputSize(int inputSize){
		numInputNodes = inputSize;
	}
	public void SetOutputSize(int outputSize){
		numOutputNodes = outputSize;
	}
	public void SetNumHiddenLayers(int hiddenLayers){
		numHiddenLayers = hiddenLayers;
	}
	public void SetNodesInEachLayer(int[] nodesInEachLayer){
		nodesPerLayer = nodesInEachLayer;
	}
	public void SetLearningRate(double d){
		learningRate = d;
	}
	public void SetHiddenLayerWeights(ArrayList<Matrix> w){
		hiddenLayerWeightsMatrices = w;
	}
	public int GetNumInputNodes(){
		return numInputNodes;
	}
	public int GetNumHiddenLayers(){
		return numHiddenLayers;
	}
	public int[] GetNodesPerLayer(){
		return nodesPerLayer;
	}
	public int GetNumOutputNodes(){
		return numOutputNodes;
	}
	public ArrayList<Matrix> GetWeights(){
		return hiddenLayerWeightsMatrices;
	}
	public boolean IsTraining(){ return training; }
	public boolean IsInitialized(){ return initialized; }
	/*
	 * Creates the network with the given dimensions and sizes.
	 */
	public void Initialize(){
		//Create a new ArrayList where each matrix in it represents the weights from
		//	layer i to i+1
		hiddenLayerWeightsMatrices = new ArrayList<Matrix>();
		//This means we'll need i+1 entries total
		for(int i = 0; i <= numHiddenLayers; i++){
			//The number of rows and columns needed for this matrix
			int rows, cols;
			//Rows will be the number of nodes in the "previous" layer
			if(i == 0) rows = numInputNodes;
			else rows = nodesPerLayer[i - 1];
			//Columns will be the number of nodes in the "next" layer
			if(i == numHiddenLayers) cols = numOutputNodes;
			else cols = nodesPerLayer[i];
			//Create a new matrix of that size
			Matrix m = new Matrix(rows, cols);
			//Randomize each entry in the matrix
			for(int j = 0; j < rows; j++){
				for(int k = 0; k < cols; k++){
					m.set(j, k, Math.random() * 10);
				}
			}
			//Add the matrix to the ArrayList
			hiddenLayerWeightsMatrices.add(m);
		}
		initialized = true;
	}
	
	/*
	 * Used on each neuron.
	 */
	private Matrix ActivationFunction(Matrix input){
		for(int i = 0; i < input.getRowDimension(); i++){
			for(int j = 0; j < input.getColumnDimension(); j++){
				input.set(i, j, 
						1 / (1 + Math.exp(-input.get(i, j)))
						);
			}
		}
		return input;
	}
	
	/*
	 * Used in backpropagation to find the gradient.
	 */
	private Matrix ActivationFunctionPrime(Matrix input){
		for(int i = 0; i < input.getRowDimension(); i++){
			for(int j = 0; j < input.getColumnDimension(); j++){
				input.set(i, j, 
						(Math.exp(-input.get(i, j))) / (Math.pow(1 + Math.exp(-input.get(i, j)), 2))
						);
			}
		}
		return input;
	}
	
	/*
	 * Randomizes all weights in the neural network.
	 */
	public void RandomizeWeights(){
		//Use the matrix in layer i
		for(int i = 0; i < hiddenLayerWeightsMatrices.size(); i++){
			//Iterate through the rows
			for(int j = 0; j < hiddenLayerWeightsMatrices.get(i).getRowDimension(); j++){
				//Iterate through the columns
				for(int k = 0; k < hiddenLayerWeightsMatrices.get(i).getColumnDimension(); k++){
					//Use a random double
					hiddenLayerWeightsMatrices.get(i).set(j, k, Math.random() * 10);
				}
			}
		}
	}
	
	/*
	 * Solves a single input in the neural network.
	 */
	public double[] Forward(double[] inputs){
		//Results is always a row vector 
		double[] results = new double[numOutputNodes];
		
		//Convert the input from a double array into a matrix
		Matrix inputMatrix = new Matrix(1, inputs.length);	
		for(int i = 0; i < inputs.length; i++){
			inputMatrix.set(0, i, inputs[i]);
		}
		
		//We are constantly right multiplying the results by the weights, so 
		//	the results matrix is copied from the created input matrix
		Matrix resultsMatrix = inputMatrix.copy();
		for(int i = 0; i <= numHiddenLayers; i++){
			//Get the weights for the current layer
			Matrix weights = hiddenLayerWeightsMatrices.get(i);
			//Right multiply the results by the weights
			resultsMatrix = resultsMatrix.times(weights);
			resultsMatrix = ActivationFunction(resultsMatrix);
		}		
		//The results will be in a row matrix; get the double array copy of that and return;
		results = resultsMatrix.getRowPackedCopy();
		return results;
	}
	
	/*
	 * Solves all inputs in the neural network, used for training.
	 */
	private Matrix ForwardAllTraining(Matrix inputs, ArrayList<Matrix> valuesBeforeActivation,
			ArrayList<Matrix> activationValues){
		Matrix results = inputs.copy();
		activationValues.add(results.copy());
		for(int i = 0; i <= numHiddenLayers; i++){
			//Get the weights for the current layer
			Matrix weights = hiddenLayerWeightsMatrices.get(i);
			//Right multiply the results by the weights
			results = results.times(weights.copy());
			valuesBeforeActivation.add(results.copy());
			//Activate each neuron
			results = ActivationFunction(results);
			
			activationValues.add(results.copy());
		}	
		//Each row will represent the outputs for the corresponding inputs row
		activationValues.remove(activationValues.size() - 1);
		return results;
	}
	/*
	 * Trains the neural network given the inputs and correctResults, working
	 * to minimize the cost.
	 * 
	 * Will repeat training until the cost is less than a given maximum cost.
	 * Randomizes weights after an arbitrary number of training cycles if progress toward
	 * the maximum cost is not made.
	 */
	public void Train(double[][] inputs, double[][] correctResults){		
		this.inputs = inputs;
		this.correctResults = correctResults;
		t = new Thread(this, "TrainingThread");
		t.start();
	}
	/*
	 * Used in backpropagation to find the gradient
	 */
	private ArrayList<Matrix> populateDeltas(Matrix actual, Matrix predicted,
			ArrayList<Matrix> valuesBeforeActivation){
		ArrayList<Matrix> deltas = new ArrayList<Matrix>();
		
		for(int i = numHiddenLayers; i >= 0; i--){
			Matrix delta;
			if(i == numHiddenLayers){
				delta = predicted.copy();
				delta = delta.minus(actual);
			}
			else{
				delta = deltas.get(0).copy();
				Matrix w = hiddenLayerWeightsMatrices.get(i + 1).copy();
				w = w.transpose();
				delta = delta.times(w);
			}
			Matrix sigmoidPrimes;
			sigmoidPrimes = ActivationFunctionPrime(valuesBeforeActivation.get(i).copy());
			delta = ElementwiseMultiplication(delta, sigmoidPrimes);
			deltas.add(0, delta.copy());
		}		
		
		return deltas;
	}
	/*
	 * Used in backpropagation to find the gradient
	 */
	private ArrayList<Matrix> costFunctionPrime(ArrayList<Matrix> deltas, ArrayList<Matrix> activationValues){
		ArrayList<Matrix> dCost_dWeights = new ArrayList<Matrix>();
		for(int i = 0; i < hiddenLayerWeightsMatrices.size(); i++){	
			Matrix dCost_dWeight = activationValues.get(i).copy().transpose();					
			dCost_dWeight = dCost_dWeight.times(deltas.get(i).copy());					
			dCost_dWeights.add(dCost_dWeight);
		}		
		return dCost_dWeights;
	}
	
	/*
	 * Find the cost of the current network with the given inputs and results
	 */
	private double cost(double[][] input, double output[][]){
		double c = 0;
		for(int i = 0; i < input.length; i++){
			double[] results = Forward(input[i]);
			
			for(int j = 0; j < results.length; j++){
				c += Math.pow(output[i][j] - results[j], 2);
			}
		}
		c /= 2.0;
		
		return c;
	}
	/*
	 * Print the matrix m 
	 */
	public void PrintMatrix(Matrix m){
		for(int i = 0; i < m.getRowDimension(); i++){
			for(int j = 0; j < m.getColumnDimension(); j++){
				System.out.print(m.get(i, j) + "  ");
			}
			System.out.println();
		}
		System.out.println();
	}
	/*
	 * Multiply each matrix spot in m1 with the same spot in m2. Return the resulting matrix.
	 * m1 and m2 must be the same size
	 */
	private Matrix ElementwiseMultiplication(Matrix m1, Matrix m2){
		Matrix m = new Matrix(m1.getRowDimension(), m1.getColumnDimension());
		for(int i = 0; i < m.getRowDimension(); i++){
			for(int j = 0; j < m.getColumnDimension(); j++){
				m.set(i, j, m1.get(i,j) * m2.get(i, j));
			}
		}
		return m;
	}
	
	/*
	 * Uses an arbitrary epsilon (e) to find the gradient of cost with respect to all weights
	 * in the network. 
	 * Useful for checking the matrix math done above.
	 */
	private ArrayList<Matrix> checkGradients(double[][] inputs, double[][] outputs){
		//Arbitrary change in the weight
		double e = 0.0001;
		//ArrayList of Matrices that'll hold the gradient
		ArrayList<Matrix> dCosts = new ArrayList<Matrix>();
		//Iterate through each weight matrix
		for(int i = 0; i <= numHiddenLayers; i++){
			//Create a new matrix of the size corresponding to the layer between i and i + 1
			Matrix m = new Matrix(hiddenLayerWeightsMatrices.get(i).getRowDimension(),
					hiddenLayerWeightsMatrices.get(i).getColumnDimension());
			//Iterate through the rows
			for(int j = 0; j < hiddenLayerWeightsMatrices.get(i).getRowDimension(); j++){
				//Iterate through the columns
				for(int k = 0; k < hiddenLayerWeightsMatrices.get(i).getColumnDimension(); k++){
					//Find the cost at the origial weight +- e, then find the derivative
					double originalValue = hiddenLayerWeightsMatrices.get(i).get(j, k);
					hiddenLayerWeightsMatrices.get(i).set(j, k, originalValue + e);					
					double c1 = cost(inputs, outputs);
					hiddenLayerWeightsMatrices.get(i).set(j, k, originalValue - e);
					double c2 = cost(inputs, outputs);
					hiddenLayerWeightsMatrices.get(i).set(j, k, originalValue);
					//Store that derivative in the matrix
					m.set(j, k, (c1 - c2) / (e * 2));
				}
			}
			//Add the matrix to the ArrayList
			dCosts.add(m);
		}
		//If all is well, this matrix should match the dCost_dWeights matrix in training.
		return dCosts;
	}
	
	/*
	 * Saves the current network configuration for future loading via XML 
	 */
	public void SaveConfiguration(String location){
		DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
		factory.setValidating(true);
		factory.setIgnoringElementContentWhitespace(true);
		try {
			DocumentBuilder builder = factory.newDocumentBuilder();

		    File file = new File("Configuration.xml");
		    Document doc = builder.parse(file);
		    
		} 
		catch(Exception e){
			
		}
	    
	}
	/*
	 * Loads a configuration from a saved file and returns the created network 
	 */
	public static NeuralNetwork LoadConfiguration(String location){
		NeuralNetwork n = new NeuralNetwork();
		DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
	    factory.setValidating(true);
	    factory.setIgnoringElementContentWhitespace(true);
	    DocumentBuilder builder;
		try {
			builder = factory.newDocumentBuilder();
		    try {
				Document doc = builder.parse(new File(location));
				
			} catch (Exception e) {
				e.printStackTrace();
			}
		    
		} catch (ParserConfigurationException e) {
			e.printStackTrace();
		}
		return n;
	}
	@Override
	public void run() {
		long startTime = System.nanoTime();
		int randomizeIndex = 0;
		iterations = 0;
		minCost = cost(inputs, correctResults);
		while(!stopThread){
			training = true;
			Matrix inputsMatrix = Matrix.constructWithCopy(inputs);

			Matrix correctResultsMatrix = Matrix.constructWithCopy(correctResults);			
			
			ArrayList<Matrix> valuesBeforeActivation = new ArrayList<Matrix>();
			ArrayList<Matrix> activationValues = new ArrayList<Matrix>();
			
			Matrix estimatedResultsMatrix = ForwardAllTraining(inputsMatrix, 
					valuesBeforeActivation, activationValues);			
			
			ArrayList<Matrix> deltas = populateDeltas(correctResultsMatrix, estimatedResultsMatrix,
					valuesBeforeActivation);
			
			ArrayList<Matrix> dCost_dWeights = costFunctionPrime(deltas, activationValues);
			//ArrayList<Matrix> dCostWithEpsilon = checkGradients(inputs, correctResults);

			for(int i = 0; i < hiddenLayerWeightsMatrices.size(); i++){
				Matrix dCost_dWeight = dCost_dWeights.get(i).copy();
				//Matrix dCostdEps = dCostWithEpsilon.get(i).copy();
				//PrintMatrix(dCost_dWeight);
				//PrintMatrix(dCostdEps);
				dCost_dWeight = dCost_dWeight.times(learningRate);
				//dCostdEps = dCostdEps.times(learningRate);
				
				hiddenLayerWeightsMatrices.set(i, 
						//hiddenLayerWeightsMatrices.get(i).minus(dCostdEps));
						hiddenLayerWeightsMatrices.get(i).minus(dCost_dWeight));
			}
			if(minCost > cost(inputs, correctResults)){
				minCost = cost(inputs, correctResults);
				minHiddenLayerWeights = (ArrayList<Matrix>)hiddenLayerWeightsMatrices.clone();
			}
			if(iterations % 5000 == 0)
				System.out.println("Iteration " + iterations + ": cost=" + cost(inputs, correctResults));
			iterations++;	
		}
		System.out.println("Training took " + 
				(System.nanoTime() - startTime) / 1000000000.0 + 
				"s with " + iterations + " iterations.");
		System.out.println("Resetting to lowest cost of " + minCost);
		hiddenLayerWeightsMatrices = (ArrayList<Matrix>)minHiddenLayerWeights.clone();
		stopThread = false;
		training = false;
	}
	public void Stop(){
		stopThread = true;
	}
}
