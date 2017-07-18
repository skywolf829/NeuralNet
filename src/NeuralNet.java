import Jama.*;
import java.util.ArrayList;

public class NeuralNet {

	private NeuralNetController controller;
	private int numInputNodes;
	private int numHiddenLayers;
	private int[] nodesPerLayer;
	private int numOutputNodes;
	
	private double learningRate = 0.1;
	
	private ArrayList<Matrix> hiddenLayerWeightsMatrices;
	
	public NeuralNet(NeuralNetController c){	
		controller = c;
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
		controller.SetHiddenLayerWeights(hiddenLayerWeightsMatrices);
	}
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
		controller.SetHiddenLayerWeights(hiddenLayerWeightsMatrices);

	}
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
	
	private Matrix ForwardAll(Matrix inputs){
		Matrix results = inputs.copy();
		for(int i = 0; i <= numHiddenLayers; i++){
			//Get the weights for the current layer
			Matrix weights = hiddenLayerWeightsMatrices.get(i);
			//Right multiply the results by the weights
			results = results.times(weights);
			//If it's not the last layer (i.e. the output node(s)), then we need
			//	to apply the activation function on each matrix entry
			results = ActivationFunction(results);
		}	
		//Each row will represent the outputs for the corresponding inputs row
		return results;
	}
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
	public void Train(double[][] inputs, double[][] correctResults){
		
		
		int j = 0;
		while(cost(inputs, correctResults) > 0.01){
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
			System.out.println(cost(inputs, correctResults));
			j++;
			if(j >= 50000){
				RandomizeWeights();
				j = 0;
			}
		}
		controller.SetHiddenLayerWeights(hiddenLayerWeightsMatrices);
	}
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
	private ArrayList<Matrix> costFunctionPrime(ArrayList<Matrix> deltas, ArrayList<Matrix> activationValues){
		ArrayList<Matrix> dCost_dWeights = new ArrayList<Matrix>();
		for(int i = 0; i < hiddenLayerWeightsMatrices.size(); i++){	
			Matrix dCost_dWeight = activationValues.get(i).copy().transpose();					
			dCost_dWeight = dCost_dWeight.times(deltas.get(i).copy());					
			dCost_dWeights.add(dCost_dWeight);
		}		
		return dCost_dWeights;
	}
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
	public void PrintMatrix(Matrix m){
		for(int i = 0; i < m.getRowDimension(); i++){
			for(int j = 0; j < m.getColumnDimension(); j++){
				System.out.print(m.get(i, j) + "  ");
			}
			System.out.println();
		}
		System.out.println();
	}
	private Matrix ElementwiseMultiplication(Matrix m1, Matrix m2){
		Matrix m = new Matrix(m1.getRowDimension(), m1.getColumnDimension());
		for(int i = 0; i < m.getRowDimension(); i++){
			for(int j = 0; j < m.getColumnDimension(); j++){
				m.set(i, j, m1.get(i,j) * m2.get(i, j));
			}
		}
		return m;
	}
	private ArrayList<Matrix> checkGradients(double[][] inputs, double[][] outputs){
		double e = 0.0001;
		ArrayList<Matrix> dCosts = new ArrayList<Matrix>();
		for(int i = 0; i <= numHiddenLayers; i++){
			Matrix m = new Matrix(hiddenLayerWeightsMatrices.get(i).getRowDimension(),
					hiddenLayerWeightsMatrices.get(i).getColumnDimension());
			
			for(int j = 0; j < hiddenLayerWeightsMatrices.get(i).getRowDimension(); j++){
				//Iterate through the columns
				for(int k = 0; k < hiddenLayerWeightsMatrices.get(i).getColumnDimension(); k++){
					double originalValue = hiddenLayerWeightsMatrices.get(i).get(j, k);
					hiddenLayerWeightsMatrices.get(i).set(j, k, originalValue + e);					
					double c1 = cost(inputs, outputs);
					hiddenLayerWeightsMatrices.get(i).set(j, k, originalValue - e);
					double c2 = cost(inputs, outputs);
					hiddenLayerWeightsMatrices.get(i).set(j, k, originalValue);
					m.set(j, k, (c1 - c2) / (e * 2));
				}
			}
			dCosts.add(m);
		}
		return dCosts;
	}
}
