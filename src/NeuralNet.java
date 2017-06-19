import Jama.*;
import java.util.ArrayList;

public class NeuralNet {

	private int numInputNodes;
	private int numHiddenLayers;
	private int[] nodesPerLayer;
	private int numOutputNodes;
	
	private double learningRate = 0.1;
	
	ArrayList<ArrayList<ArrayList<Double>>> hiddenLayerWeights;
	
	public NeuralNet(){	
		
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
		hiddenLayerWeights = new ArrayList<ArrayList<ArrayList<Double>>>();
		for(int i = 0; i <= numHiddenLayers; i++){
			ArrayList<ArrayList<Double>> layer = new ArrayList<ArrayList<Double>>();
			int previousLayerSize, nextLayerSize;
			
			if(i == 0) previousLayerSize = numInputNodes;
			else previousLayerSize = nodesPerLayer[i - 1];
			
			if(i == numHiddenLayers) nextLayerSize = numOutputNodes;
			else nextLayerSize = nodesPerLayer[i];
			
			for(int j = 0; j < previousLayerSize; j++){
				ArrayList<Double> weights = new ArrayList<Double>();
				for(int k = 0; k < nextLayerSize; k++){
					weights.add(Math.random());
				}
				layer.add(weights);
			}
			hiddenLayerWeights.add(layer);
		}
	}
	private double ActivationFunction(double input){
		return 1 / (1 + Math.exp(-input));
	}
	private double ActivationFunctionPrime(double input){
		return (Math.exp(-input)) / (Math.pow(1 + Math.exp(-input), 2));
	}
	
	public void RandomizeWeights(){
		//Layer i
		for(int i = 0; i < hiddenLayerWeights.size(); i++){
			//From node j
			for(int j = 0; j < hiddenLayerWeights.get(i).size(); j++){
				//Going to node k
				for(int k = 0; k < hiddenLayerWeights.get(i).get(j).size(); k++){
					hiddenLayerWeights.get(i).get(j).set(k, Math.random());
				}
			}
		}
	}
	
	public double[] Solve(double[] inputs){
		double[] results = new double[numOutputNodes];
		for(int i = 0; i < numHiddenLayers; i++){
			
		}		
		return results;
	}
}
