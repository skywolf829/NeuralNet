import java.util.ArrayList;

import Jama.Matrix;

public class NeuralNetController {
	NeuralNetwork network;
	NeuralNetView view;
	
	public NeuralNetController(){	
		network = new NeuralNetwork();		
		view = new NeuralNetView(this);
		view.setSize(800,  400);
		UpdateView();
	}
	
	public void Initialize(){		
		network.Initialize();
		UpdateView();
	}
	public double[] Forward(double[] inputs){
		return network.Forward(inputs);
	}
	public boolean IsTraining(){
		return network.IsTraining();
	}
	public void StopTraining(){
		network.Stop();
		UpdateView();
	}
	public void TrainOnSineFunction(){
		ArrayList<double[]> data = new ArrayList<double[]>();
		ArrayList<double[]> results = new ArrayList<double[]>();
		for(int j = 0; j < 1; j++){
			for(int i = 0; i < 100; i++){
				double x = Math.random();
				double y = Math.sin(x);
				data.add(new double[] {x});
				results.add(new double[] {y});
			}
			double[][] d , r;
			d = new double[][] {{}};
			r = new double[][] {{}};
			network.Train(data.toArray(d), results.toArray(r));
		}
		UpdateView();
	}
	public void TrainOnXSquared(){
		ArrayList<double[]> data = new ArrayList<double[]>();
		ArrayList<double[]> results = new ArrayList<double[]>();
		for(int j = 0; j < 1; j++){
			for(int i = 0; i < 100; i++){
				double x = Math.random();
				double y = x * x;
				data.add(new double[] {x});
				results.add(new double[] {y});
			}
			double[][] d , r;
			d = new double[][] {{}};
			r = new double[][] {{}};
			network.Train(data.toArray(d), results.toArray(r));
		}
		UpdateView();
	}
	public void ResetWeights(){
		network.RandomizeWeights();
		UpdateView();
	}
	public void SetNumInputNodes(int n){
		network.SetInputSize(n);
		UpdateView();
	}
	public void SetNodesPerLayer(int[] n){
		network.SetNumHiddenLayers(n.length);
		network.SetNodesInEachLayer(n);
		UpdateView();
	}
	public void SetNumOutputNodes(int n){
		network.SetOutputSize(n);
		UpdateView();
	}
	public void SetHiddenLayerWeights(ArrayList<Matrix> w){
		network.SetHiddenLayerWeights(w);
		UpdateView();
	}
	public String TrainingInfo(){
		return "Iteration " + network.GetCurrentIteration() + ": cost=" + network.GetCurrentCost();
	}
	public void UpdateView(){
		view.SetNumInputNodes(network.GetNumInputNodes());
		view.SetNumOutputNodes(network.GetNumOutputNodes());
		view.SetNodesPerLayer(network.GetNodesPerLayer());
		if(network.GetWeights() != null)
			view.SetHiddenLayerWeights(network.GetWeights());	
		
		view.createNeuralNetButton.setEnabled(!network.IsTraining());
		view.trainButton.setEnabled(!network.IsTraining() && network.IsInitialized());
		view.stopTrainingButton.setEnabled(network.IsTraining());
		view.resetWeightsButton.setEnabled(network.IsInitialized());
		
		
	}
}
