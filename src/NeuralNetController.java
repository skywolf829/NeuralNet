import java.util.ArrayList;
import java.util.Scanner;

import Jama.Matrix;

public class NeuralNetController {
	NeuralNet network;
	NeuralNetView view;
	
	public NeuralNetController(){	
		network = new NeuralNet(this);		
		view = new NeuralNetView(this);
		view.setSize(500,  400);
	}
	
	public void Initialize(){		
		network.Initialize();
	}
	public void BeginTraining(){
		ArrayList<double[]> data = new ArrayList<double[]>();
		ArrayList<double[]> results = new ArrayList<double[]>();
		createDataAndResults1(data, results);
		double[][] d , r;
		d = new double[][] {{}};
		r = new double[][] {{}};
		network.Train(data.toArray(d), results.toArray(r));
		
		Scanner keys = new Scanner(System.in);
		boolean flag = true;
		while(flag){
			double x = keys.nextDouble();			
			double y = keys.nextDouble();
			double result[] = network.Forward(new double[] {x, y});			
			System.out.println(result[0] + " " + result[1]);
		}
	}
	public void SetNumInputNodes(int n){
		network.SetInputSize(n);
		view.SetNumInputNodes(n);
	}
	public void SetNodesPerLayer(int[] n){
		network.SetNumHiddenLayers(n.length);
		network.SetNodesInEachLayer(n);
		view.SetNodesPerLayer(n);
	}
	public void SetNumOutputNodes(int n){
		network.SetOutputSize(n);
		view.SetNumOutputNodes(n);
	}
	public void SetHiddenLayerWeights(ArrayList<Matrix> h){
		view.SetHiddenLayerWeights(h);
	}
	private static void createDataAndResults1(ArrayList<double[]> data, ArrayList<double[]> results){
		//Lets work with a function y = x. Result is 0 if below line, 1 if above
		for(int i = 0; i < 1000; i++){
			double x = Math.random() * 20 - 10;
			double y = Math.random() * 20 - 10;
			double result[];
			if(y > x) result = new double[]{1, 0};
			else result = new double[]{0, 1};
			double[] d = {x, y};			
			data.add(d);
			results.add(result);			
		}			
	}
	private static void createDataAndResults2(ArrayList<double[]> data, ArrayList<double[]> results){
		//Lets work with a function y = x. Result is 0 if below line, 1 if above
		for(int i = 0; i < 3000; i++){
			double x = Math.random() * 20 - 10;
			double result = x;
			double[] d = {x};			
			data.add(d);
			double[] r = {result};
			results.add(r);			
		}			
	}
}
