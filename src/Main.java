import java.util.ArrayList;
import java.util.Scanner;

public class Main {

	public static void main(String[] args) {
		
		NeuralNet SkyNet = new NeuralNet();
		
		SkyNet.SetInputSize(2);
		SkyNet.SetNumHiddenLayers(1);
		SkyNet.SetNodesInEachLayer(new int[] {3});
		SkyNet.SetOutputSize(1);
		SkyNet.Initialize();
		
		ArrayList<double[]> data = new ArrayList<double[]>();
		ArrayList<double[]> results = new ArrayList<double[]>();
		createDataAndResults1(data, results);
		double[][] d , r;
		d = new double[][] {{}};
		r = new double[][] {{}};
		SkyNet.Train(data.toArray(d), results.toArray(r));
		
		Scanner keys = new Scanner(System.in);
		boolean flag = true;
		while(flag){
			double x = keys.nextDouble();
			
			double y = keys.nextDouble();
			System.out.println(SkyNet.Forward(new double[] {x, y})[0]);
		}
		
	}
	
	private static void createDataAndResults1(ArrayList<double[]> data, ArrayList<double[]> results){
		//Lets work with a function y = x. Result is 0 if below line, 1 if above
		for(int i = 0; i < 100; i++){
			double x = Math.random() * 20 - 10;
			double y = Math.random() * 20 - 10;
			double result;
			if(y > x) result = 1;
			else result = 0;
			double[] d = {x, y};			
			data.add(d);
			double[] r = {result};
			results.add(r);			
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
