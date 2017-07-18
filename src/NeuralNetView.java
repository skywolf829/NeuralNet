import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.util.ArrayList;

import javax.swing.*;

import Jama.Matrix;
import Shapes.Neuron;
import Shapes.Synapse;

public class NeuralNetView extends JFrame implements ActionListener, ComponentListener{
	private NeuralNetController controller;
	private boolean creatingNetwork;
	private int numInputNodes;
	private int numHiddenLayers;
	private int[] nodesPerLayer;
	private int numOutputNodes;
	private ArrayList<Matrix> hiddenLayerWeightsMatrices;
	
	private ArrayList<ArrayList<Neuron>> neurons;
	private ArrayList<ArrayList<ArrayList<Synapse>>> synapses;
	
	private DrawingPanel panel;
	
	private JLabel input, hidden, output, training;
	private JTextField numInputNodesTextField, hiddenLayersTextField, outputNodesTextField,
		trainingFileTextField;
	private ArrayList<JTextField> inputTextFields, outputTextFields;
	private JButton createNeuralNetButton, trainButton, computeButton, resetWeightsButton;
	
	private int buttonHeight = 50,
			buttonWidth = 150,
			buttonPanelSpacing = 30,
			textAreaWidth = 330,
			textAreaHeight = 300,
			textAreaXOffset = 20,
			bufferSpace = 100, 
			neuronSize = 50, 
			distanceBetweenNeurons = 80;
	
	private int widthNeeded, extraWidth, heightNeeded, extraHeight;
	
	
	public NeuralNetView(NeuralNetController c){
		super();
		controller = c;
		this.addComponentListener(this);
		neurons = new ArrayList<ArrayList<Neuron>>();
		synapses = new ArrayList<ArrayList<ArrayList<Synapse>>>();
		
        panel = new DrawingPanel(this);
        add(panel);
        
        setTitle("Neural network");
        setLocationRelativeTo(null);  
        setFocusable(true);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(null);
        setSize(500, 400);
        
        pack();
        setVisible(true);
	}
	public void SetNumInputNodes(int n){
		numInputNodes = n;
	}
	public void SetNodesPerLayer(int[] n){
		nodesPerLayer = n;
	}
	public void SetNumOutputNodes(int n){
		numOutputNodes = n;
	}
	public void SetHiddenLayerWeights(ArrayList<Matrix> h){
		for(int i = 0; i < h.size(); i++){
			for(int j = 0; j < h.get(i).getRowDimension(); j++){
				for(int k = 0; k < h.get(i).getColumnDimension(); k++){
					synapses.get(i).get(j).get(k).SetWeight(h.get(i).get(j, k));
				}
			}
		}
		panel.repaint();
	}
	private void createNeuralNetButtonPressed(){
		int inputs = Integer.parseInt(numInputNodesTextField.getText());
		int outputs = Integer.parseInt(outputNodesTextField.getText());
		String hiddenString = hiddenLayersTextField.getText();
		int[] hidden = hiddenLayersInputParser(hiddenString);
		controller.SetNodesPerLayer(hidden);
		controller.SetNumInputNodes(inputs);
		controller.SetNumOutputNodes(outputs);
		populateNeurons();
		populateSynapses();
		clearOldInputsAndOutputs();
		populateInputTextFields();
		populateOutputTextFields();
		controller.Initialize();
		panel.repositionItems();		

	}
	private void trainButtonPressed(){
		training.setText("Training...");
		panel.repositionItems();
		controller.TrainOnSineFunction();
		training.setText("Training complete!");
		panel.repositionItems();
		this.setSize(new Dimension(this.getWidth() + 1, this.getHeight()));
		
	}
	private void computeButtonPressed(){
		double[] inputs = new double[inputTextFields.size()];
		for(int i = 0; i < inputTextFields.size(); i++){
			inputs[i] = Double.parseDouble(inputTextFields.get(i).getText());
		}
		double[] results = controller.Forward(inputs);
		for(int i = 0; i < results.length; i++){
			outputTextFields.get(i).setText(((int)(results[i] * 10000)) / 10000.0 + "");
			
		}
		panel.repositionItems();
	}
	private void resetWeightsButtonPressed(){
		controller.ResetWeights();
		panel.repositionItems();
	}
	private void clearOldInputsAndOutputs(){
		if(inputTextFields != null){
			while(inputTextFields.size() > 0){
				panel.remove(inputTextFields.get(0));
				inputTextFields.remove(0);
			}
		}
		if(outputTextFields != null){
			while(outputTextFields.size() > 0){
				panel.remove(outputTextFields.get(0));
				outputTextFields.remove(0);
			}
		}
		if(computeButton != null)
			panel.remove(computeButton);
	}
	private void populateInputTextFields(){
		inputTextFields = new ArrayList<JTextField>();		
		
		for(int i = 0; i < numInputNodes; i++){
			JTextField in = new JTextField();
			in.setEditable(true);
			in.setColumns(4);
			in.setText("");in.addActionListener(this);
			inputTextFields.add(in);
			panel.add(in);
		}
		computeButton = new JButton("Compute output");
		computeButton.addActionListener(this);
		panel.add(computeButton);
	}
	private void populateOutputTextFields(){
		outputTextFields = new ArrayList<JTextField>();		
		for(int i = 0; i < numOutputNodes; i++){
			JTextField out = new JTextField();
			out.setEditable(false);
			out.setColumns(4);
			out.setText("");
			outputTextFields.add(out);
			panel.add(out);
		}
	}
	private void populateNeurons(){
		int currentLayer = 0;
		int j = 0;
		neurons = new ArrayList<ArrayList<Neuron>>();
		ArrayList<Neuron> layer = new ArrayList<Neuron>();
		for(int i = 0; i < numInputNodes + numOutputNodes + numHiddenNodes(nodesPerLayer); i++){
			if(i < numInputNodes){
				layer.add(new Neuron(0, 0, neuronSize));
				if(i + 1 == numInputNodes) {
					neurons.add(layer);
					layer = new ArrayList<Neuron>();
				}
			}
			else if(i >= numInputNodes && currentLayer < nodesPerLayer.length){
				if(j < nodesPerLayer[currentLayer]){
					layer.add(new Neuron(0, 0, neuronSize));
					j++;
					if(j == nodesPerLayer[currentLayer]){
						neurons.add(layer);
						layer = new ArrayList<Neuron>();
						j = 0;
						currentLayer++;
					}
				}
			}
			else{
				layer.add(new Neuron(0,0, neuronSize));
			}			
		}
		neurons.add(layer);	
		
	}
	private void populateSynapses(){
		synapses = new ArrayList<ArrayList<ArrayList<Synapse>>>();
		//layer i
		for(int i = 0; i < neurons.size() - 1; i++){
			//from the jth thing in i
			ArrayList<ArrayList<Synapse>> layer = new ArrayList<ArrayList<Synapse>>();
			for(int j = 0; j < neurons.get(i).size(); j++){
				ArrayList<Synapse> list = new ArrayList<Synapse>();
				//to the kth thing in i + 1
				for(int k = 0; k < neurons.get(i + 1).size(); k++){
					list.add(new Synapse(neurons.get(i).get(j), neurons.get(i + 1).get(k)));
				}
				layer.add(list);
			}
			synapses.add(layer);
		}
	}
	private int numHiddenNodes(int[] hidden){
		int sum = 0;
		for(int i = 0; i < hidden.length; i++){
			sum += hidden[i];
		}
		return sum;
	}
	private int[] hiddenLayersInputParser(String s){
		ArrayList<Integer> list = new ArrayList<Integer>();

		char[] array = s.toCharArray();
		int spot = 1;
		int current = 0;
		for(int i = 0; i < array.length; i++){
			if(Character.isDigit(array[i]) || array[i] == ' '){
				if(Character.isDigit(array[i])){
					current *= 10;
					current += Integer.parseInt(array[i] + "");
				}
				else{
					list.add(current);
					current = 0;
				}
			}
			else{
				//error
			}
		}
		list.add(current);
		int[] done = new int[list.size()];
		for(int i = 0; i < list.size(); i++){
			done[i] = list.get(i);
		}

		return done;
	}
	
	public class DrawingPanel extends JPanel{
		private NeuralNetView frame;
				
		public DrawingPanel(NeuralNetView f){
			super();
			frame = f;
			
			input = new JLabel("Number of input nodes");
			numInputNodesTextField = new JTextField();
			numInputNodesTextField.setEditable(true);
			numInputNodesTextField.setColumns(4);
			numInputNodesTextField.setText("2");
			numInputNodesTextField.addActionListener(frame);
			
			hidden = new JLabel("Number of hidden nodes in each layer, separated by a space");
			hiddenLayersTextField = new JTextField();
			hiddenLayersTextField.setEditable(true);
			hiddenLayersTextField.setColumns(4);
			hiddenLayersTextField.setText("3");
			hiddenLayersTextField.addActionListener(frame);
			
			output = new JLabel("Number of output nodes");
			outputNodesTextField = new JTextField();
			outputNodesTextField.setEditable(true);
			outputNodesTextField.setColumns(4);
			outputNodesTextField.setText("2");
			outputNodesTextField.addActionListener(frame);
			
			createNeuralNetButton = new JButton("Create");
			createNeuralNetButton.addActionListener(frame);
			
			training = new JLabel("Enter the training data file");
			trainingFileTextField = new JTextField();
			trainingFileTextField.setEditable(true);
			trainingFileTextField.setColumns(8);
			trainingFileTextField.setText("Filename.txt");
			trainingFileTextField.addActionListener(frame);
			
			trainButton = new JButton("Train network");
			trainButton.addActionListener(frame);
			
			resetWeightsButton = new JButton("Reset weights");
			resetWeightsButton.addActionListener(frame);
			
			this.add(input);
			this.add(numInputNodesTextField);
			this.add(hidden);
			this.add(hiddenLayersTextField);
			this.add(output);
			this.add(outputNodesTextField);
			this.add(createNeuralNetButton);
			this.add(training);
			this.add(trainingFileTextField);
			this.add(trainButton);
			this.add(resetWeightsButton);
			
			pack();
		}
		@Override
	    public void paintComponent(Graphics g) { 
	        super.paintComponent(g);
	        paint((Graphics2D)g);
	    }

		public void paint(Graphics2D g){
	        for(int i = 0; i < synapses.size(); i++){
	        	for(int j = 0; j < synapses.get(i).size(); j++){
	        		for(int k = 0; k < synapses.get(i).get(j).size(); k++){
	        			synapses.get(i).get(j).get(k).draw(g);
	        		}
	        	}
	        }
	        for(int i = 0; i < neurons.size(); i++){
	        	for(int j = 0; j < neurons.get(i).size(); j++){
	        		neurons.get(i).get(j).draw(g);
	        	}
	        }
		}
		private int nodesHeightNeeded(){
			if(neurons == null) return 0;
			int maxNodes = numInputNodes;
			for(int i = 0; i < neurons.size(); i++){
				if(maxNodes < neurons.get(i).size()) maxNodes = neurons.get(i).size();
			}
			if(numOutputNodes > maxNodes) maxNodes = numOutputNodes;
			int height = (int)(maxNodes * neuronSize + (maxNodes) * distanceBetweenNeurons*0.5);
			return height;
			
		}
		private int nodesWidthNeeded(){
			if(nodesPerLayer == null) return 0;
			int w = 2 + nodesPerLayer.length;
			w = (int)(w * neuronSize + (w - 1) * distanceBetweenNeurons);
			return w;
		}
		public int GetRequiredHeight(){
			if(nodesHeightNeeded() > textAreaHeight) heightNeeded = nodesHeightNeeded() + bufferSpace;
			else heightNeeded = textAreaHeight + bufferSpace;
			extraHeight = this.getHeight() - heightNeeded;
			if(extraHeight < 0) extraHeight = 0;
			return heightNeeded;
		}
		public int GetRequiredWidth(){
			widthNeeded = textAreaWidth + bufferSpace + nodesWidthNeeded() + bufferSpace;
			extraWidth = this.getWidth() - widthNeeded;
			if(extraWidth < 0) extraWidth = 0;
			return widthNeeded;
		}
		public void repositionItems(){
			GetRequiredWidth();
			GetRequiredHeight();
			
			input.setLocation(extraWidth / 2, 0 + extraHeight / 2);
			numInputNodesTextField.setLocation(
					//(input.getWidth()-numInputNodesTextField.getWidth()+extraWidth) / 2, 
					textAreaXOffset + extraWidth / 2,
					input.getY() + input.getHeight());
			
			hidden.setLocation(0 + extraWidth / 2, 
					numInputNodesTextField.getY() + numInputNodesTextField.getHeight());
			hiddenLayersTextField.setLocation(
					//(hidden.getWidth()-hiddenLayersTextField.getWidth()+extraWidth) / 2, 
					textAreaXOffset + extraWidth / 2, 
					hidden.getY() + hidden.getHeight());
			
			output.setLocation(0 + extraWidth / 2, 
					hiddenLayersTextField.getY() + hiddenLayersTextField.getHeight());
			outputNodesTextField.setLocation(
					//(output.getWidth()-outputNodesTextField.getWidth()+extraWidth) / 2, 
					textAreaXOffset + extraWidth / 2,
					output.getY() + output.getHeight());
			
			createNeuralNetButton.setLocation(textAreaXOffset - 13 + extraWidth / 2, 
					outputNodesTextField.getY() + outputNodesTextField.getHeight());
			
			training.setLocation(0 + extraWidth / 2,
					createNeuralNetButton.getY() + createNeuralNetButton.getHeight() + bufferSpace / 2);
			trainingFileTextField.setLocation(textAreaXOffset + extraWidth / 2,
					training.getY() + training.getHeight());
			trainButton.setLocation(textAreaXOffset - 13 + extraWidth / 2,
					trainingFileTextField.getY() + trainingFileTextField.getHeight());
			resetWeightsButton.setLocation(trainButton.getX() + trainButton.getWidth(),
					trainButton.getY());
			
			for(int i = 0; neurons != null && i < neurons.size(); i++){
				for(int j = 0; j < neurons.get(i).size(); j++){
					neurons.get(i).get(j).SetPos(
							extraWidth / 2 + textAreaWidth + bufferSpace + i * (neuronSize + distanceBetweenNeurons),
							(int)(((j+1) / ((double)neurons.get(i).size() + 1)) * nodesHeightNeeded() + extraHeight / 2));
				}
			}
 			for(int i = 0; inputTextFields != null && i < inputTextFields.size(); i++){
				inputTextFields.get(i).setLocation(neurons.get(0).get(i).GetX() - 60,
						neurons.get(0).get(i).GetY() + neuronSize / 4);
			}
			for(int i = 0; outputTextFields != null && i < outputTextFields.size(); i++){
				outputTextFields.get(i).setLocation(neurons.get(neurons.size()-1).get(i).GetX() + neuronSize,
						neurons.get(neurons.size() - 1).get(i).GetY() + neuronSize / 4);
			}
			if(inputTextFields != null){
				computeButton.setLocation(inputTextFields.get(inputTextFields.size() - 1).getX() - 40,
						inputTextFields.get(inputTextFields.size() - 1).getY() + 40);
			}
			repaint();
		}
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if(e.getSource().equals(createNeuralNetButton)){
			createNeuralNetButtonPressed();
		}
		else if(e.getSource().equals(trainButton)){
			trainButtonPressed();
		}
		else if(e.getSource().equals(computeButton)){
			computeButtonPressed();
		}
		else if(e.getSource().equals(resetWeightsButton)){
			resetWeightsButtonPressed();
		}
	}
	@Override
	public void componentResized(ComponentEvent e) {
		panel.setSize(this.getSize());		
		panel.repositionItems();
	}
	@Override
	public void componentMoved(ComponentEvent e) {
		
	}
	@Override
	public void componentShown(ComponentEvent e) {
			
	}
	@Override
	public void componentHidden(ComponentEvent e) {
				
	}
}
