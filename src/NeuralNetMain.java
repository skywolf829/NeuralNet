import javax.swing.*;

public class NeuralNetMain{

	public static void main(String[] args) {	
		SwingUtilities.invokeLater(new Runnable() {		            
	        @Override
	        public void run() {            
	        		@SuppressWarnings("unused")
					NeuralNetController network = new NeuralNetController();	
	        }
		});   
		
	}

}
