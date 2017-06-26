import java.util.ArrayList;
import java.util.Scanner;
import javax.swing.*;

import java.awt.*;

public class NeuralNetMain{

	public static void main(String[] args) {	
		SwingUtilities.invokeLater(new Runnable() {		            
	        @Override
	        public void run() {            
	        		NeuralNetController network = new NeuralNetController();	
	        }
		});   
		
	}

}
