package Shapes;

import java.awt.Color;
import java.awt.Graphics2D;

public class Synapse {

	private Neuron start, end;
	private double weight;
		
	public Synapse(Neuron s, Neuron e){
		start = s;
		end = e;
	}
	public void SetWeight(double w){
		weight = w;
	}
	public void draw(Graphics2D g){
		g.setColor(Color.black);
		g.drawLine(start.GetX() + start.GetSize() / 2, 
				start.GetY()+ start.GetSize() / 2, 
				end.GetX() + end.GetSize() / 2, 
				end.GetY() + end.GetSize() / 2);
		
		String s = weight + "";
		s = s.substring(0, 4);
		g.drawChars(s.toCharArray(), 0, s.length(), 
				(end.GetX() - start.GetX()) / 2 + start.GetX() + start.GetSize() / 2 - 7, 
				(end.GetY() - start.GetY()) / 2  + start.GetY() + start.GetSize() / 2 - 4);
				
	}
}
