package Shapes;

import java.awt.Color;
import java.awt.Graphics2D;

public class Neuron {
	private int xPos, yPos;
	private int size;
	
	public Neuron(int x, int y, int s){
		xPos = x;
		yPos = y;
		size = s;
	}
	public int GetX(){ return xPos; }
	public int GetY(){ return yPos; }
	public int GetSize(){ return size; }
	public void SetPos(int x, int y){ xPos = x; yPos = y; }
	public void draw(Graphics2D g){
		g.setColor(Color.blue);
		g.fillOval(xPos, yPos, size, size);
	}
}
