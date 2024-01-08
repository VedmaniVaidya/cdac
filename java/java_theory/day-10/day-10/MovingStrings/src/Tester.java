import java.awt.*;

class MovingStrings extends Frame implements Runnable
{
	Thread t1, t2;
	int x1,x2;
	
	public MovingStrings(){
		x1=200;
		x2=400;
		t1=new Thread(this,"dai");
		t2=new Thread(this,"dac");
		t1.start();
		t2.start();
	}
	
	 public void paint(Graphics g) {
		 g.setColor(Color.BLUE);
		 g.setFont(new Font("Tahoma",50,50));
		 g.drawString("PG-DAI", x1, 200);
		 g.setColor(Color.GREEN);
		 g.drawString("PG-DAC", x2, 400);
	 }

	public void run() {
		while(true)
		{
			if(Thread.currentThread()==t1)
			{
				x1++;
				if(x1==this.getWidth())
					x1=-25;
				try {
					Thread.sleep(30);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			else if(Thread.currentThread()==t2)
			{
				x2--;
				if(x2==0)
					x2=this.getWidth();
				try {
					Thread.sleep(30);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			repaint();
		}
	}
	
}

public class Tester {

	public static void main(String[] args) 
	{
		MovingStrings fr=new MovingStrings();
		fr.setSize(600, 600);
		fr.setVisible(true);
	}
}
