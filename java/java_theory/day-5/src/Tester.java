class Circle implements Cloneable
{
	private int radius;
	
	public Circle()
	{
		radius=1;
	}
	public Circle(int r)
	{
		radius=r;
	}
	public double calcArea()
	{
		return 3.142*radius*radius;
	}
	public double calcPeri()
	{
		return 2*3.142*radius;
	}
	
	 public Object clone() throws CloneNotSupportedException 
	 {
		 return super.clone();
	 }
}

public class Tester {

	public static void main(String[] args) throws CloneNotSupportedException {
		
		Circle c1=new Circle(11);
		
		Circle c2=(Circle)c1.clone();
		
		System.out.println(c2.calcArea());
	}
}
