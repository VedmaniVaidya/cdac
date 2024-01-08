class Laptop{
	private int srno;
	private String make;
	private double cost;
	//No args constructor
	public Laptop() {
		srno=101;
		make="hp";
		cost=30000;
		System.out.println("Inside No argument constructor");
	}
	public Laptop(int srno, String make, double cost) {
		this.srno = srno;
		this.make = make;
		this.cost = cost;
	}
	public String toString()
	{
		return srno+" "+make+" "+ cost;
	}
	
	
}

public class Classes {

	public static void main(String[] args) {
		Laptop l1 = new Laptop();
		Laptop l2 = new Laptop(102,"dell",43000);
		Laptop l3 = new Laptop();
		System.out.println(l1);
		System.out.println(l2);
		System.out.println(l3);
	}

}
