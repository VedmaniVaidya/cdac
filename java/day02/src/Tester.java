class Laptop1{
	private int srno;
	private String make;
	private double cost;
	
	public Laptop1(int srno, String make, double cost) {
		super();
		this.srno = srno;
		this.make = make;
		this.cost = cost;
	}

	public String toString() {
		return "Laptop [srno=" + srno + ", make=" + make + ", cost=" + cost + "]";
	}
	
}


public class Tester {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Laptop laptop1 = new Laptop(101,"hp",30000);
		System.out.print(laptop1);

	}

}

