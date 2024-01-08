import java.util.*;

class Product 
{
	private int pid;
	private String name;
	private double cost;
	
	public Product(int pid, String name, double cost) {
		this.pid = pid;
		this.name = name;
		this.cost = cost;
	}
	
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(cost);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		result = prime * result + ((name == null) ? 0 : name.hashCode());
		result = prime * result + pid;
		return result;
	}


	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Product other = (Product) obj;
		if (Double.doubleToLongBits(cost) != Double.doubleToLongBits(other.cost))
			return false;
		if (name == null) {
			if (other.name != null)
				return false;
		} else if (!name.equals(other.name))
			return false;
		if (pid != other.pid)
			return false;
		return true;
	}


	public void display()
	{
		System.out.println(pid+"  "+name+"  "+cost);
	}


	
}

public class Tester {

	public static void main(String[] args) 
	{
		Set<Product> s=new HashSet<>();
		s.add(new Product(51,"marker",60));
		s.add(new Product(52,"projector",40000));
		s.add(new Product(53,"keyboard",500));
		s.add(new Product(54,"keyboard",500));
		
		for(Product p:s)
		{
			p.display();
		}		
	}
}
