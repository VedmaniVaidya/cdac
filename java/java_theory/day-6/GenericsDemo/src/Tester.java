class Algo<G>
{
	public void Swap(G a, G b)
	{
		G temp;
		temp=a;
		a=b;
		b=temp;
		System.out.println(a+"  "+b);
	}
}
public class Tester {
	public static void main(String[] args) {
		Algo<Integer> a=new Algo<>();
		a.Swap(10, 20);
		
		Algo<String> b=new Algo<>();
		b.Swap("infosys", "tcs");
	}
}
