import java.util.*;

class A
{
	private int i;
	protected int j;
}
class B extends A
{
	public void fn()
	{
		j=10;
	}
}

public class Tester {
	public static void main(String[] args) {
			A a=new A();
			a.j=90;
			
	}

}
