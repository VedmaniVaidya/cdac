class Stack<G>
{
	private int size;
	private G [] arr;
	private int top;
	
	public Stack(){
		size=5;
		arr=(G[])new Object[size];
		top=-1;
	}
	
	public Stack(int s){
		size=s;
		arr=(G[])new Object[size];
		top=-1;
	}
	
	public void push(G val)
	{
		if(!isFull())
			arr[++top]=val;
		else
			System.out.println("overflow!");
	}
	
	public G pop()
	{
		if(!isEmpty())
			return arr[top--];
		else
		{
			System.out.println("underflow!");
			return null;
		}
	}
	
	public boolean isFull()
	{
		return top==size-1;
	}
	public boolean isEmpty()
	{
		return top==-1;
	}
}

public class Tester {
	public static void main(String[] args) {
			Stack<Integer> s1=new Stack<>();
			s1.push(10);
			s1.push(20);
			s1.push(30);
			s1.push(40);
			s1.push(50);
			System.out.println(s1.pop());
			System.out.println(s1.pop());
			System.out.println(s1.pop());
			System.out.println(s1.pop());
			System.out.println(s1.pop());
			System.out.println();
			
			Stack<String> s2=new Stack<>();
			s2.push("Infy");
			s2.push("Wipro");
			s2.push("TCS");
			System.out.println(s2.pop());
			System.out.println(s2.pop());
			System.out.println(s2.pop());
	}
}
