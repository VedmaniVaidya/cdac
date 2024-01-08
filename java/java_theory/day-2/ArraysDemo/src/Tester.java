import java.util.Scanner;

public class Tester {	
	public static void main(String[] args) 
	{
		int temp;
		Scanner sc=new Scanner(System.in);
		System.out.println("Enter array size");
		int [] a= new int[sc.nextInt()];
		
		System.out.println("Enter array elements:");
		for(int i=0; i<a.length; i++)
			a[i]=sc.nextInt();
		
		for(int i=0; i<a.length/2; i++)
		{
			temp=a[i];
			a[i]=a[a.length-i-1];
			a[a.length-i-1]=temp;
		}
		
		for(int val:a)
			System.out.print(val+"  ");
		System.out.println();
	}
}






