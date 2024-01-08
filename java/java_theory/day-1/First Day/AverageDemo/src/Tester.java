import java.util.Scanner;

public class Tester {
	public static void main(String[] args) 
	{
		int num=11;
		if(num%2==0)						//if-else syntax
			System.out.println("Even..");
		else 
			System.out.println("Odd..");
		
		int n=5, cnt=1;
		while(cnt<=10)						//print table using while
		{
			System.out.print(n*cnt+"  ");
			cnt++;
		}
		System.out.println();
		
		for(int i=1; i<=10; i++)			//print table using for
			System.out.print(n*i+"  ");
	}

}


