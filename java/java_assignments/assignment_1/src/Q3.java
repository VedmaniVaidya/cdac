import java.util.Scanner;

public class Q3 {

	public static void main(String[] args) {
		System.out.print("enter two variables:");
		Scanner sc = new Scanner(System.in);
		int num1 = sc.nextInt();
		int num2 = sc.nextInt();
		
		int temp = num1;
		num1 = num2;
		num2 = temp;
		
//		num1 = num1+num2;
//		num2 = num1-num2;
//		num1 = num1-num2;
		
		
		System.out.println(" num1 :"+num1+" num2: "+num2);
		   
	}
	
}