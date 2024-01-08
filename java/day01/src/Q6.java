import java.util.Scanner;

public class Q6 {

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		System.out.println("Enter two numbers");
		float a = sc.nextFloat();
		float b = sc.nextFloat();
		System.out.println("1.addition\n"
				+ "2.Subtraction\n"
				+ "3.Multiplication\n"
				+ "4.Division\n"
				+ "choice:");
		int choice = sc.nextInt();
		switch (choice) {
		case 1: {
			float c =a+b;
			System.out.println("Addition:"+c);
		}
		case 2:{
			float c =a-b;
			System.out.println("Subtraction:"+c);
		}
		case 3:{
			float c =a*b;
			System.out.println("Multiplication:"+c);
		}
		case 4:{
			float c =a/b;
			System.out.println("Division:"+c);
		}
		}

	}
	
}
	