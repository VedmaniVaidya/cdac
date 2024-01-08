import java.util.Scanner;

public class Q9 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int prime = 0;
		System.out.println("enter a number");
		Scanner sc = new Scanner(System.in);
		prime = sc.nextInt();
		for(int i = 1;i<=prime/2;i++) {
			if(i!=1 && prime%i == 0) {
				System.out.println("not a prime");
				break;
			}
			else if(i==prime/2) {
				System.out.println("it is a prime number");
			}
		}

	}

}
