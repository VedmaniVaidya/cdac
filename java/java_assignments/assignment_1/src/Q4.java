import java.util.Scanner;

public class Q4 {
	
	public static void main(String[] args) {
		float basic_salary;
		System.out.println("Enter basic salary amount:");
		Scanner sc = new Scanner(System.in);
		basic_salary = sc.nextFloat();
		float hra = 15.0f/100*basic_salary;
		float da = 30.0f/100*basic_salary;
		float pf = 12.5f/100*basic_salary;
		float net_salary = basic_salary+hra+da-pf;
		System.out.println("Net salary:"+net_salary);
		
	}
}
