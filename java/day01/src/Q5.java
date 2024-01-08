import java.util.Scanner;
public class Q5 {

	public static void main(String[] args) {
		
		int basic_salary,total_sales;
        float commission_earned=0,net_salary;
        
		Scanner sc = new Scanner(System.in);
		
		System.out.println("enter basic_salary ");
		basic_salary = sc.nextInt();
		System.out.println("total_sales : ");
		total_sales = sc.nextInt();	
		
		if(total_sales>500 && total_sales<=7500) {
			commission_earned = total_sales*(0.03f);
		}
		else if(total_sales>=7501 && total_sales<=10500){
			commission_earned = total_sales*(0.08f);
			
		}
		else if(total_sales>=10501 && total_sales<=15000){
			commission_earned = total_sales*(0.11f);
			
		}
		else if(total_sales>15000){
			commission_earned = total_sales*(0.15f);
			
		}
		
		net_salary = basic_salary+commission_earned;
		
		System.out.println(" net salary is "+ net_salary);;
		System.out.println("commission earned is "+commission_earned);
		
		
	}

}


