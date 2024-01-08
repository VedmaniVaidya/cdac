
public class Q12 {

	public static void main(String[] args) {
		int num=5;
		int sum=1;
		for (int i=2;i<(num/2)+1;i++) {
			if (num%i==0) {
				sum +=i;
			}
		}
		if (sum==num) 
			System.out.println("Perfect Number");
		else
				System.out.println("Not a Perfect Number");
	}

}
