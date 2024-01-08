public class Q8 {

	public static void main(String[] args) {
		int num = 153;
		int temp =num;
		int cube_sum=0,digit;
		while(num!=0) {
			digit = num%10;
			cube_sum = cube_sum+digit*digit*digit;
			num = num/10;
		}
		if (cube_sum==temp) {
			System.out.println("Number is Armstrong");
		}
		else
			System.out.println("Number is not Armstrong");
	}

}
