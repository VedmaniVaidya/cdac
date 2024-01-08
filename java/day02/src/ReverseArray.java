import java.util.Scanner;

public class ReverseArray {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println("enter length of array");
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();
        int[] num = new int[n];	
        for(int val:num) {
        	System.out.print(val + " ");
        }
        
        for (int i=0;i<num.length;i++) {
        	num[i] = sc.nextInt();
        }
        
        System.out.println("before reverse");
        for(int val:num) {
        	System.out.print(val+" ");
        }
        System.out.println();
        
        for (int i =0;i<n/2;i++) {
        	int temp = num[i];
        	num[i] = num[n-i-1];
        	num[n-1-i] = temp;
        }
        
        System.out.println("After reverse");
        for(int val:num) {
        	System.out.print(val + " ");
        }

	}

}
