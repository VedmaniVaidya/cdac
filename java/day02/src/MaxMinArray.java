import java.util.Scanner;

public class MaxMinArray {

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		int []arr = new int[10];
		for(int i=0;i<arr.length;i++) {
			arr[i] = sc.nextInt();
		}
		
		for(int val:arr)
			System.out.print(val+" ");
		System.out.println();
		
		int max = Integer.MIN_VALUE;
		int min = Integer.MAX_VALUE;
		
		for(int val:arr) {
//			max = Math.max(max, val);
//			min = Math.min(val, min);
			if(val>max) {
				max = val;
			}
			if(val<min) {
				min = val;
			}
		}
		System.out.println("Maximum:"+max);
		System.out.println("Minimum:"+min);
	}

}

